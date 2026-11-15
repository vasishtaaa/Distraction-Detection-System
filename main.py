"""
Real-Time Distraction Detection System (DDS)
Desktop Application Pipeline integrating OpenCV, Google MediaPipe, and Ultralytics YOLOv8.
"""

import os
import sys
import time
import argparse
from datetime import datetime
import cv2

os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "."), "matplotlib")

import config
from camera import ThreadedCamera
from face_mesh_detector import DriverFaceAnalyzer
from yolo_detector import DistractionObjectDetector
from alert_system import AlertManager, AlertType
from analytics import AttentionTelemetry
from ui_overlay import HUDOverlay

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Distraction Detection System (DDS)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: camera index (e.g. 0) or path to video file (e.g. video.mp4)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=config.DEFAULT_MODE,
        choices=[config.MODE_GENERAL, config.MODE_STUDY, config.MODE_DRIVING],
        help="Operational mode: GENERAL, STUDY, or DRIVING"
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Disable YOLOv8 object detection (face landmark analysis only)"
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="Start with audio alerts muted"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI window (for benchmarking / automated testing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to record HUD-annotated session to video file"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before exiting"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 65)
    print(f"  DISTRACTION DETECTION SYSTEM (DDS) | MODE: {args.mode.upper()}")
    print("  OpenCV  |  Google MediaPipe  |  Ultralytics YOLOv8")
    print("=" * 65)

    # 1. Initialize Threaded Video Capture
    camera = ThreadedCamera(source=args.source, loop_video=True)
    if not camera.is_opened:
        print(f"[ERROR] Could not initialize video source: {args.source}")
        sys.exit(1)

    # 2. Initialize Detection Modules
    print("[INIT] Loading MediaPipe 3D Face Landmarker...")
    face_analyzer = DriverFaceAnalyzer(model_path=config.FACE_LANDMARKER_MODEL)

    yolo_detector = None
    if not args.no_yolo:
        print("[INIT] Loading YOLOv8 Distraction Detection Model...")
        yolo_detector = DistractionObjectDetector(model_path=config.YOLO_MODEL_PATH)

    alert_manager = AlertManager(enabled=not args.mute)
    telemetry = AttentionTelemetry(mode=args.mode.upper())
    hud = HUDOverlay()

    # 3. Optional Video Output Writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_w, out_h = camera.width, camera.height
        writer = cv2.VideoWriter(args.output, fourcc, 30.0, (out_w, out_h))
        print(f"[RECORD] Saving annotated session -> {args.output}")

    MODES = [config.MODE_GENERAL, config.MODE_STUDY, config.MODE_DRIVING]
    current_mode_idx = MODES.index(args.mode.upper()) if args.mode.upper() in MODES else 0

    print("[SUCCESS] All systems active. Commencing real-time distraction detection.")
    print("Controls: [TAB / 1-3] Switch Mode | [Q] Quit | [M] Audio Mute | [R] Reset | [S] Snapshot | [D] 3D Mesh")
    print("-" * 65)

    prev_time = time.time()
    fps = 30.0
    frame_idx = 0

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break

            # Rolling FPS
            curr_time = time.time()
            delta = curr_time - prev_time
            prev_time = curr_time
            if delta > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / delta)

            # 1. Facial Landmark & Pose Analysis
            face_data = face_analyzer.analyze_frame(frame)
            face_bbox = face_data["face_bbox"] if face_data else None

            # 2. YOLOv8 Object Detection (Cell Phone, Objects)
            yolo_distracted = False
            detections = []
            if yolo_detector is not None:
                yolo_distracted, detections = yolo_detector.detect(frame, face_bbox=face_bbox)

            # 3. Rule Evaluation
            if face_data and face_data["is_eyes_closed"]:
                alert_manager.trigger(AlertType.DROWSINESS)
            elif yolo_distracted:
                alert_manager.trigger(AlertType.PHONE)
            elif face_data and face_data["is_head_distracted"]:
                alert_manager.trigger(AlertType.LOOKING_AWAY)
            elif face_data and face_data["is_yawning"]:
                alert_manager.trigger(AlertType.YAWN)
            else:
                alert_manager.clear()

            # 4. Telemetry Update
            is_eyes_closed = face_data["is_eyes_closed"] if face_data else False
            is_yawning = face_data["is_yawning"] if face_data else False
            is_head_distracted = face_data["is_head_distracted"] if face_data else False

            telemetry.update(
                is_eyes_closed=is_eyes_closed,
                is_yawning=is_yawning,
                is_head_distracted=is_head_distracted,
                is_phone_distracted=yolo_distracted,
                fps=fps
            )

            # 5. Render Sleek Minimalist HUD
            display_frame = hud.render(
                frame=frame,
                face_data=face_data,
                yolo_distracted=yolo_distracted,
                detections=detections,
                alert_manager=alert_manager,
                telemetry=telemetry,
                fps=fps
            )

            if writer is not None:
                writer.write(display_frame)

            # 6. Interactive GUI Display
            if not args.headless:
                cv2.imshow("Distraction Detection System", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == 9 or key == ord('t'):  # TAB key or 't' to cycle modes
                    current_mode_idx = (current_mode_idx + 1) % len(MODES)
                    new_mode = MODES[current_mode_idx]
                    telemetry.set_mode(new_mode)
                    hud.trigger_mode_toast(new_mode)
                    print(f"[MODE] Cycled to: {new_mode}")
                elif key == ord('1'):
                    current_mode_idx = 0
                    new_mode = MODES[0]
                    telemetry.set_mode(new_mode)
                    hud.trigger_mode_toast(new_mode)
                    print(f"[MODE] Selected: {new_mode}")
                elif key == ord('2'):
                    current_mode_idx = 1
                    new_mode = MODES[1]
                    telemetry.set_mode(new_mode)
                    hud.trigger_mode_toast(new_mode)
                    print(f"[MODE] Selected: {new_mode}")
                elif key == ord('3'):
                    current_mode_idx = 2
                    new_mode = MODES[2]
                    telemetry.set_mode(new_mode)
                    hud.trigger_mode_toast(new_mode)
                    print(f"[MODE] Selected: {new_mode}")
                elif key == ord('m'):
                    state = alert_manager.toggle_mute()
                    print(f"[AUDIO] Alerts {'UNMUTED' if state else 'MUTED'}")
                elif key == ord('r'):
                    telemetry = AttentionTelemetry(mode=args.mode.upper())
                    print("[STATS] Counters reset to 0.")
                elif key == ord('d'):
                    mesh_on = hud.toggle_debug_mesh()
                    print(f"[HUD] 3D Facial Mesh: {'ON' if mesh_on else 'OFF'}")
                elif key == ord('s'):
                    snap_name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    snap_path = os.path.join(config.SNAPSHOTS_DIR, snap_name)
                    cv2.imwrite(snap_path, display_frame)
                    print(f"[SAVED] Screenshot -> {snap_path}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        camera.stop()
        if writer is not None:
            writer.release()
        alert_manager.stop()
        if not args.headless:
            cv2.destroyAllWindows()

        json_path, csv_path = telemetry.export_report()
        summary = telemetry.get_summary()

        print("\n" + "=" * 65)
        print("               SESSION ATTENTION & DISTRACTION REPORT            ")
        print("=" * 65)
        print(f" Mode               : {summary['mode']}")
        print(f" Session ID         : {summary['session_id']}")
        print(f" Duration           : {summary['duration_formatted']} ({summary['duration_seconds']}s)")
        print(f" Frames Processed   : {summary['total_frames']} (Avg {summary['average_fps']} FPS)")
        print(f" Final Score        : {summary['attention_score']}% ({summary['rating']})")
        print(f" Phone Incidents    : {summary['phone_usage_episodes']}")
        print(f" Gaze Distractions  : {summary['head_distraction_episodes']}")
        print(f" Fatigue / Drowsy   : {summary['microsleep_episodes']} (PERCLOS {summary['perclos_percentage']}%)")
        print(f" Blinks Tracked     : {summary['blink_count']} ({summary['blinks_per_minute']} BPM)")
        print("-" * 65)
        print(f"[EXPORT] JSON -> {json_path}")
        print(f"[EXPORT] CSV  -> {csv_path}")
        print("=" * 65)

if __name__ == "__main__":
    main()

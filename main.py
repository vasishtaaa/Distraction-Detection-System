"""
Real-Time Driver Safety & Distraction Detection System
Main Application Pipeline integrating OpenCV, MediaPipe Face Landmarker, and YOLOv8.
"""

import os
import sys
import time
import argparse
from datetime import datetime
import cv2

# Set MPLCONFIGDIR to temp directory to avoid matplotlib cache warnings
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "."), "matplotlib")

import config
from face_mesh_detector import DriverFaceAnalyzer
from yolo_detector import DistractionObjectDetector
from alert_system import AlertManager, AlertType
from analytics import DriverTelemetry
from ui_overlay import HUDOverlay

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Driver Safety & Distraction Detection System"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (e.g. 0) or path to video file (e.g. video.mp4)"
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
        help="Run without displaying GUI window (useful for automated benchmarks/testing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to record HUD-annotated output video"
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=True,
        help="Loop video file playback continuously until 'q' is pressed (default: True)"
    )
    parser.add_argument(
        "--no-loop",
        action="store_false",
        dest="loop",
        help="Do not loop video file playback"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before exiting (useful for testing)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 65)
    print("  REAL-TIME DRIVER SAFETY & DISTRACTION DETECTION SYSTEM")
    print("  OpenCV  |  Google MediaPipe  |  Ultralytics YOLOv8")
    print("=" * 65)

    # 1. Initialize Video Capture
    source = args.source
    if str(source).isdigit():
        source = int(source)
        print(f"[INIT] Opening live camera index {source} (trying DirectShow backend)...")
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[INIT] DirectShow failed, trying default backend for camera {source}...")
            cap = cv2.VideoCapture(source)
    else:
        print(f"[INIT] Opening video source file: {source}...")
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        if isinstance(source, int) and os.path.exists("sample_driver_demo.mp4"):
            print(f"[INFO] Camera index {source} could not be opened.")
            print("[INFO] Automatically falling back to demo video 'sample_driver_demo.mp4'...")
            source = "sample_driver_demo.mp4"
            cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Failed to open video source: {args.source}")
        print("[HINT] For webcam, verify your camera is connected. Or specify a video file with --source <file.mp4>")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    # 2. Initialize Models & Modules
    print("[INIT] Loading MediaPipe Face Landmarker...")
    face_analyzer = DriverFaceAnalyzer(model_path=config.FACE_LANDMARKER_MODEL)

    yolo_detector = None
    if not args.no_yolo:
        print("[INIT] Loading YOLOv8 Distraction Detection Model...")
        yolo_detector = DistractionObjectDetector(model_path=config.YOLO_MODEL_PATH)
    else:
        print("[INFO] YOLOv8 distraction detection disabled by flag.")

    alert_manager = AlertManager(enabled=not args.mute)
    telemetry = DriverTelemetry()
    hud = HUDOverlay()

    # 3. Optional Video Output Writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.FRAME_WIDTH
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.FRAME_HEIGHT
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(args.output, fourcc, fps_in, (out_w, out_h))
        print(f"[INIT] Recording annotated session to {args.output} ({out_w}x{out_h} @ {fps_in:.1f} fps)")

    print("[SUCCESS] All systems initialized. Commencing real-time monitoring.")
    print("Controls: [Q] Quit | [M] Mute/Unmute | [R] Reset Stats | [S] Save Snapshot | [D] Toggle Mesh")
    print("-" * 65)

    prev_time = time.time()
    fps = 30.0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.loop and isinstance(source, str) and os.path.exists(source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of video stream or camera disconnected.")
                    break

            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                print(f"[INFO] Reached maximum requested frames ({args.max_frames}). Exiting.")
                break

            # Calculate Rolling FPS
            curr_time = time.time()
            time_delta = curr_time - prev_time
            prev_time = curr_time
            if time_delta > 0:
                instant_fps = 1.0 / time_delta
                fps = 0.9 * fps + 0.1 * instant_fps

            # 1. Facial Landmark & Pose Analysis
            face_data = face_analyzer.analyze_frame(frame)

            # 2. YOLOv8 Object Detection
            yolo_distracted = False
            detections = []
            if yolo_detector is not None:
                face_bbox = face_data["face_bbox"] if face_data else None
                yolo_distracted, detections = yolo_detector.detect(frame, face_bbox=face_bbox)

            # 3. Alert Rule Evaluation
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

            # 5. Render HUD Overlay
            display_frame = hud.render(
                frame=frame,
                face_data=face_data,
                yolo_distracted=yolo_distracted,
                detections=detections,
                alert_manager=alert_manager,
                telemetry=telemetry,
                fps=fps
            )

            # 6. Record Output if enabled
            if writer is not None:
                writer.write(display_frame)

            # 7. GUI Display
            if not args.headless:
                cv2.imshow("Real-Time Driver Safety System", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("[USER] Quit command received.")
                    break
                elif key == ord('m'):
                    is_enabled = alert_manager.toggle_mute()
                    state = "UNMUTED" if is_enabled else "MUTED"
                    print(f"[AUDIO] Audio alerts {state}")
                elif key == ord('r'):
                    telemetry = DriverTelemetry()
                    print("[STATS] Telemetry counters reset to zero.")
                elif key == ord('d'):
                    mesh_state = hud.toggle_debug_mesh()
                    print(f"[HUD] Debug facial mesh: {'ON' if mesh_state else 'OFF'}")
                elif key == ord('s'):
                    snap_name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    snap_path = os.path.join(config.SNAPSHOTS_DIR, snap_name)
                    cv2.imwrite(snap_path, display_frame)
                    print(f"[SNAPSHOT] Saved snapshot -> {snap_path}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user keyboard interrupt.")
    finally:
        # Cleanup
        cap.release()
        if writer is not None:
            writer.release()
        alert_manager.stop()
        if not args.headless:
            cv2.destroyAllWindows()

        # Export analytics report
        json_path, csv_path = telemetry.export_report()
        summary = telemetry.get_summary()

        print("\n" + "=" * 65)
        print("                 DRIVE SESSION SUMMARY REPORT                ")
        print("=" * 65)
        print(f" Session ID         : {summary['session_id']}")
        print(f" Start Time         : {summary['start_time']}")
        print(f" Total Duration     : {summary['duration_formatted']} ({summary['duration_seconds']}s)")
        print(f" Frames Processed   : {summary['total_frames']} (Avg {summary['average_fps']} FPS)")
        print(f" Final Safety Score : {summary['safety_score']} / 100")
        print(f" Driver Rating      : {summary['rating_grade']}")
        print(f" PERCLOS Fatigue    : {summary['perclos_percentage']}%")
        print(f" Blinks Tracked     : {summary['blink_count']} ({summary['blinks_per_minute']} BPM)")
        print(f" Microsleeps (Sleep): {summary['microsleep_episodes']}")
        print(f" Yawning Episodes   : {summary['yawn_episodes']}")
        print(f" Head Distractions  : {summary['head_distraction_episodes']}")
        print(f" Phone Distractions : {summary['phone_usage_episodes']}")
        print(f" Incident Log Count : {summary['incident_count']}")
        print("-" * 65)
        print(f"[REPORT SAVED] JSON -> {json_path}")
        print(f"[REPORT SAVED] CSV  -> {csv_path}")
        print("=" * 65)

if __name__ == "__main__":
    main()

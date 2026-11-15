"""
Modern Automotive Heads-Up Display (HUD) and Computer Vision Overlay.
Renders real-time telemetry cards, EAR/MAR gauges, 3D gaze arrows,
object bounding boxes, and flashing emergency alerts.
"""

import cv2
import numpy as np
import time
import config
from alert_system import AlertType

class HUDOverlay:
    """
    Renders high-tech HUD graphics, telemetry cards, and safety alerts
    directly onto the video stream.
    """

    def __init__(self):
        self.show_debug_mesh = False
        self.flash_counter = 0

    def toggle_debug_mesh(self):
        """Toggles facial mesh wireframe overlay."""
        self.show_debug_mesh = not self.show_debug_mesh
        return self.show_debug_mesh

    def render(self, frame, face_data, yolo_distracted, detections, alert_manager, telemetry, fps):
        """
        Main rendering pipeline for the HUD.
        """
        h, w, _ = frame.shape
        self.flash_counter += 1

        # 1. Draw Visual Landmarks & 3D Gaze if face is present
        if face_data:
            self._draw_facial_contours(frame, face_data)
            self._draw_head_gaze(frame, face_data)

        # 2. Draw YOLOv8 Object Detections (Phone, Bottles, etc.)
        self._draw_yolo_detections(frame, detections)

        # 3. Draw Top Header Bar
        self._draw_header(frame, telemetry, fps)

        # 4. Draw Left Telemetry Panel
        self._draw_telemetry_card(frame, face_data, yolo_distracted, alert_manager, telemetry)

        # 5. Draw Flashing Emergency Alert Banner if triggered
        if alert_manager.active_alert != AlertType.NONE:
            self._draw_alert_banner(frame, alert_manager.active_alert)

        # 6. Draw Bottom Controls Legend
        self._draw_footer(frame, alert_manager.enabled)

        return frame

    def _draw_header(self, frame, telemetry, fps):
        """Renders top dark navigation bar with system title, timer, and safety score."""
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), config.COLOR_CARD_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Bottom border line for header
        cv2.line(frame, (0, 55), (w, 55), (60, 70, 85), 1)

        # Title
        cv2.putText(
            frame, "DRIVER SAFETY & ATTENTION MONITOR", (20, 36),
            cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Timer & FPS
        time_str = telemetry.get_summary()["duration_formatted"]
        info_str = f"TIME: {time_str} | FPS: {fps:.1f}"
        cv2.putText(
            frame, info_str, (w - 380, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 220, 240), 1, cv2.LINE_AA
        )

        # Safety Score Badge
        score = telemetry.compute_safety_score()
        score_color = config.COLOR_SAFE if score >= 80 else (config.COLOR_WARNING if score >= 60 else config.COLOR_DANGER)
        score_str = f"SCORE: {int(score)}/100"
        cv2.putText(
            frame, score_str, (w - 170, 36),
            cv2.FONT_HERSHEY_DUPLEX, 0.65, score_color, 2, cv2.LINE_AA
        )

    def _draw_telemetry_card(self, frame, face_data, yolo_distracted, alert_manager, telemetry):
        """Renders semi-transparent card on the left side with live driver metrics."""
        overlay = frame.copy()
        card_w = 320
        card_h = 320
        card_x = 20
        card_y = 70

        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), config.COLOR_CARD_BG, -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (60, 70, 85), 1)

        # Card Title
        cv2.putText(frame, "TELEMETRY METRICS", (card_x + 15, card_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 190, 50), 1, cv2.LINE_AA)

        # 1. Driver Overall State
        status_text = "ATTENTIVE"
        status_color = config.COLOR_SAFE

        if alert_manager.active_alert == AlertType.DROWSINESS or (face_data and face_data["is_eyes_closed"]):
            status_text = "DROWSY / EYES CLOSED"
            status_color = config.COLOR_DANGER
        elif yolo_distracted:
            status_text = "PHONE DISTRACTION"
            status_color = config.COLOR_DANGER
        elif face_data and face_data["is_head_distracted"]:
            status_text = "OFF-ROAD DISTRACTION"
            status_color = config.COLOR_WARNING
        elif face_data and face_data["is_yawning"]:
            status_text = "YAWNING / FATIGUE"
            status_color = config.COLOR_WARNING
        elif not face_data:
            status_text = "NO DRIVER DETECTED"
            status_color = config.COLOR_MUTED

        cv2.putText(frame, "Driver State:", (card_x + 15, card_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, config.COLOR_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, status_text, (card_x + 15, card_y + 78),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, status_color, 1, cv2.LINE_AA)

        # 2. EAR Gauge
        ear = face_data["ear"] if face_data else 0.0
        ear_color = config.COLOR_DANGER if ear < config.EAR_THRESHOLD else config.COLOR_SAFE
        cv2.putText(frame, f"EAR (Eyes): {ear:.2f}  (Thresh: {config.EAR_THRESHOLD})",
                    (card_x + 15, card_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_TEXT, 1, cv2.LINE_AA)
        self._draw_progress_bar(frame, card_x + 15, card_y + 118, 200, 10,
                               min(1.0, max(0.0, ear / 0.40)), ear_color)

        # 3. MAR Gauge
        mar = face_data["mar"] if face_data else 0.0
        mar_color = config.COLOR_WARNING if mar > config.MAR_THRESHOLD else config.COLOR_SAFE
        cv2.putText(frame, f"MAR (Mouth): {mar:.2f}  (Thresh: {config.MAR_THRESHOLD})",
                    (card_x + 15, card_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_TEXT, 1, cv2.LINE_AA)
        self._draw_progress_bar(frame, card_x + 15, card_y + 158, 200, 10,
                               min(1.0, max(0.0, mar / 1.0)), mar_color)

        # 4. Head Pose (Pitch & Yaw)
        pitch = face_data["head_pitch"] if face_data else 0.0
        yaw = face_data["head_yaw"] if face_data else 0.0
        pose_color = config.COLOR_WARNING if (face_data and face_data["is_head_distracted"]) else config.COLOR_SAFE
        cv2.putText(frame, f"Head Pose: Pitch {pitch:+.1f} deg | Yaw {yaw:+.1f} deg",
                    (card_x + 15, card_y + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pose_color, 1, cv2.LINE_AA)

        # 5. PERCLOS & Incident Counters
        perclos = telemetry.get_perclos()
        cv2.putText(frame, f"PERCLOS: {perclos:.1f}%  |  Blinks: {telemetry.blink_count}",
                    (card_x + 15, card_y + 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Microsleeps: {telemetry.microsleep_count}  |  Yawns: {telemetry.yawn_count}",
                    (card_x + 15, card_y + 258), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Head Distr.: {telemetry.head_distraction_count}  |  Phone: {telemetry.phone_distraction_count}",
                    (card_x + 15, card_y + 286), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_TEXT, 1, cv2.LINE_AA)

    def _draw_progress_bar(self, frame, x, y, w, h, fraction, color):
        """Draws a rounded modern progress gauge bar."""
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 55, 65), -1)
        fill_w = int(w * fraction)
        if fill_w > 0:
            cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 95, 105), 1)

    def _draw_facial_contours(self, frame, face_data):
        """Draws eye landmarks and mouth contours."""
        eye_color = config.COLOR_DANGER if face_data["is_eyes_closed"] else config.COLOR_SAFE
        mouth_color = config.COLOR_WARNING if face_data["is_yawning"] else config.COLOR_SAFE

        # Left eye outline
        left_pts = np.array(face_data["left_eye_pts"], dtype=np.int32)
        cv2.polylines(frame, [left_pts], True, eye_color, 2, cv2.LINE_AA)

        # Right eye outline
        right_pts = np.array(face_data["right_eye_pts"], dtype=np.int32)
        cv2.polylines(frame, [right_pts], True, eye_color, 2, cv2.LINE_AA)

        # Mouth vertical & horizontal axes
        p_l, p_r, p_t, p_b = face_data["mouth_pts"]
        cv2.line(frame, p_l, p_r, mouth_color, 1, cv2.LINE_AA)
        cv2.line(frame, p_t, p_b, mouth_color, 2, cv2.LINE_AA)

        # Optional full debug mesh
        if self.show_debug_mesh and "landmarks_raw" in face_data:
            h, w, _ = frame.shape
            for lm in face_data["landmarks_raw"]:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 1, (200, 200, 100), -1)

    def _draw_head_gaze(self, frame, face_data):
        """Draws 3D head gaze vector projection starting from nose tip."""
        nose_2d = face_data["nose_2d"]
        gaze_endpoint = face_data["gaze_endpoint_2d"]
        gaze_color = config.COLOR_WARNING if face_data["is_head_distracted"] else (0, 255, 255)

        # Draw projected gaze arrow
        cv2.arrowedLine(frame, nose_2d, gaze_endpoint, gaze_color, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(frame, nose_2d, 4, (0, 200, 255), -1)

    def _draw_yolo_detections(self, frame, detections):
        """Draws bounding boxes for detected cell phones and distraction objects."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_id = det["class_id"]
            name = det["name"]
            conf = det["confidence"]

            if cls_id == config.PHONE_CLASS_ID:
                box_color = config.COLOR_DANGER
                label = f"DISTRACTION: {name.upper()} ({conf*100:.0f}%)"
                # Thick attention border
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            else:
                box_color = config.COLOR_WARNING
                label = f"{name} ({conf*100:.0f}%)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Label banner
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + lw + 10, y1), box_color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_alert_banner(self, frame, alert_type):
        """Draws high-visibility flashing emergency warning banner."""
        # 12 frames on, 6 frames off flash cycle
        if (self.flash_counter % 18) > 12:
            return

        h, w, _ = frame.shape
        banner_h = 70
        banner_y = int(h * 0.42)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, banner_y), (w, banner_y + banner_h), (10, 10, 200), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Border lines
        cv2.line(frame, (0, banner_y), (w, banner_y), (50, 50, 255), 3)
        cv2.line(frame, (0, banner_y + banner_h), (w, banner_y + banner_h), (50, 50, 255), 3)

        # Warning Text Centered
        text = f"!!! {alert_type} !!!"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
        tx = int((w - tw) / 2)
        ty = banner_y + int((banner_h + th) / 2) - 3

        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_footer(self, frame, audio_enabled):
        """Renders bottom toolbar with keyboard shortcut hints."""
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 35), (w, h), config.COLOR_CARD_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        audio_str = "MUTE [M]" if audio_enabled else "UNMUTE [M]"
        legend = f"[Q] Quit  |  [{audio_str}]  |  [R] Reset Stats  |  [S] Save Snapshot  |  [D] Toggle Mesh"
        cv2.putText(frame, legend, (20, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_MUTED, 1, cv2.LINE_AA)

"""
Minimalist, High-Tech Computer Vision HUD Overlay.
Designed with modern glassmorphism aesthetics, floating status pills,
and clean unobtrusive graphics for a polished user experience.
"""

import cv2
import numpy as np
import config
from alert_system import AlertType

class HUDOverlay:
    """
    Renders a sleek, modern visual interface over the video stream.
    Focuses on clarity, minimalist aesthetics, and zero visual clutter.
    """

    def __init__(self):
        self.show_debug_mesh = False
        self.pulse_counter = 0

    def toggle_debug_mesh(self):
        self.show_debug_mesh = not self.show_debug_mesh
        return self.show_debug_mesh

    def render(self, frame, face_data, yolo_distracted, detections, alert_manager, telemetry, fps):
        h, w, _ = frame.shape
        self.pulse_counter += 1

        # 1. Subtle Face Tracking & Gaze
        if face_data:
            self._draw_face_brackets(frame, face_data["face_bbox"])
            self._draw_eye_contours(frame, face_data)
            self._draw_gaze_arrow(frame, face_data)

            if self.show_debug_mesh and "landmarks_raw" in face_data:
                for lm in face_data["landmarks_raw"]:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 1, (220, 200, 100), -1)

        # 2. YOLOv8 Target Brackets (Phones, Bottles)
        self._draw_detected_objects(frame, detections)

        # 3. Top Floating Status Pill (Apple/Tesla style indicator)
        self._draw_status_pill(frame, face_data, yolo_distracted, alert_manager)

        # 4. Minimalist Bottom Telemetry Bar
        self._draw_bottom_bar(frame, telemetry, fps, alert_manager.enabled)

        # 5. Emergency Warning Flash Banner (only on critical danger)
        if alert_manager.active_alert in [AlertType.DROWSINESS, AlertType.PHONE]:
            self._draw_critical_banner(frame, alert_manager.active_alert)

        return frame

    def _draw_status_pill(self, frame, face_data, yolo_distracted, alert_manager):
        """Draws a centered floating status pill badge at the top of the video."""
        h, w, _ = frame.shape

        if alert_manager.active_alert == AlertType.DROWSINESS or (face_data and face_data["is_eyes_closed"]):
            status_text = "DROWSINESS DETECTED"
            pill_color = config.COLOR_DANGER
        elif yolo_distracted:
            status_text = "DISTRACTION: CELL PHONE"
            pill_color = config.COLOR_DANGER
        elif face_data and face_data["is_head_distracted"]:
            status_text = "DISTRACTION: LOOKING AWAY"
            pill_color = config.COLOR_WARNING
        elif face_data and face_data["is_yawning"]:
            status_text = "FATIGUE: YAWNING"
            pill_color = config.COLOR_WARNING
        elif not face_data:
            status_text = "SEARCHING FOR USER..."
            pill_color = config.COLOR_MUTED
        else:
            status_text = "ATTENTIVE & FOCUSED"
            pill_color = config.COLOR_SAFE

        # Measure text size
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.58
        thickness = 1
        (tw, th), _ = cv2.getTextSize(status_text, font, font_scale, thickness)

        pill_w = tw + 45
        pill_h = 36
        pill_x = int((w - pill_w) / 2)
        pill_y = 18

        # Glassmorphic rounded pill background
        overlay = frame.copy()
        cv2.rectangle(overlay, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), (18, 22, 32), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), pill_color, 1)

        # Status pulse dot
        dot_x = pill_x + 18
        dot_y = pill_y + int(pill_h / 2)
        cv2.circle(frame, (dot_x, dot_y), 5, pill_color, -1)

        # Status label
        text_x = dot_x + 14
        text_y = pill_y + int((pill_h + th) / 2) - 1
        cv2.putText(frame, status_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _draw_face_brackets(self, frame, bbox):
        """Draws subtle modern corner brackets around the user's face."""
        x1, y1, x2, y2 = bbox
        bracket_len = min(20, int((x2 - x1) * 0.2))
        color = (180, 200, 220)
        thick = 1

        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, thick)
        cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, thick)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, thick)
        cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, thick)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, thick)
        cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, thick)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, thick)
        cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, thick)

    def _draw_eye_contours(self, frame, face_data):
        """Draws smooth outlines around eyes showing open/closed state."""
        eye_color = config.COLOR_DANGER if face_data["is_eyes_closed"] else config.COLOR_SAFE
        left_pts = np.array(face_data["left_eye_pts"], dtype=np.int32)
        right_pts = np.array(face_data["right_eye_pts"], dtype=np.int32)
        cv2.polylines(frame, [left_pts], True, eye_color, 1, cv2.LINE_AA)
        cv2.polylines(frame, [right_pts], True, eye_color, 1, cv2.LINE_AA)

    def _draw_gaze_arrow(self, frame, face_data):
        """Draws 3D head gaze vector projection."""
        nose_2d = face_data["nose_2d"]
        gaze_endpoint = face_data["gaze_endpoint_2d"]
        color = config.COLOR_WARNING if face_data["is_head_distracted"] else (0, 220, 255)
        cv2.arrowedLine(frame, nose_2d, gaze_endpoint, color, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(frame, nose_2d, 3, (0, 220, 255), -1)

    def _draw_detected_objects(self, frame, detections):
        """Draws clean target boxes for detected phones or objects."""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            name = det["name"]
            conf = det["confidence"]

            if det["class_id"] == config.PHONE_CLASS_ID:
                box_color = config.COLOR_DANGER
                label = f"PHONE ({conf*100:.0f}%)"
            else:
                box_color = config.COLOR_WARNING
                label = f"{name} ({conf*100:.0f}%)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

    def _draw_bottom_bar(self, frame, telemetry, fps, audio_enabled):
        """Draws a sleek, dark bottom navigation bar with essential stats."""
        h, w, _ = frame.shape
        bar_h = 34
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (14, 18, 26), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.line(frame, (0, h - bar_h), (w, h - bar_h), (35, 45, 60), 1)

        # Mode Badge
        mode_text = f"MODE: {telemetry.mode}"
        cv2.putText(frame, mode_text, (18, h - 11),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, config.COLOR_PRIMARY, 1, cv2.LINE_AA)

        # Focus Score & Time
        score = telemetry.compute_score()
        score_color = config.COLOR_SAFE if score >= 80 else (config.COLOR_WARNING if score >= 60 else config.COLOR_DANGER)
        time_str = telemetry.get_summary()["duration_formatted"]

        mid_text = f"SCORE: {int(score)}%  |  TIME: {time_str}  |  {fps:.1f} FPS"
        cv2.putText(frame, mid_text, (170, h - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, score_color, 1, cv2.LINE_AA)

        # Right-aligned keyboard hints
        audio_label = "MUTE [M]" if audio_enabled else "UNMUTE [M]"
        controls_text = f"[D] Mesh  |  [S] Save  |  [R] Reset  |  [Q] Exit"
        cv2.putText(frame, controls_text, (w - 320, h - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, config.COLOR_MUTED, 1, cv2.LINE_AA)

    def _draw_critical_banner(self, frame, alert_type):
        """Subtle top alert banner that pulses during active critical violations."""
        if (self.pulse_counter % 16) > 10:
            return

        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 5), config.COLOR_DANGER, -1)
        cv2.rectangle(overlay, (0, h - 5), (w, h), config.COLOR_DANGER, -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

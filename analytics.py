"""
Driver Safety Telemetry, PERCLOS Fatigue Calculation, and Session Analytics.
Generates comprehensive post-drive safety reports in JSON and CSV formats.
"""

import time
import json
import csv
import os
from datetime import datetime
import config

class DriverTelemetry:
    """
    Records real-time driving metrics and evaluates driver safety score
    using industry standards like PERCLOS (Percentage of Eye Closure).
    """

    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.last_frame_time = self.start_time

        # Frame and timing stats
        self.total_frames = 0
        self.closed_eye_frames = 0
        self.yawn_frames = 0
        self.distracted_head_frames = 0
        self.phone_usage_frames = 0

        # Event counts & logs
        self.blink_count = 0
        self.microsleep_count = 0
        self.yawn_count = 0
        self.head_distraction_count = 0
        self.phone_distraction_count = 0

        self.incident_log = []

        # State trackers for event transition detection
        self._is_blinking = False
        self._blink_start_time = 0.0
        self._eye_closed_start_time = 0.0
        self._is_microsleeping = False

        self._is_yawning = False
        self._yawn_start_time = 0.0
        self._last_yawn_time = 0.0

        self._is_head_distracted = False
        self._head_distract_start_time = 0.0

        self._is_phone_distracted = False
        self._phone_distract_start_time = 0.0

    def update(self, is_eyes_closed, is_yawning, is_head_distracted, is_phone_distracted, fps):
        """
        Updates telemetry per video frame.
        """
        now = time.time()
        self.total_frames += 1

        # 1. Eye Closure & Blink Tracking
        if is_eyes_closed:
            self.closed_eye_frames += 1
            if not self._is_blinking:
                self._is_blinking = True
                self._blink_start_time = now
                self._eye_closed_start_time = now
            else:
                closure_duration = now - self._eye_closed_start_time
                if closure_duration >= config.DROWSY_TIME_THRESHOLD_SEC and not self._is_microsleeping:
                    self._is_microsleeping = True
                    self.microsleep_count += 1
                    self._log_incident("MICROSLEEP_DROWSINESS", closure_duration)
        else:
            if self._is_blinking:
                duration = now - self._blink_start_time
                if duration < 0.4:
                    self.blink_count += 1
                self._is_blinking = False
                self._is_microsleeping = False

        # 2. Yawn Tracking
        if is_yawning:
            self.yawn_frames += 1
            if not self._is_yawning and (now - self._last_yawn_time) > config.YAWN_COOLDOWN_SEC:
                self._is_yawning = True
                self._yawn_start_time = now
        else:
            if self._is_yawning:
                duration = now - self._yawn_start_time
                if duration >= 1.0:
                    self.yawn_count += 1
                    self._last_yawn_time = now
                    self._log_incident("YAWN_FATIGUE", duration)
                self._is_yawning = False

        # 3. Head Pose Distraction Tracking
        if is_head_distracted:
            self.distracted_head_frames += 1
            if not self._is_head_distracted:
                self._is_head_distracted = True
                self._head_distract_start_time = now
            else:
                distract_duration = now - self._head_distract_start_time
                if distract_duration >= config.DISTRACTION_TIME_SEC and distract_duration < (config.DISTRACTION_TIME_SEC + 0.15):
                    self.head_distraction_count += 1
                    self._log_incident("HEAD_DISTRACTION", distract_duration)
        else:
            self._is_head_distracted = False

        # 4. Phone Distraction Tracking
        if is_phone_distracted:
            self.phone_usage_frames += 1
            if not self._is_phone_distracted:
                self._is_phone_distracted = True
                self._phone_distract_start_time = now
                self.phone_distraction_count += 1
                self._log_incident("PHONE_USAGE", 0.0)
        else:
            self._is_phone_distracted = False

    def _log_incident(self, incident_type, duration):
        """Logs an event timestamp and details."""
        elapsed = time.time() - self.start_time
        self.incident_log.append({
            "timestamp": datetime.now().isoformat(),
            "drive_time_sec": round(elapsed, 1),
            "type": incident_type,
            "duration_sec": round(duration, 2)
        })

    def get_perclos(self):
        """
        PERCLOS: Percentage of Eye Closure time.
        Industry Standard (NHTSA):
        < 8% : Attentive / Alert
        8% - 15% : Borderline Drowsy
        > 15% : Significantly Fatigued
        """
        if self.total_frames == 0:
            return 0.0
        return (self.closed_eye_frames / self.total_frames) * 100.0

    def compute_safety_score(self):
        """
        Calculates driver safety score out of 100.
        Penalties based on safety violations:
        - Microsleep: -15 pts per episode
        - Phone Usage: -12 pts per episode
        - Prolonged Head Distraction: -8 pts per episode
        - Frequent Yawning: -4 pts per episode
        - Elevated PERCLOS (>10%): -10 pts
        """
        score = 100.0
        score -= (self.microsleep_count * 15.0)
        score -= (self.phone_distraction_count * 12.0)
        score -= (self.head_distraction_count * 8.0)
        score -= (self.yawn_count * 4.0)

        perclos = self.get_perclos()
        if perclos > 15.0:
            score -= 20.0
        elif perclos > 8.0:
            score -= 10.0

        score = max(0.0, min(100.0, score))
        return round(score, 1)

    def get_grade(self, score):
        """Converts score into qualitative letter grade."""
        if score >= 90:
            return "A+ (Excellent Attention)"
        elif score >= 80:
            return "A (Safe Driver)"
        elif score >= 70:
            return "B (Minor Inattention)"
        elif score >= 55:
            return "C (Caution: Elevated Risk)"
        elif score >= 40:
            return "D (Dangerous Inattention)"
        else:
            return "F (CRITICAL SAFETY HAZARD)"

    def get_summary(self):
        """Returns structured session summary dictionary."""
        total_time = max(1.0, time.time() - self.start_time)
        perclos = self.get_perclos()
        safety_score = self.compute_safety_score()
        blinks_per_min = (self.blink_count / (total_time / 60.0))

        return {
            "session_id": self.session_id,
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(total_time, 1),
            "duration_formatted": f"{int(total_time // 60):02d}:{int(total_time % 60):02d}",
            "total_frames": self.total_frames,
            "average_fps": round(self.total_frames / total_time, 1),
            "safety_score": safety_score,
            "rating_grade": self.get_grade(safety_score),
            "perclos_percentage": round(perclos, 2),
            "blink_count": self.blink_count,
            "blinks_per_minute": round(blinks_per_min, 1),
            "microsleep_episodes": self.microsleep_count,
            "yawn_episodes": self.yawn_count,
            "head_distraction_episodes": self.head_distraction_count,
            "phone_usage_episodes": self.phone_distraction_count,
            "incident_count": len(self.incident_log),
            "incidents": self.incident_log
        }

    def export_report(self):
        """
        Saves full session report to JSON and CSV in config.REPORTS_DIR.
        """
        summary = self.get_summary()

        # Save JSON
        json_path = os.path.join(config.REPORTS_DIR, f"report_{self.session_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        # Save Incident CSV
        csv_path = os.path.join(config.REPORTS_DIR, f"incidents_{self.session_id}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Drive Time (s)", "Incident Type", "Duration (s)"])
            for inc in self.incident_log:
                writer.writerow([inc["timestamp"], inc["drive_time_sec"], inc["type"], inc["duration_sec"]])

        return json_path, csv_path

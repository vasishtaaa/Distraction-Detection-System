"""
Real-Time Attention & Distraction Telemetry Engine.
Tracks focus duration, PERCLOS fatigue index, incident timelines,
and computes dynamic Attention / Focus Scores.
"""

import time
import json
import csv
import os
from datetime import datetime
import config

class AttentionTelemetry:
    """
    Quantifies user attentiveness, distraction events, and fatigue levels.
    """

    def __init__(self, mode=config.DEFAULT_MODE):
        self.mode = mode
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()

        # Frame counters
        self.total_frames = 0
        self.closed_eye_frames = 0
        self.yawn_frames = 0
        self.distracted_head_frames = 0
        self.phone_usage_frames = 0

        # Event counts
        self.blink_count = 0
        self.microsleep_count = 0
        self.yawn_count = 0
        self.head_distraction_count = 0
        self.phone_distraction_count = 0

        self.incident_log = []

        # Internal state tracking
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

    def set_mode(self, mode):
        """Updates the operational mode (STUDY, DRIVING, GENERAL)."""
        self.mode = mode

    def update(self, is_eyes_closed, is_yawning, is_head_distracted, is_phone_distracted, fps):
        """Processes frame analysis results to update telemetry statistics."""
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
                    self._log_incident("DROWSINESS_EPISODE", closure_duration)
        else:
            if self._is_blinking:
                duration = now - self._blink_start_time
                if duration < 0.4:
                    self.blink_count += 1
                self._is_blinking = False
                self._is_microsleeping = False

        # 2. Yawning Tracking
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
                    self._log_incident("FATIGUE_YAWN", duration)
                self._is_yawning = False

        # 3. Head Distraction (Looking Away)
        if is_head_distracted:
            self.distracted_head_frames += 1
            if not self._is_head_distracted:
                self._is_head_distracted = True
                self._head_distract_start_time = now
            else:
                distract_duration = now - self._head_distract_start_time
                if distract_duration >= config.DISTRACTION_TIME_SEC and distract_duration < (config.DISTRACTION_TIME_SEC + 0.15):
                    self.head_distraction_count += 1
                    self._log_incident("LOOKING_AWAY", distract_duration)
        else:
            self._is_head_distracted = False

        # 4. Cell Phone Distraction
        if is_phone_distracted:
            self.phone_usage_frames += 1
            if not self._is_phone_distracted:
                self._is_phone_distracted = True
                self.phone_distraction_count += 1
                self._log_incident("PHONE_USAGE", 0.0)
        else:
            self._is_phone_distracted = False

    def _log_incident(self, incident_type, duration):
        elapsed = time.time() - self.start_time
        self.incident_log.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "session_time_sec": round(elapsed, 1),
            "type": incident_type,
            "duration_sec": round(duration, 2)
        })

    def get_perclos(self):
        """Percentage of Eye Closure time (Fatigue Index)."""
        if self.total_frames == 0:
            return 0.0
        return (self.closed_eye_frames / self.total_frames) * 100.0

    def compute_score(self):
        """
        Calculates dynamic Attention / Focus Score (0 - 100).
        """
        score = 100.0
        score -= (self.phone_distraction_count * 10.0)
        score -= (self.head_distraction_count * 6.0)
        score -= (self.microsleep_count * 12.0)
        score -= (self.yawn_count * 3.0)

        perclos = self.get_perclos()
        if perclos > 15.0:
            score -= 15.0
        elif perclos > 8.0:
            score -= 8.0

        return max(0.0, min(100.0, round(score, 1)))

    def get_rating(self, score):
        if score >= 90:
            return "Excellent Focus"
        elif score >= 80:
            return "Good Attention"
        elif score >= 65:
            return "Moderate Inattention"
        elif score >= 50:
            return "Elevated Distraction"
        else:
            return "Critical Distraction"

    def get_summary(self):
        total_time = max(1.0, time.time() - self.start_time)
        score = self.compute_score()
        blinks_per_min = self.blink_count / (total_time / 60.0)

        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(total_time, 1),
            "duration_formatted": f"{int(total_time // 60):02d}:{int(total_time % 60):02d}",
            "total_frames": self.total_frames,
            "average_fps": round(self.total_frames / total_time, 1),
            "attention_score": score,
            "rating": self.get_rating(score),
            "perclos_percentage": round(self.get_perclos(), 1),
            "blink_count": self.blink_count,
            "blinks_per_minute": round(blinks_per_min, 1),
            "microsleep_episodes": self.microsleep_count,
            "yawn_episodes": self.yawn_count,
            "head_distraction_episodes": self.head_distraction_count,
            "phone_usage_episodes": self.phone_distraction_count,
            "incident_count": len(self.incident_log),
            "recent_incidents": self.incident_log[-10:]
        }

    def export_report(self):
        summary = self.get_summary()
        json_path = os.path.join(config.REPORTS_DIR, f"session_{self.session_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        csv_path = os.path.join(config.REPORTS_DIR, f"incidents_{self.session_id}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Session Time (s)", "Incident Type", "Duration (s)"])
            for inc in self.incident_log:
                writer.writerow([inc["timestamp"], inc["session_time_sec"], inc["type"], inc["duration_sec"]])

        return json_path, csv_path

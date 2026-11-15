"""
Asynchronous Multi-Threaded Audio & Visual Alert Manager
Handles safety alarms, warning beeps, and rate-limited notifications without stalling CV loop.
"""

import time
import threading
import queue
import config

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

class AlertType:
    NONE = "NORMAL"
    DROWSINESS = "CRITICAL: DROWSINESS DETECTED"
    PHONE = "WARNING: CELL PHONE DISTRACTION"
    LOOKING_AWAY = "WARNING: EYES OFF ROAD"
    YAWN = "CAUTION: FATIGUE / YAWNING"

class AlertManager:
    """
    Manages non-blocking driver alerts using a background worker thread.
    Prevents repeated alarm spam via a configurable cooldown timer.
    """

    def __init__(self, enabled=config.AUDIO_ALERTS_ENABLED, cooldown=config.ALERT_COOLDOWN_SEC):
        self.enabled = enabled
        self.cooldown = cooldown
        self.last_alert_time = {
            AlertType.DROWSINESS: 0.0,
            AlertType.PHONE: 0.0,
            AlertType.LOOKING_AWAY: 0.0,
            AlertType.YAWN: 0.0
        }
        self.alert_queue = queue.Queue(maxsize=10)
        self.active_alert = AlertType.NONE
        self.alert_start_time = 0.0
        self.is_running = True

        # Start background worker daemon
        self.worker = threading.Thread(target=self._audio_worker, daemon=True)
        self.worker.start()

    def trigger(self, alert_type):
        """
        Requests an alert. If cooldown has passed, pushes request to audio queue.
        """
        now = time.time()
        last_time = self.last_alert_time.get(alert_type, 0.0)

        # Update active state for visual HUD
        self.active_alert = alert_type
        self.alert_start_time = now

        if not self.enabled:
            return

        if (now - last_time) >= self.cooldown:
            self.last_alert_time[alert_type] = now
            try:
                self.alert_queue.put_nowait(alert_type)
            except queue.Full:
                pass

    def clear(self):
        """Clears active alert when driver returns to attentive state."""
        if time.time() - self.alert_start_time > 1.2:
            self.active_alert = AlertType.NONE

    def toggle_mute(self):
        """Toggles audio mute state."""
        self.enabled = not self.enabled
        return self.enabled

    def _audio_worker(self):
        """Background thread executing sound patterns."""
        while self.is_running:
            try:
                alert = self.alert_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.enabled:
                continue

            try:
                if alert == AlertType.DROWSINESS:
                    self._play_drowsy_pattern()
                elif alert == AlertType.PHONE:
                    self._play_phone_pattern()
                elif alert == AlertType.LOOKING_AWAY:
                    self._play_distraction_pattern()
                elif alert == AlertType.YAWN:
                    self._play_yawn_pattern()
            except Exception:
                pass  # Graceful audio device fallback

    def _play_drowsy_pattern(self):
        """Rapid urgent alarm pattern."""
        if HAS_WINSOUND:
            try:
                # Primary attempt: high-frequency beeps
                winsound.Beep(2000, 150)
                time.sleep(0.08)
                winsound.Beep(2500, 200)
            except Exception:
                try:
                    winsound.MessageBeep(winsound.MB_ICONHAND)
                except Exception:
                    pass

    def _play_phone_pattern(self):
        """Dual warning beep."""
        if HAS_WINSOUND:
            try:
                winsound.Beep(1200, 120)
                time.sleep(0.06)
                winsound.Beep(1200, 120)
            except Exception:
                try:
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                except Exception:
                    pass

    def _play_distraction_pattern(self):
        """Moderate alert tone."""
        if HAS_WINSOUND:
            try:
                winsound.Beep(1000, 200)
            except Exception:
                try:
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                except Exception:
                    pass

    def _play_yawn_pattern(self):
        """Gentle reminder chime."""
        if HAS_WINSOUND:
            try:
                winsound.Beep(700, 180)
            except Exception:
                try:
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
                except Exception:
                    pass

    def stop(self):
        """Stops background worker."""
        self.is_running = False

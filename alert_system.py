"""
Asynchronous Multi-Threaded Audio & Visual Alert Manager
Handles notifications, alarms, and rate-limited beeps without blocking video capture or AI inference.
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
    NONE = "ATTENTIVE"
    DROWSINESS = "DROWSINESS DETECTED"
    PHONE = "CELL PHONE DISTRACTION"
    LOOKING_AWAY = "LOOKING AWAY"
    YAWN = "FATIGUE / YAWNING"

class AlertManager:
    """
    Non-blocking alert coordinator with background audio queue and cooldown management.
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

        self.worker = threading.Thread(target=self._audio_worker, daemon=True)
        self.worker.start()

    def trigger(self, alert_type):
        """Pushes an alert to the visual HUD and audio worker if cooldown elapsed."""
        now = time.time()
        last_time = self.last_alert_time.get(alert_type, 0.0)

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
        """Resets active alert when user returns to focused attention."""
        if time.time() - self.alert_start_time > 0.8:
            self.active_alert = AlertType.NONE

    def toggle_mute(self):
        """Toggles audio mute state."""
        self.enabled = not self.enabled
        return self.enabled

    def _audio_worker(self):
        """Background thread executing sound alerts."""
        while self.is_running:
            try:
                alert = self.alert_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self.enabled:
                continue

            try:
                if alert == AlertType.DROWSINESS:
                    self._play_drowsy_sound()
                elif alert == AlertType.PHONE:
                    self._play_phone_sound()
                elif alert == AlertType.LOOKING_AWAY:
                    self._play_distraction_sound()
                elif alert == AlertType.YAWN:
                    self._play_yawn_sound()
            except Exception:
                pass

    def _play_drowsy_sound(self):
        if HAS_WINSOUND:
            try:
                winsound.Beep(2200, 150)
                time.sleep(0.06)
                winsound.Beep(2600, 180)
            except Exception:
                try:
                    winsound.MessageBeep(winsound.MB_ICONHAND)
                except Exception:
                    pass

    def _play_phone_sound(self):
        if HAS_WINSOUND:
            try:
                winsound.Beep(1400, 120)
                time.sleep(0.05)
                winsound.Beep(1400, 120)
            except Exception:
                try:
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                except Exception:
                    pass

    def _play_distraction_sound(self):
        if HAS_WINSOUND:
            try:
                winsound.Beep(1100, 160)
            except Exception:
                try:
                    winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                except Exception:
                    pass

    def _play_yawn_sound(self):
        if HAS_WINSOUND:
            try:
                winsound.Beep(750, 180)
            except Exception:
                try:
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
                except Exception:
                    pass

    def stop(self):
        """Shuts down background worker thread."""
        self.is_running = False

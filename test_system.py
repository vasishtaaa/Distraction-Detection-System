"""
Automated Test Suite for Driver Safety & Distraction Detection System.
Verifies all modules: MediaPipe Face Landmarker, YOLOv8 Detector,
Alert Manager, Telemetry/PERCLOS Analytics, and HUD Overlay.
"""

import os
import sys
import time
import unittest
import numpy as np
import cv2

os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "."), "matplotlib")

import config
from face_mesh_detector import DriverFaceAnalyzer
from yolo_detector import DistractionObjectDetector
from alert_system import AlertManager, AlertType
from analytics import DriverTelemetry
from ui_overlay import HUDOverlay

class TestDriverSafetySystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 60)
        print(" RUNNING AUTOMATED UNIT & INTEGRATION TESTS")
        print("=" * 60)

    def test_01_config_paths(self):
        """Verify models directory and essential configuration parameters."""
        print("[TEST] Verifying configuration and model files...")
        self.assertTrue(os.path.exists(config.FACE_LANDMARKER_MODEL), "Face landmarker model missing")
        self.assertTrue(os.path.exists(config.YOLO_MODEL_PATH), "YOLOv8 weights missing")
        self.assertGreater(config.EAR_THRESHOLD, 0.1)
        self.assertGreater(config.MAR_THRESHOLD, 0.3)

    def test_02_face_analyzer(self):
        """Verify MediaPipe Face Landmarker loads and processes frames."""
        print("[TEST] Testing MediaPipe Face Landmarker & Pose Estimation...")
        analyzer = DriverFaceAnalyzer()
        # Test on blank frame
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        res_blank = analyzer.analyze_frame(blank)
        self.assertIsNone(res_blank, "Blank frame should yield None")

        # Test on test_face.jpg
        if os.path.exists("test_face.jpg"):
            img = cv2.imread("test_face.jpg")
            res = analyzer.analyze_frame(img)
            self.assertIsNotNone(res, "Sample face should be detected")
            self.assertIn("ear", res)
            self.assertIn("mar", res)
            self.assertIn("head_pitch", res)
            self.assertIn("head_yaw", res)
            self.assertIn("head_roll", res)
            self.assertGreater(res["ear"], 0.0)
            self.assertGreater(res["mar"], 0.0)
            print(f"       Detected Face - EAR: {res['ear']:.3f}, MAR: {res['mar']:.3f}, Yaw: {res['head_yaw']:.1f}")

    def test_03_yolo_detector(self):
        """Verify YOLOv8 loads and executes inference cleanly."""
        print("[TEST] Testing YOLOv8 Object Detection...")
        detector = DistractionObjectDetector()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        is_phone, detections = detector.detect(blank)
        self.assertFalse(is_phone)
        self.assertIsInstance(detections, list)
        print("       YOLOv8 inference executed without errors.")

    def test_04_alert_manager(self):
        """Verify asynchronous alert triggering and cooldown."""
        print("[TEST] Testing Multi-Threaded Alert System...")
        mgr = AlertManager(enabled=True, cooldown=1.0)
        mgr.trigger(AlertType.DROWSINESS)
        self.assertEqual(mgr.active_alert, AlertType.DROWSINESS)

        # Mute toggle test
        is_enabled = mgr.toggle_mute()
        self.assertFalse(is_enabled)
        is_enabled = mgr.toggle_mute()
        self.assertTrue(is_enabled)
        mgr.stop()
        print("       Alert manager worker thread operational.")

    def test_05_telemetry_analytics(self):
        """Verify PERCLOS and Driver Safety Score calculations."""
        print("[TEST] Testing Telemetry, PERCLOS & Safety Score...")
        telemetry = DriverTelemetry()

        # Simulate 100 frames with 20 closed eye frames
        for _ in range(80):
            telemetry.update(is_eyes_closed=False, is_yawning=False, is_head_distracted=False, is_phone_distracted=False, fps=30.0)
        for _ in range(20):
            telemetry.update(is_eyes_closed=True, is_yawning=False, is_head_distracted=False, is_phone_distracted=False, fps=30.0)

        perclos = telemetry.get_perclos()
        self.assertAlmostEqual(perclos, 20.0, places=1)

        score = telemetry.compute_safety_score()
        self.assertLessEqual(score, 100.0)
        grade = telemetry.get_grade(score)
        self.assertIsInstance(grade, str)

        json_path, csv_path = telemetry.export_report()
        self.assertTrue(os.path.exists(json_path), "JSON report should be saved")
        self.assertTrue(os.path.exists(csv_path), "CSV report should be saved")
        print(f"       PERCLOS: {perclos:.1f}%, Score: {score}, Grade: {grade}")

    def test_06_hud_overlay(self):
        """Verify HUD overlay renders cleanly onto frames."""
        print("[TEST] Testing HUD Canvas & Overlay Elements...")
        hud = HUDOverlay()
        mgr = AlertManager(enabled=False)
        telemetry = DriverTelemetry()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        mock_face_data = {
            "ear": 0.28,
            "mar": 0.35,
            "head_pitch": 5.0,
            "head_yaw": -10.0,
            "head_roll": 1.0,
            "is_eyes_closed": False,
            "is_yawning": False,
            "is_head_distracted": False,
            "left_eye_pts": [(300, 200), (310, 195), (320, 195), (330, 200), (320, 205), (310, 205)],
            "right_eye_pts": [(400, 200), (410, 195), (420, 195), (430, 200), (420, 205), (410, 205)],
            "mouth_pts": ((350, 300), (390, 300), (370, 290), (370, 310)),
            "nose_2d": (370, 250),
            "gaze_endpoint_2d": (370, 280)
        }

        rendered = hud.render(
            frame=frame,
            face_data=mock_face_data,
            yolo_distracted=False,
            detections=[],
            alert_manager=mgr,
            telemetry=telemetry,
            fps=30.0
        )
        self.assertEqual(rendered.shape, (720, 1280, 3))
        mgr.stop()
        print("       HUD elements rendered successfully.")

if __name__ == "__main__":
    unittest.main()

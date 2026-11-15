"""
Automated Unit & Integration Test Suite.
Verifies all modules: Threaded Camera, MediaPipe Landmarker, YOLOv8,
Alert Manager, Attention Telemetry, and Minimalist HUD.
"""

import os
import unittest
import numpy as np
import cv2

os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "."), "matplotlib")

import config
from camera import ThreadedCamera
from face_mesh_detector import DriverFaceAnalyzer
from yolo_detector import DistractionObjectDetector
from alert_system import AlertManager, AlertType
from analytics import AttentionTelemetry
from ui_overlay import HUDOverlay

class TestDistractionDetectionSystem(unittest.TestCase):

    def test_01_config_and_models(self):
        """Verify model files and configurations exist."""
        self.assertTrue(os.path.exists(config.FACE_LANDMARKER_MODEL))
        self.assertTrue(os.path.exists(config.YOLO_MODEL_PATH))

    def test_02_threaded_camera(self):
        """Verify threaded camera initializes with fallback."""
        cam = ThreadedCamera(source="sample_driver_demo.mp4")
        self.assertTrue(cam.is_opened)
        ret, frame = cam.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        cam.stop()

    def test_03_face_analyzer(self):
        """Verify MediaPipe landmark and head pose estimation."""
        analyzer = DriverFaceAnalyzer()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertIsNone(analyzer.analyze_frame(blank))

        if os.path.exists("test_face.jpg"):
            img = cv2.imread("test_face.jpg")
            res = analyzer.analyze_frame(img)
            self.assertIsNotNone(res)
            self.assertIn("ear", res)
            self.assertIn("head_yaw", res)

    def test_04_yolo_detector(self):
        """Verify YOLOv8 runs without error."""
        detector = DistractionObjectDetector()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        is_phone, dets = detector.detect(blank)
        self.assertFalse(is_phone)
        self.assertIsInstance(dets, list)

    def test_05_attention_telemetry(self):
        """Verify Attention score and mode support."""
        telemetry = AttentionTelemetry(mode="STUDY")
        for _ in range(30):
            telemetry.update(False, False, False, False, 30.0)
        score = telemetry.compute_score()
        self.assertEqual(score, 100.0)
        json_path, csv_path = telemetry.export_report()
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(csv_path))

    def test_06_hud_overlay(self):
        """Verify clean HUD overlay renders onto frame."""
        hud = HUDOverlay()
        mgr = AlertManager(enabled=False)
        telemetry = AttentionTelemetry()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rendered = hud.render(frame, None, False, [], mgr, telemetry, 30.0)
        self.assertEqual(rendered.shape, (480, 640, 3))
        mgr.stop()

if __name__ == "__main__":
    unittest.main()

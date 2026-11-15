"""
YOLOv8 Object Detection Module for External Distraction Vectors
(Cell Phone Usage, Drinking/Eating Objects).
"""

import cv2
import numpy as np
from ultralytics import YOLO
import config

class DistractionObjectDetector:
    """
    Leverages YOLOv8 to detect external objects that cause driver distraction:
    - Cell phones (texting, calling, hand-held usage)
    - Bottles / Cups (eating/drinking while driving)
    """

    CLASS_NAMES = {
        67: "Cell Phone",
        39: "Bottle",
        41: "Cup"
    }

    def __init__(self, model_path=config.YOLO_MODEL_PATH, conf=config.YOLO_CONFIDENCE):
        self.model_path = model_path
        self.conf = conf
        self.model = YOLO(self.model_path)
        self.phone_detected_consecutive_frames = 0

    def detect(self, frame_bgr, face_bbox=None):
        """
        Runs YOLOv8 inference on the video frame.
        Optionally evaluates proximity to driver's face to verify active hand-held phone distraction.
        Returns:
            distraction_found (bool): True if verified phone or high-confidence distraction object present.
            detections (list of dict): Details of all detected target objects.
        """
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            classes=config.TARGET_OBJECT_CLASSES,
            verbose=False,
            device="cpu"
        )

        detections = []
        has_phone = False

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(v) for v in coords]

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                is_near_face = False
                if face_bbox is not None:
                    fx1, fy1, fx2, fy2 = face_bbox
                    # Expand face zone to cover head + chest/hands area
                    margin_x = (fx2 - fx1) * 1.5
                    margin_y = (fy2 - fy1) * 2.2
                    zone_x1 = max(0, fx1 - margin_x)
                    zone_y1 = max(0, fy1 - margin_y * 0.2)
                    zone_x2 = fx2 + margin_x
                    zone_y2 = fy2 + margin_y

                    if zone_x1 <= cx <= zone_x2 and zone_y1 <= cy <= zone_y2:
                        is_near_face = True

                name = self.CLASS_NAMES.get(cls_id, f"Object {cls_id}")

                if cls_id == config.PHONE_CLASS_ID:
                    has_phone = True

                detections.append({
                    "class_id": cls_id,
                    "name": name,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "near_driver": is_near_face
                })

        if has_phone:
            self.phone_detected_consecutive_frames += 1
        else:
            self.phone_detected_consecutive_frames = max(0, self.phone_detected_consecutive_frames - 1)

        is_phone_distracted = (
            self.phone_detected_consecutive_frames >= config.PHONE_DISTRACTION_CONSEC_FRAMES
        )

        return is_phone_distracted, detections

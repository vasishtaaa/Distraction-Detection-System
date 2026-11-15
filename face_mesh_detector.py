"""
Driver Face Landmark Detection, Eye-Tracking (EAR), Yawn Detection (MAR),
and 3D Head Pose Estimation using Google MediaPipe Face Landmarker.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import config

class DriverFaceAnalyzer:
    """
    Analyzes driver face landmarks in real time for:
    - Eye Aspect Ratio (EAR) -> micro-sleep / drowsiness
    - Mouth Aspect Ratio (MAR) -> yawning fatigue
    - 3D Head Pose (Pitch, Yaw, Roll) -> distraction / looking away
    """

    # 3D generic facial anthropometric model points (in mm)
    MODEL_POINTS_3D = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye outer corner
        (225.0, 170.0, -135.0),   # Right eye outer corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ], dtype=np.float64)

    # Landmark indices for EAR
    # Left eye: [p1(lateral), p2, p3, p4(medial), p5, p6]
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    # Right eye: [p1(lateral), p2, p3, p4(medial), p5, p6]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    # Full contour landmark indices for drawing smooth eye boundaries
    LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    # Mouth landmark indices for MAR
    # Left corner: 61, Right corner: 291
    # Vertical points: (13, 14), (82, 312), (87, 317)
    MOUTH_CORNER_LEFT = 61
    MOUTH_CORNER_RIGHT = 291
    MOUTH_TOP_LIP = 13
    MOUTH_BOTTOM_LIP = 14
    MOUTH_VERT_2_TOP = 82
    MOUTH_VERT_2_BOTTOM = 312
    MOUTH_VERT_3_TOP = 87
    MOUTH_VERT_3_BOTTOM = 317

    # Inner lips contour for drawing
    INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

    # Key landmark indices for 3D Head Pose
    POSE_LANDMARKS_IDX = [1, 152, 33, 263, 61, 291]

    def __init__(self, model_path=config.FACE_LANDMARKER_MODEL):
        self.model_path = model_path
        base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _euclidean_dist(pt1, pt2):
        """Calculates 2D Euclidean distance between two points."""
        return np.linalg.norm(np.array(pt1) - np.array(pt2))

    def _get_pixel_coord(self, landmark, width, height):
        """Converts normalized landmark coordinates into pixel coordinates."""
        return int(landmark.x * width), int(landmark.y * height)

    def analyze_frame(self, frame_bgr):
        """
        Runs face landmarker detection and extracts all fatigue/distraction metrics.
        Returns a dictionary with comprehensive analysis results or None if no face found.
        """
        h, w, _ = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(mp_image)
        if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
            return None

        landmarks = detection_result.face_landmarks[0]

        # 1. Compute Eye Aspect Ratio (EAR)
        left_ear, left_eye_pts = self._compute_ear_for_eye(landmarks, self.LEFT_EYE_IDX, w, h)
        right_ear, right_eye_pts = self._compute_ear_for_eye(landmarks, self.RIGHT_EYE_IDX, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        # 2. Compute Mouth Aspect Ratio (MAR)
        mar, mouth_pts = self._compute_mar(landmarks, w, h)

        # 3. Estimate Head Pose (Pitch, Yaw, Roll)
        pitch, yaw, roll, nose_2d, gaze_endpoint_2d = self._estimate_head_pose(landmarks, w, h)

        # 4. Extract Face Bounding Box
        all_xs = [lm.x * w for lm in landmarks]
        all_ys = [lm.y * h for lm in landmarks]
        face_bbox = (
            max(0, int(min(all_xs)) - 15),
            max(0, int(min(all_ys)) - 25),
            min(w, int(max(all_xs)) + 15),
            min(h, int(max(all_ys)) + 15)
        )

        return {
            "landmarks_raw": landmarks,
            "face_bbox": face_bbox,
            "ear": avg_ear,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "left_eye_pts": left_eye_pts,
            "right_eye_pts": right_eye_pts,
            "mar": mar,
            "mouth_pts": mouth_pts,
            "head_pitch": pitch,
            "head_yaw": yaw,
            "head_roll": roll,
            "nose_2d": nose_2d,
            "gaze_endpoint_2d": gaze_endpoint_2d,
            "is_eyes_closed": avg_ear < config.EAR_THRESHOLD,
            "is_yawning": mar > config.MAR_THRESHOLD,
            "is_head_distracted": (
                abs(yaw) > config.HEAD_YAW_THRESHOLD or
                pitch > config.HEAD_PITCH_DOWN_THRESHOLD or
                pitch < -config.HEAD_PITCH_UP_THRESHOLD
            )
        }

    def _compute_ear_for_eye(self, landmarks, idx_list, w, h):
        """
        Computes EAR for a 6-point eye definition:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        pts = [self._get_pixel_coord(landmarks[idx], w, h) for idx in idx_list]
        p1, p2, p3, p4, p5, p6 = pts

        d_v1 = self._euclidean_dist(p2, p6)
        d_v2 = self._euclidean_dist(p3, p5)
        d_h = self._euclidean_dist(p1, p4)

        if d_h == 0:
            return 0.0, pts
        ear = (d_v1 + d_v2) / (2.0 * d_h)
        return float(ear), pts

    def _compute_mar(self, landmarks, w, h):
        """
        Computes Mouth Aspect Ratio (MAR) to detect yawning:
        MAR = (||top - bottom|| + ||v2_top - v2_bottom|| + ||v3_top - v3_bottom||) / (3 * ||left - right||)
        """
        p_left = self._get_pixel_coord(landmarks[self.MOUTH_CORNER_LEFT], w, h)
        p_right = self._get_pixel_coord(landmarks[self.MOUTH_CORNER_RIGHT], w, h)
        p_top = self._get_pixel_coord(landmarks[self.MOUTH_TOP_LIP], w, h)
        p_bottom = self._get_pixel_coord(landmarks[self.MOUTH_BOTTOM_LIP], w, h)
        p_v2_t = self._get_pixel_coord(landmarks[self.MOUTH_VERT_2_TOP], w, h)
        p_v2_b = self._get_pixel_coord(landmarks[self.MOUTH_VERT_2_BOTTOM], w, h)
        p_v3_t = self._get_pixel_coord(landmarks[self.MOUTH_VERT_3_TOP], w, h)
        p_v3_b = self._get_pixel_coord(landmarks[self.MOUTH_VERT_3_BOTTOM], w, h)

        d_v1 = self._euclidean_dist(p_top, p_bottom)
        d_v2 = self._euclidean_dist(p_v2_t, p_v2_b)
        d_v3 = self._euclidean_dist(p_v3_t, p_v3_b)
        d_h = self._euclidean_dist(p_left, p_right)

        if d_h == 0:
            return 0.0, (p_left, p_right, p_top, p_bottom)

        mar = (d_v1 + d_v2 + d_v3) / (3.0 * d_h)
        return float(mar), (p_left, p_right, p_top, p_bottom)

    def _estimate_head_pose(self, landmarks, w, h):
        """
        Estimates driver's 3D head pose (Pitch, Yaw, Roll) via solvePnP.
        Pitch: positive = looking down, negative = looking up
        Yaw: positive = turned right, negative = turned left
        Roll: tilt sideways
        """
        image_points_2d = np.array([
            self._get_pixel_coord(landmarks[idx], w, h)
            for idx in self.POSE_LANDMARKS_IDX
        ], dtype=np.float64)

        focal_length = w
        optical_center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, optical_center[0]],
            [0, focal_length, optical_center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0, (int(w/2), int(h/2)), (int(w/2), int(h/2))

        # Convert rotation vector to rotation matrix
        rot_matrix, _ = cv2.Rodrigues(rvec)

        # Decompose rotation matrix into Euler angles
        # RQDecomp3x3 returns angles in degrees: [Pitch, Yaw, Roll]
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_matrix)
        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        # Project 3D gaze vector from nose tip (0, 0, 500mm ahead)
        nose_point_3d = np.array([(0.0, 0.0, 500.0)], dtype=np.float64)
        gaze_endpoint_2d, _ = cv2.projectPoints(
            nose_point_3d, rvec, tvec, camera_matrix, dist_coeffs
        )

        nose_2d = (int(image_points_2d[0][0]), int(image_points_2d[0][1]))
        gaze_endpoint = (
            int(gaze_endpoint_2d[0][0][0]),
            int(gaze_endpoint_2d[0][0][1])
        )

        return pitch, yaw, roll, nose_2d, gaze_endpoint

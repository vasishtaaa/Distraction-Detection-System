"""
Real-Time Driver Safety & Distraction Detection System
Configuration Parameters and Thresholds
"""

import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "session_reports")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

# Pretrained Model Paths
FACE_LANDMARKER_MODEL = os.path.join(MODELS_DIR, "face_landmarker.task")
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")

# Camera & Video Capture Settings
DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

# Drowsiness / Fatigue Detection Thresholds
# Eye Aspect Ratio (EAR)
EAR_THRESHOLD = 0.21               # EAR below this is considered eye closed
EAR_CONSEC_FRAMES = 25             # ~0.8s - 1.0s at 30fps for micro-sleep warning
DROWSY_TIME_THRESHOLD_SEC = 1.5    # Continuous seconds of eye closure for critical alert

# Mouth Aspect Ratio (MAR) for Yawning
MAR_THRESHOLD = 0.65               # MAR above this is considered yawning
MAR_CONSEC_FRAMES = 30             # Frames required to confirm sustained yawn
YAWN_COOLDOWN_SEC = 5.0            # Seconds before registering another yawn

# Head Pose & Gaze Distraction Thresholds
HEAD_YAW_THRESHOLD = 25.0          # Degrees left/right to trigger off-road distraction
HEAD_PITCH_DOWN_THRESHOLD = 18.0   # Degrees down (looking at phone/lap)
HEAD_PITCH_UP_THRESHOLD = 25.0     # Degrees up
DISTRACTION_TIME_SEC = 1.8         # Sustained seconds looking away before alert

# YOLOv8 Distraction Detection Thresholds
YOLO_CONFIDENCE = 0.35
# COCO class IDs: 67 = cell phone, 39 = bottle, 41 = cup
TARGET_OBJECT_CLASSES = [67, 39, 41]
PHONE_CLASS_ID = 67
PHONE_DISTRACTION_CONSEC_FRAMES = 10  # Consecutive frames to trigger phone alert

# Alert System Settings
AUDIO_ALERTS_ENABLED = True
ALERT_COOLDOWN_SEC = 2.5           # Min seconds between audio alerts
BEEP_FREQUENCY = 1500              # Hz for warning beep
BEEP_DURATION_MS = 350             # ms

# UI & HUD Styling (BGR Colors)
COLOR_SAFE = (72, 199, 116)         # Soft Emerald Green
COLOR_WARNING = (30, 180, 255)      # Amber / Orange
COLOR_DANGER = (50, 50, 245)        # Vivid Crimson Red
COLOR_PRIMARY = (255, 165, 0)       # Cyan/Blue Accent
COLOR_CARD_BG = (20, 24, 30)        # Dark Charcoal
COLOR_TEXT = (245, 245, 245)        # Crisp White
COLOR_MUTED = (160, 160, 160)       # Light Slate Gray

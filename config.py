"""
Real-Time Distraction Detection System (DDS)
Configuration Parameters, Thresholds, and Operation Modes.
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

# Operation Modes
MODE_STUDY = "STUDY"          # Online classes, studying, deep work, proctoring
MODE_DRIVING = "DRIVING"      # Automotive driving & road safety
MODE_GENERAL = "GENERAL"      # General real-time attention & distraction detection
DEFAULT_MODE = MODE_GENERAL

# Default Video Capture Settings
DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

# Eye Tracking & Drowsiness Thresholds (EAR)
EAR_THRESHOLD = 0.21               # EAR below this is considered eye closed
DROWSY_TIME_THRESHOLD_SEC = 1.5    # Continuous eye closure for drowsiness alert

# Mouth Aspect Ratio (MAR) for Yawning / Fatigue
MAR_THRESHOLD = 0.65               # MAR above this is considered yawning
MAR_CONSEC_FRAMES = 25
YAWN_COOLDOWN_SEC = 4.0

# Head Pose & Gaze Distraction Thresholds
HEAD_YAW_THRESHOLD = 28.0          # Degrees left/right to trigger looking-away distraction
HEAD_PITCH_DOWN_THRESHOLD = 26.0   # Degrees down (looking at phone/lap/desk)
HEAD_PITCH_UP_THRESHOLD = 30.0     # Degrees up (staring at ceiling)
HEAD_DISTRACTION_CONSEC_FRAMES = 14 # Consecutive frames to confirm looking away (~0.7s)
DISTRACTION_TIME_SEC = 1.5         # Sustained seconds looking away before incident log

# YOLOv8 External Object Distraction Thresholds
YOLO_CONFIDENCE = 0.35
# COCO class IDs: 67 = cell phone, 39 = bottle, 41 = cup, 76 = scissors/tools
TARGET_OBJECT_CLASSES = [67, 39, 41]
PHONE_CLASS_ID = 67
PHONE_DISTRACTION_CONSEC_FRAMES = 8  # Debounce consecutive frames to trigger phone alert

# Alert System Settings
AUDIO_ALERTS_ENABLED = True
ALERT_COOLDOWN_SEC = 2.0           # Minimum seconds between audio alerts
BEEP_FREQUENCY = 1600              # Hz for warning tone
BEEP_DURATION_MS = 250             # ms

# Aesthetic UI Theme Palette (BGR for OpenCV, Hex for Web)
COLOR_SAFE = (118, 230, 0)          # Vibrant Spring Green (00e676)
COLOR_WARNING = (0, 171, 255)       # Amber Gold (ffab00)
COLOR_DANGER = (68, 23, 255)        # Vivid Crimson Red (ff1744)
COLOR_PRIMARY = (255, 210, 0)       # Cyan Accent (00d2ff)
COLOR_CARD_BG = (22, 28, 40)        # Midnight Navy Glass
COLOR_TEXT = (245, 248, 250)        # Pure Crisp White
COLOR_MUTED = (165, 180, 195)       # Soft Slate Gray

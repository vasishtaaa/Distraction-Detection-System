import os
import sys
import urllib.request
import requests

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_LANDMARKER_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

YOLO_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")

def download_file(url, target_path, name):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        print(f"[OK] {name} already exists at {target_path} ({os.path.getsize(target_path)} bytes)")
        return True
    print(f"[DOWNLOADING] {name} from {url} ...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response, open(target_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"[SUCCESS] Downloaded {name} -> {target_path} ({os.path.getsize(target_path)} bytes)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {name}: {e}")
        return False

def ensure_models():
    print("=== Downloading Required Pretrained Models ===")
    # 1. MediaPipe Face Landmarker
    download_file(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH, "MediaPipe Face Landmarker")
    
    # 2. YOLOv8n
    if not (os.path.exists(YOLO_PATH) and os.path.getsize(YOLO_PATH) > 1000):
        print("[DOWNLOADING] YOLOv8n model...")
        try:
            from ultralytics import YOLO
            # Downloading via ultralytics directly
            model = YOLO("yolov8n.pt")
            # If it downloaded to root or cache, copy to models/
            import shutil
            if os.path.exists("yolov8n.pt"):
                shutil.copy("yolov8n.pt", YOLO_PATH)
                print(f"[SUCCESS] Saved YOLOv8n to {YOLO_PATH}")
        except Exception as e:
            print(f"[INFO] Ultralytics direct download error: {e}. Trying direct asset URL...")
            urls = [
                "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
                "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
                "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
            ]
            for u in urls:
                if download_file(u, YOLO_PATH, "YOLOv8n"):
                    break

if __name__ == "__main__":
    ensure_models()

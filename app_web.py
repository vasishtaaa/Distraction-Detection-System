"""
FastAPI Web Dashboard Interface for Driver Safety & Distraction Detection System.
Provides a modern, high-tech automotive browser UI at http://127.0.0.1:8000
with live MJPEG video streaming, interactive HUD controls, and real-time telemetry.
"""

import os
import sys
import time
import cv2
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "."), "matplotlib")

import config
from face_mesh_detector import DriverFaceAnalyzer
from yolo_detector import DistractionObjectDetector
from alert_system import AlertManager, AlertType
from analytics import DriverTelemetry
from ui_overlay import HUDOverlay

app = FastAPI(title="Driver Safety & Distraction Detection System")

class VideoStreamer:
    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.face_analyzer = DriverFaceAnalyzer()
        self.yolo_detector = DistractionObjectDetector()
        self.alert_manager = AlertManager(enabled=False) # Mute audio by default in web mode
        self.telemetry = DriverTelemetry()
        self.hud = HUDOverlay()
        self.fps = 30.0
        self.prev_time = time.time()
        self.open_source(source)

    def open_source(self, source):
        if self.cap is not None:
            self.cap.release()
        self.source = source
        if str(source).isdigit():
            source_idx = int(source)
            self.cap = cv2.VideoCapture(source_idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(source_idx)
        else:
            self.cap = cv2.VideoCapture(str(source))

        if not self.cap.isOpened() and os.path.exists("sample_driver_demo.mp4"):
            self.source = "sample_driver_demo.mp4"
            self.cap = cv2.VideoCapture("sample_driver_demo.mp4")

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            # Loop playback for demo video
            if isinstance(self.source, str) and os.path.exists(self.source):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if not ret:
                return None

        curr_time = time.time()
        delta = curr_time - self.prev_time
        self.prev_time = curr_time
        if delta > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / delta)

        face_data = self.face_analyzer.analyze_frame(frame)
        face_bbox = face_data["face_bbox"] if face_data else None
        yolo_distracted, detections = self.yolo_detector.detect(frame, face_bbox=face_bbox)

        if face_data and face_data["is_eyes_closed"]:
            self.alert_manager.trigger(AlertType.DROWSINESS)
        elif yolo_distracted:
            self.alert_manager.trigger(AlertType.PHONE)
        elif face_data and face_data["is_head_distracted"]:
            self.alert_manager.trigger(AlertType.LOOKING_AWAY)
        elif face_data and face_data["is_yawning"]:
            self.alert_manager.trigger(AlertType.YAWN)
        else:
            self.alert_manager.clear()

        is_eyes_closed = face_data["is_eyes_closed"] if face_data else False
        is_yawning = face_data["is_yawning"] if face_data else False
        is_head_distracted = face_data["is_head_distracted"] if face_data else False

        self.telemetry.update(
            is_eyes_closed=is_eyes_closed,
            is_yawning=is_yawning,
            is_head_distracted=is_head_distracted,
            is_phone_distracted=yolo_distracted,
            fps=self.fps
        )

        display_frame = self.hud.render(
            frame=frame,
            face_data=face_data,
            yolo_distracted=yolo_distracted,
            detections=detections,
            alert_manager=self.alert_manager,
            telemetry=self.telemetry,
            fps=self.fps
        )

        ret, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buffer.tobytes()

streamer = None

def get_streamer():
    global streamer
    if streamer is None:
        streamer = VideoStreamer(source=0)
    return streamer

def generate_frames():
    s = get_streamer()
    while True:
        frame_bytes = s.get_frame()
        if frame_bytes is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/telemetry")
def get_telemetry():
    s = get_streamer()
    summary = s.telemetry.get_summary()
    return JSONResponse(content=summary)

@app.post("/api/toggle_mesh")
def toggle_mesh():
    s = get_streamer()
    state = s.hud.toggle_debug_mesh()
    return {"mesh_enabled": state}

@app.post("/api/reset")
def reset_telemetry():
    s = get_streamer()
    s.telemetry = DriverTelemetry()
    return {"status": "reset_successful"}

@app.post("/api/switch_camera")
def switch_camera(camera: str = "0"):
    s = get_streamer()
    s.open_source(camera)
    return {"status": f"switched_to_{camera}"}

@app.get("/", response_class=HTMLResponse)
def index_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Driver Safety & Distraction Monitor - Web Dashboard</title>
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(22, 28, 45, 0.85);
            --border-color: rgba(60, 75, 110, 0.4);
            --accent-blue: #00d2ff;
            --safe-green: #00e676;
            --warn-amber: #ffab00;
            --danger-red: #ff1744;
            --text-main: #f0f4f8;
            --text-muted: #8a9bb5;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: var(--bg-color); color: var(--text-main); min-height: 100vh; display: flex; flex-direction: column; }
        
        header {
            background: linear-gradient(180deg, rgba(16, 24, 40, 0.95), rgba(11, 15, 25, 0.95));
            border-bottom: 1px solid var(--border-color);
            padding: 16px 28px;
            display: flex; justify-content: space-between; align-items: center;
        }
        .header-title { display: flex; align-items: center; gap: 12px; }
        .logo-icon { width: 14px; height: 14px; border-radius: 50%; background: var(--accent-blue); box-shadow: 0 0 12px var(--accent-blue); }
        h1 { font-size: 1.15rem; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; }
        .badge { background: rgba(0, 210, 255, 0.15); color: var(--accent-blue); border: 1px solid var(--accent-blue); padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
        
        main { display: grid; grid-template-columns: 1fr 340px; gap: 20px; padding: 20px; flex: 1; }
        
        .video-card {
            background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px;
            overflow: hidden; display: flex; flex-direction: column; box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        }
        .video-header { padding: 12px 18px; border-bottom: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; }
        .video-container { position: relative; width: 100%; flex: 1; display: flex; align-items: center; justify-content: center; background: #000; min-height: 480px; }
        .video-container img { width: 100%; height: auto; max-height: 640px; object-fit: contain; }
        
        .controls-toolbar { padding: 14px 18px; border-top: 1px solid var(--border-color); display: flex; gap: 10px; flex-wrap: wrap; }
        button {
            background: rgba(30, 42, 65, 0.8); color: var(--text-main); border: 1px solid var(--border-color);
            padding: 8px 16px; border-radius: 8px; font-size: 0.82rem; font-weight: 600; cursor: pointer;
            transition: all 0.2s ease; display: inline-flex; align-items: center; gap: 6px;
        }
        button:hover { background: rgba(0, 210, 255, 0.2); border-color: var(--accent-blue); }
        
        .sidebar { display: flex; flex-direction: column; gap: 16px; }
        .widget-card { background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 16px; box-shadow: 0 6px 20px rgba(0,0,0,0.3); }
        .widget-title { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-muted); margin-bottom: 12px; font-weight: 700; }
        
        .score-circle { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 10px 0; }
        .score-val { font-size: 2.8rem; font-weight: 800; color: var(--safe-green); }
        .score-grade { font-size: 0.9rem; font-weight: 600; color: var(--text-muted); margin-top: 2px; }
        
        .stat-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.06); font-size: 0.85rem; }
        .stat-row:last-child { border-bottom: none; }
        .stat-label { color: var(--text-muted); }
        .stat-value { font-weight: 600; color: var(--text-main); }
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <div class="logo-icon"></div>
            <h1>Driver Safety & Distraction Monitor</h1>
            <span class="badge">PROTOTYPE LIVE</span>
        </div>
        <div id="live-timer" style="font-size: 0.85rem; color: var(--text-muted);">TIME: 00:00</div>
    </header>

    <main>
        <div class="video-card">
            <div class="video-header">
                <div>LIVE AUTOMOTIVE HUD STREAM</div>
                <div id="fps-badge" style="color: var(--accent-blue); font-weight: 600;">-- FPS</div>
            </div>
            <div class="video-container">
                <img src="/video_feed" alt="Driver HUD Video Feed">
            </div>
            <div class="controls-toolbar">
                <button onclick="switchCamera('0')">Integrated Camera (0)</button>
                <button onclick="switchCamera('1')">Camera (1)</button>
                <button onclick="switchCamera('sample_driver_demo.mp4')">Scenario Demo</button>
                <button onclick="toggleMesh()">Toggle 3D Mesh</button>
                <button onclick="resetStats()">Reset Stats</button>
            </div>
        </div>

        <div class="sidebar">
            <div class="widget-card">
                <div class="widget-title">Driver Safety Score</div>
                <div class="score-circle">
                    <div class="score-val" id="score-val">100</div>
                    <div class="score-grade" id="score-grade">A+ (Excellent)</div>
                </div>
            </div>

            <div class="widget-card">
                <div class="widget-title">Live Telemetry</div>
                <div class="stat-row"><span class="stat-label">PERCLOS (Fatigue):</span><span class="stat-value" id="val-perclos">0.0%</span></div>
                <div class="stat-row"><span class="stat-label">Blinks Tracked:</span><span class="stat-value" id="val-blinks">0</span></div>
                <div class="stat-row"><span class="stat-label">Microsleeps:</span><span class="stat-value" id="val-microsleeps">0</span></div>
                <div class="stat-row"><span class="stat-label">Yawn Episodes:</span><span class="stat-value" id="val-yawns">0</span></div>
                <div class="stat-row"><span class="stat-label">Head Distractions:</span><span class="stat-value" id="val-head">0</span></div>
                <div class="stat-row"><span class="stat-label">Phone Distractions:</span><span class="stat-value" id="val-phone">0</span></div>
                <div class="stat-row"><span class="stat-label">Total Frames:</span><span class="stat-value" id="val-frames">0</span></div>
            </div>
        </div>
    </main>

    <script>
        async function fetchTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                if (res.ok) {
                    const data = await res.json();
                    document.getElementById('score-val').innerText = Math.round(data.safety_score);
                    document.getElementById('score-grade').innerText = data.rating_grade;
                    document.getElementById('val-perclos').innerText = data.perclos_percentage + '%';
                    document.getElementById('val-blinks').innerText = data.blink_count + ' (' + data.blinks_per_minute + ' BPM)';
                    document.getElementById('val-microsleeps').innerText = data.microsleep_episodes;
                    document.getElementById('val-yawns').innerText = data.yawn_episodes;
                    document.getElementById('val-head').innerText = data.head_distraction_episodes;
                    document.getElementById('val-phone').innerText = data.phone_usage_episodes;
                    document.getElementById('val-frames').innerText = data.total_frames;
                    document.getElementById('fps-badge').innerText = data.average_fps + ' FPS';
                    document.getElementById('live-timer').innerText = 'TIME: ' + data.duration_formatted;
                }
            } catch (e) {}
        }
        setInterval(fetchTelemetry, 1000);

        async function switchCamera(src) {
            await fetch('/api/switch_camera?camera=' + encodeURIComponent(src), { method: 'POST' });
        }
        async function toggleMesh() {
            await fetch('/api/toggle_mesh', { method: 'POST' });
        }
        async function resetStats() {
            await fetch('/api/reset', { method: 'POST' });
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("=" * 65)
    print("  DRIVER SAFETY & DISTRACTION MONITOR - WEB DASHBOARD")
    print("  Serving at: http://127.0.0.1:8000")
    print("=" * 65)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

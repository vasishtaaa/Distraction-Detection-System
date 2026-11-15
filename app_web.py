"""
Unified, Modern Web Dashboard for Distraction Detection System (DDS).
Provides a responsive, high-tech interface with real-time video streaming,
mode-switching, live telemetry gauges, and chronological incident logging.
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
from camera import ThreadedCamera
from face_mesh_detector import DriverFaceAnalyzer
from yolo_detector import DistractionObjectDetector
from alert_system import AlertManager, AlertType
from analytics import AttentionTelemetry
from ui_overlay import HUDOverlay

app = FastAPI(title="Distraction Detection System (DDS)")

class DDSProcessor:
    """
    Coordinates threaded camera acquisition, AI inference, and HUD rendering.
    """

    def __init__(self, initial_source=0, mode=config.DEFAULT_MODE):
        self.mode = mode
        self.camera = ThreadedCamera(source=initial_source, loop_video=True)
        self.face_analyzer = DriverFaceAnalyzer()
        self.yolo_detector = DistractionObjectDetector()
        self.alert_manager = AlertManager(enabled=False) # Visual alerts by default in browser
        self.telemetry = AttentionTelemetry(mode=self.mode)
        self.hud = HUDOverlay()
        self.fps = 30.0
        self.prev_time = time.time()
        self.is_paused = False

    def change_source(self, new_source):
        self.camera.open_source(new_source)

    def set_mode(self, new_mode):
        self.mode = new_mode
        self.telemetry.set_mode(new_mode)

    def process_next_frame(self):
        ret, frame = self.camera.read()
        if not ret or frame is None:
            return None

        if self.is_paused:
            ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return buf.tobytes()

        # Measure FPS
        curr_time = time.time()
        delta = curr_time - self.prev_time
        self.prev_time = curr_time
        if delta > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / delta)

        # 1. MediaPipe Face Landmark & Head Pose
        face_data = self.face_analyzer.analyze_frame(frame)
        face_bbox = face_data["face_bbox"] if face_data else None

        # 2. YOLOv8 Object Detection
        yolo_distracted, detections = self.yolo_detector.detect(frame, face_bbox=face_bbox)

        # 3. Rule Evaluation
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

        # 4. Telemetry Update
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

        # 5. Render HUD Overlay
        annotated_frame = self.hud.render(
            frame=frame,
            face_data=face_data,
            yolo_distracted=yolo_distracted,
            detections=detections,
            alert_manager=self.alert_manager,
            telemetry=self.telemetry,
            fps=self.fps
        )

        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buffer.tobytes()

dds = None

def get_dds():
    global dds
    if dds is None:
        dds = DDSProcessor(initial_source=0)
    return dds

def stream_generator():
    processor = get_dds()
    while True:
        frame_bytes = processor.process_next_frame()
        if frame_bytes is None:
            time.sleep(0.04)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/telemetry")
def get_telemetry():
    p = get_dds()
    return JSONResponse(content=p.telemetry.get_summary())

@app.post("/api/set_mode")
def set_mode(mode: str = "GENERAL"):
    p = get_dds()
    p.set_mode(mode.upper())
    return {"status": "mode_updated", "mode": p.mode}

@app.post("/api/switch_source")
def switch_source(source: str = "0"):
    p = get_dds()
    p.change_source(source)
    return {"status": "source_switched", "source": source}

@app.post("/api/toggle_mesh")
def toggle_mesh():
    p = get_dds()
    state = p.hud.toggle_debug_mesh()
    return {"mesh_enabled": state}

@app.post("/api/toggle_mute")
def toggle_mute():
    p = get_dds()
    state = p.alert_manager.toggle_mute()
    return {"audio_enabled": state}

@app.post("/api/reset")
def reset_session():
    p = get_dds()
    p.telemetry = AttentionTelemetry(mode=p.mode)
    return {"status": "reset_successful"}

@app.get("/", response_class=HTMLResponse)
def dashboard_html():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Distraction Detection System (DDS)</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #080b11;
            --bg-surface: #101622;
            --bg-card: rgba(20, 27, 42, 0.75);
            --border-glass: rgba(255, 255, 255, 0.08);
            --border-focus: rgba(0, 210, 255, 0.4);
            
            --cyan: #00d2ff;
            --green: #00e676;
            --amber: #ffab00;
            --red: #ff1744;
            --purple: #b388ff;
            
            --text-primary: #f0f4f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-base);
            color: var(--text-primary);
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        /* Top Navigation Header */
        header {
            background: rgba(16, 22, 34, 0.85);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border-glass);
            padding: 14px 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand-container { display: flex; align-items: center; gap: 14px; }
        .brand-pill {
            width: 32px; height: 32px; border-radius: 8px;
            background: linear-gradient(135deg, var(--cyan), #0070f3);
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 0 16px rgba(0, 210, 255, 0.35);
        }
        .brand-pill svg { width: 18px; height: 18px; fill: #fff; }
        .brand-title { font-size: 1.1rem; font-weight: 800; letter-spacing: -0.2px; }
        .brand-title span { color: var(--cyan); }

        .mode-selector {
            display: flex; background: rgba(0,0,0,0.3); border: 1px solid var(--border-glass);
            padding: 3px; border-radius: 10px; gap: 4px;
        }
        .mode-btn {
            background: transparent; color: var(--text-secondary); border: none;
            padding: 6px 14px; border-radius: 7px; font-size: 0.8rem; font-weight: 600;
            cursor: pointer; transition: all 0.2s ease;
        }
        .mode-btn.active {
            background: var(--cyan); color: #041019;
            box-shadow: 0 2px 10px rgba(0, 210, 255, 0.35);
        }

        .header-meta { display: flex; align-items: center; gap: 16px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
        .live-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 10px var(--green); animation: pulse 2s infinite; }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.9); }
        }

        /* Main Workspace Grid */
        main {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 24px;
            padding: 24px;
            max-width: 1720px;
            margin: 0 auto;
            width: 100%;
            flex: 1;
        }

        @media (max-width: 1100px) {
            main { grid-template-columns: 1fr; }
        }

        /* Video Showcase */
        .video-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-glass);
            border-radius: 16px;
            backdrop-filter: blur(20px);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        }

        .video-meta-bar {
            padding: 12px 20px;
            border-bottom: 1px solid var(--border-glass);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.82rem;
            color: var(--text-secondary);
        }

        .video-viewport {
            background: #020408;
            position: relative;
            width: 100%;
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 480px;
        }

        .video-viewport img {
            width: 100%;
            height: auto;
            max-height: 620px;
            object-fit: contain;
            display: block;
        }

        /* Video Toolbar */
        .video-controls {
            padding: 14px 20px;
            border-top: 1px solid var(--border-glass);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            background: rgba(10, 14, 22, 0.6);
        }

        .control-group { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }

        .btn {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
            border: 1px solid var(--border-glass);
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .btn:hover {
            background: rgba(0, 210, 255, 0.15);
            border-color: var(--cyan);
            color: #fff;
        }
        .btn-danger:hover {
            background: rgba(255, 23, 68, 0.2);
            border-color: var(--red);
        }

        /* Sidebar Analytics */
        .sidebar { display: flex; flex-direction: column; gap: 20px; }

        .glass-card {
            background: var(--bg-card);
            border: 1px solid var(--border-glass);
            border-radius: 16px;
            backdrop-filter: blur(20px);
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        .card-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 700;
            color: var(--text-secondary);
        }

        /* Circular Score Meter */
        .score-hero {
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 10px 0;
        }

        .score-dial {
            position: relative;
            width: 105px;
            height: 105px;
            flex-shrink: 0;
        }
        .score-dial svg { transform: rotate(-90deg); width: 105px; height: 105px; }
        .dial-track { fill: none; stroke: rgba(255, 255, 255, 0.08); stroke-width: 8; }
        .dial-fill {
            fill: none;
            stroke: var(--green);
            stroke-width: 8;
            stroke-linecap: round;
            stroke-dasharray: 283;
            stroke-dashoffset: 28;
            transition: stroke-dashoffset 0.6s ease, stroke 0.3s ease;
        }
        .dial-text {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.65rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
        }

        .score-details { display: flex; flex-direction: column; gap: 4px; }
        .score-status { font-size: 1.15rem; font-weight: 700; color: var(--green); }
        .score-sub { font-size: 0.82rem; color: var(--text-muted); }

        /* Telemetry Stat Grid */
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 4px;
        }
        .stat-box {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-glass);
            border-radius: 10px;
            padding: 12px;
        }
        .stat-label { font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px; }
        .stat-num { font-size: 1.15rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

        /* Incident Timeline */
        .incident-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-height: 220px;
            overflow-y: auto;
            margin-top: 8px;
        }
        .incident-list::-webkit-scrollbar { width: 4px; }
        .incident-list::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 4px; }

        .incident-item {
            background: rgba(255, 255, 255, 0.02);
            border-left: 3px solid var(--amber);
            padding: 8px 12px;
            border-radius: 0 8px 8px 0;
            font-size: 0.8rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .incident-item.danger { border-left-color: var(--red); }
        .incident-type { font-weight: 600; color: var(--text-primary); }
        .incident-time { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--text-muted); }

        .empty-placeholder {
            text-align: center;
            padding: 24px 0;
            color: var(--text-muted);
            font-size: 0.82rem;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand-container">
            <div class="brand-pill">
                <svg viewBox="0 0 24 24"><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/></svg>
            </div>
            <div class="brand-title">Distraction Detection <span>System</span></div>
        </div>

        <div style="display: flex; align-items: center; gap: 8px;">
            <div class="mode-selector">
                <button class="mode-btn active" id="btn-mode-general" onclick="switchMode('GENERAL')">1 General</button>
                <button class="mode-btn" id="btn-mode-study" onclick="switchMode('STUDY')">2 Study & Work</button>
                <button class="mode-btn" id="btn-mode-driving" onclick="switchMode('DRIVING')">3 Driving</button>
            </div>
            <span style="font-size: 0.72rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;">[Tab / 1-3]</span>
        </div>

        <div class="header-meta">
            <div class="live-dot"></div>
            <span id="session-clock">00:00</span>
        </div>
    </header>

    <main>
        <!-- Video Feed Section -->
        <div class="video-panel">
            <div class="video-meta-bar">
                <span>REAL-TIME AI VISION HUD</span>
                <span id="fps-display" style="color: var(--cyan); font-family: 'JetBrains Mono';">30.0 FPS</span>
            </div>

            <div class="video-viewport">
                <img id="feed-img" src="/video_feed" alt="Video Feed">
            </div>

            <div class="video-controls">
                <div class="control-group">
                    <button class="btn" onclick="switchSource('0')">Camera 0 (Integrated)</button>
                    <button class="btn" onclick="switchSource('1')">Camera 1</button>
                    <button class="btn" onclick="switchSource('sample_driver_demo.mp4')">Demo Simulation</button>
                </div>
                <div class="control-group">
                    <button class="btn" onclick="toggleMesh()">Toggle 3D Mesh</button>
                    <button class="btn btn-danger" onclick="resetStats()">Reset Stats</button>
                </div>
            </div>
        </div>

        <!-- Sidebar Telemetry -->
        <div class="sidebar">
            <!-- Score Card -->
            <div class="glass-card">
                <div class="card-header">
                    <div class="card-title">Real-Time Attention Index</div>
                </div>
                <div class="score-hero">
                    <div class="score-dial">
                        <svg viewBox="0 0 100 100">
                            <circle class="dial-track" cx="50" cy="50" r="45"></circle>
                            <circle class="dial-fill" id="dial-circle" cx="50" cy="50" r="45"></circle>
                        </svg>
                        <div class="dial-text" id="score-text">100</div>
                    </div>
                    <div class="score-details">
                        <div class="score-status" id="score-status">Focused & Alert</div>
                        <div class="score-sub" id="score-sub">Zero inattention detected</div>
                    </div>
                </div>
            </div>

            <!-- Key Metrics Grid -->
            <div class="glass-card">
                <div class="card-header">
                    <div class="card-title">Live Telemetry Metrics</div>
                </div>
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-label">Phone Distractions</div>
                        <div class="stat-num" id="stat-phone" style="color: var(--red);">0</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Looking Away</div>
                        <div class="stat-num" id="stat-head" style="color: var(--amber);">0</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Fatigue / Drowsiness</div>
                        <div class="stat-num" id="stat-sleep" style="color: var(--purple);">0</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Blinks (Rate)</div>
                        <div class="stat-num" id="stat-blinks" style="color: var(--cyan);">0</div>
                    </div>
                </div>
            </div>

            <!-- Incident Log Timeline -->
            <div class="glass-card" style="flex: 1;">
                <div class="card-header">
                    <div class="card-title">Chronological Event Timeline</div>
                </div>
                <div class="incident-list" id="incident-container">
                    <div class="empty-placeholder">No distraction incidents logged yet.</div>
                </div>
            </div>
        </div>
    </main>

    <script>
        async function updateDashboard() {
            try {
                const res = await fetch('/api/telemetry');
                if (res.ok) {
                    const data = await res.json();
                    
                    // 1. Score Dial Animation
                    const score = Math.round(data.attention_score);
                    document.getElementById('score-text').innerText = score;
                    const offset = 283 - (283 * (score / 100));
                    const dial = document.getElementById('dial-circle');
                    dial.style.strokeDashoffset = offset;
                    
                    if (score >= 80) {
                        dial.style.stroke = 'var(--green)';
                        document.getElementById('score-status').innerText = 'Focused & Alert';
                        document.getElementById('score-status').style.color = 'var(--green)';
                    } else if (score >= 60) {
                        dial.style.stroke = 'var(--amber)';
                        document.getElementById('score-status').innerText = 'Minor Inattention';
                        document.getElementById('score-status').style.color = 'var(--amber)';
                    } else {
                        dial.style.stroke = 'var(--red)';
                        document.getElementById('score-status').innerText = 'Distraction Alert';
                        document.getElementById('score-status').style.color = 'var(--red)';
                    }
                    document.getElementById('score-sub').innerText = data.rating;

                    // 2. Metrics
                    document.getElementById('stat-phone').innerText = data.phone_usage_episodes;
                    document.getElementById('stat-head').innerText = data.head_distraction_episodes;
                    document.getElementById('stat-sleep').innerText = data.microsleep_episodes + ' (PERCLOS ' + data.perclos_percentage + '%)';
                    document.getElementById('stat-blinks').innerText = data.blink_count + ' (' + data.blinks_per_minute + ' BPM)';
                    document.getElementById('session-clock').innerText = data.duration_formatted;
                    document.getElementById('fps-display').innerText = data.average_fps + ' FPS';

                    // 3. Incident List
                    const container = document.getElementById('incident-container');
                    if (data.recent_incidents && data.recent_incidents.length > 0) {
                        container.innerHTML = '';
                        data.recent_incidents.slice().reverse().forEach(inc => {
                            const item = document.createElement('div');
                            const isDanger = inc.type.includes('PHONE') || inc.type.includes('DROWSY');
                            item.className = 'incident-item ' + (isDanger ? 'danger' : '');
                            item.innerHTML = `
                                <span class="incident-type">${inc.type.replace('_', ' ')}</span>
                                <span class="incident-time">${inc.timestamp} (+${inc.duration_sec}s)</span>
                            `;
                            container.appendChild(item);
                        });
                    }
                }
            } catch (err) {}
        }

        setInterval(updateDashboard, 800);

        const MODES = ['GENERAL', 'STUDY', 'DRIVING'];
        let currentModeIndex = 0;

        async function switchMode(mode) {
            currentModeIndex = MODES.indexOf(mode);
            document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
            const activeBtn = document.getElementById('btn-mode-' + mode.toLowerCase());
            if (activeBtn) activeBtn.classList.add('active');
            await fetch('/api/set_mode?mode=' + encodeURIComponent(mode), { method: 'POST' });
        }

        // Global Keyboard Shortcuts for Instant Mode Switching
        window.addEventListener('keydown', (e) => {
            // Ignore if user is typing in an input field
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            if (e.key === 'Tab' || e.key.toLowerCase() === 't') {
                e.preventDefault();
                currentModeIndex = (currentModeIndex + 1) % MODES.length;
                switchMode(MODES[currentModeIndex]);
            } else if (e.key === '1') {
                switchMode('GENERAL');
            } else if (e.key === '2') {
                switchMode('STUDY');
            } else if (e.key === '3') {
                switchMode('DRIVING');
            } else if (e.key.toLowerCase() === 'd') {
                toggleMesh();
            } else if (e.key.toLowerCase() === 'r') {
                resetStats();
            }
        });

        async function switchSource(src) {
            await fetch('/api/switch_source?source=' + encodeURIComponent(src), { method: 'POST' });
        }

        async function toggleMesh() {
            await fetch('/api/toggle_mesh', { method: 'POST' });
        }

        async function resetStats() {
            await fetch('/api/reset', { method: 'POST' });
            document.getElementById('incident-container').innerHTML = '<div class="empty-placeholder">Telemetry reset.</div>';
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("=" * 65)
    print("  DISTRACTION DETECTION SYSTEM (DDS) - FLUID WEB DASHBOARD")
    print("  Serving at: http://127.0.0.1:8000")
    print("=" * 65)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

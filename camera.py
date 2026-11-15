"""
Threaded, High-Performance Camera Capture Module.
Runs frame grabbing in an independent thread to prevent I/O blocking,
ensuring 0-latency streaming and smooth FPS across all interfaces.
"""

import os
import time
import threading
import cv2
import numpy as np

class ThreadedCamera:
    """
    Asynchronous camera grabber that continuously fetches frames from webcam or video file.
    Decouples video I/O from AI inference for stutter-free real-time performance.
    """

    def __init__(self, source=0, loop_video=True):
        self.source = source
        self.loop_video = loop_video
        self.cap = None
        self.current_frame = None
        self.is_running = False
        self.is_opened = False
        self.lock = threading.Lock()
        self.thread = None
        self.fps = 30.0
        self.width = 640
        self.height = 480
        self.open_source(source)

    def open_source(self, source):
        """Initializes or switches video capture source."""
        self.stop()
        self.source = source

        # 1. Physical Camera Index
        if str(source).isdigit():
            src_idx = int(source)
            # Try DirectShow on Windows first for fast startup
            self.cap = cv2.VideoCapture(src_idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(src_idx)
        else:
            # 2. Video File or Stream URL
            self.cap = cv2.VideoCapture(str(source))

        # 3. Fallback to demo video if camera cannot be opened
        if not self.cap.isOpened() and os.path.exists("sample_driver_demo.mp4"):
            print(f"[CAMERA] Source '{source}' unavailable. Falling back to 'sample_driver_demo.mp4'...")
            self.source = "sample_driver_demo.mp4"
            self.cap = cv2.VideoCapture("sample_driver_demo.mp4")

        if self.cap.isOpened():
            self.is_opened = True
            # Read first frame to initialize resolution
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.current_frame = frame
                self.height, self.width = frame.shape[:2]
            else:
                self.current_frame = self._create_placeholder("Camera connected, awaiting feed...")
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.thread.start()
        else:
            self.is_opened = False
            self.current_frame = self._create_placeholder(f"Source '{source}' could not be opened.")

    def _capture_worker(self):
        """Worker thread continuously buffering the freshest frame."""
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # If reading video file and looping enabled, rewind
                if self.loop_video and isinstance(self.source, str) and os.path.exists(self.source):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                else:
                    time.sleep(0.02)
                    continue

            with self.lock:
                self.current_frame = frame

            time.sleep(0.005) # Prevent CPU spinning

    def read(self):
        """Returns the latest captured frame instantly without I/O wait."""
        with self.lock:
            if self.current_frame is not None:
                return True, self.current_frame.copy()
            return False, self._create_placeholder("No frame available")

    def _create_placeholder(self, message):
        """Generates a stylish dark placeholder canvas if video feed is unavailable."""
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        canvas[:] = (20, 24, 32)
        cv2.putText(canvas, "DISTRACTION DETECTION SYSTEM", (80, 220),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 210, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, message, (80, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 175, 195), 1, cv2.LINE_AA)
        return canvas

    def stop(self):
        """Stops the worker thread and releases video capture hardware."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_opened = False

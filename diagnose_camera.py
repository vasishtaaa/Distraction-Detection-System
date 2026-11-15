"""
Camera Diagnostic & Backend Discovery Utility
Scans all available camera indices and backends (DirectShow, MSMF, Default)
to detect connected webcams on Windows.
"""

import cv2

def scan_cameras(max_indices=5):
    print("=" * 60)
    print(" SCANNING FOR CONNECTED WEBCAMS & CAPTURE BACKENDS")
    print("=" * 60)

    backends = [
        ("cv2.CAP_DSHOW (DirectShow - Windows standard)", cv2.CAP_DSHOW),
        ("cv2.CAP_MSMF (Media Foundation - Windows 10/11)", cv2.CAP_MSMF),
        ("cv2.CAP_ANY (Default Auto-Detect)", cv2.CAP_ANY)
    ]

    found_cameras = []

    for idx in range(max_indices):
        for name, backend in backends:
            print(f"Testing Camera Index [{idx}] using {name} ...", end=" ")
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w, _ = frame.shape
                        print(f"SUCCESS! (Resolution: {w}x{h})")
                        found_cameras.append((idx, backend, name, (w, h)))
                    else:
                        print("Opened but failed to grab frame.")
                    cap.release()
                else:
                    print("Could not open.")
            except Exception as e:
                print(f"Error: {e}")

    print("-" * 60)
    if found_cameras:
        print(f"Found {len(found_cameras)} accessible camera configuration(s):")
        for idx, backend, name, res in found_cameras:
            print(f"  -> Index: {idx} | Backend: {name} | Resolution: {res[0]}x{res[1]}")
    else:
        print("NO ACCESSIBLE CAMERAS FOUND across scanned indices.")
        print("\nCommon Causes & Quick Fixes:")
        print("1. Windows Privacy Settings: Settings -> Privacy & Security -> Camera")
        print("   -> Enable 'Allow apps to access your camera' AND 'Let desktop apps access your camera'.")
        print("2. Camera is in use by another application (Zoom, Teams, Discord, Browser, or Windows Camera app).")
        print("3. Hardware switch or physical shutter key (Fn key or sliding cover on laptop).")
    print("=" * 60)

    return found_cameras

if __name__ == "__main__":
    scan_cameras()

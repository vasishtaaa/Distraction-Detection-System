"""
Synthetic Driver Scenario Generator.
Creates a high-definition simulation video (`sample_driver_demo.mp4`)
simulating different driver states:
1. Normal Alert Driving (Road focused)
2. Eye Closure / Microsleep Fatigue Episode (EAR < 0.21)
3. Yawning Fatigue Episode (MAR > 0.65)
4. Off-Road Head Distraction (Looking right/down)
5. Hand-Held Cell Phone Usage (YOLOv8 target)
6. Recovery to Attentive Road Focus
"""

import os
import cv2
import numpy as np

def generate_demo_video(output_path="sample_driver_demo.mp4", duration_sec=14, fps=30):
    base_img = cv2.imread("test_face.jpg")
    if base_img is None:
        # Fallback to neutral canvas if test_face.jpg is not present
        base_img = np.full((512, 512, 3), 120, dtype=np.uint8)
        cv2.circle(base_img, (256, 256), 120, (200, 180, 160), -1)

    target_w, target_h = 960, 540
    # Center face in 960x540 frame
    face_resized = cv2.resize(base_img, (360, 360))
    fh, fw, _ = face_resized.shape

    total_frames = int(duration_sec * fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))

    print(f"[GENERATING] Creating {duration_sec}s ({total_frames} frames) synthetic test video -> {output_path}...")

    # Load cell phone image or draw high-contrast phone icon/sprite
    phone_sprite = np.zeros((130, 70, 3), dtype=np.uint8)
    cv2.rectangle(phone_sprite, (2, 2), (68, 128), (20, 20, 20), -1)
    cv2.rectangle(phone_sprite, (6, 12), (64, 114), (220, 180, 60), -1)  # illuminated screen
    cv2.circle(phone_sprite, (35, 122), 3, (120, 120, 120), -1)           # home button
    cv2.circle(phone_sprite, (35, 7), 2, (80, 80, 80), -1)               # speaker notch

    for f_idx in range(total_frames):
        # Create vehicle interior background
        frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        # Gradient dark cabin background
        for r in range(target_h):
            shade = int(35 + (r / target_h) * 25)
            frame[r, :] = (shade, shade - 5, shade - 10)

        # Steering wheel silhouette
        cv2.ellipse(frame, (target_w // 2, target_h + 100), (320, 220), 0, 180, 360, (25, 25, 30), 38)
        cv2.ellipse(frame, (target_w // 2, target_h + 100), (320, 220), 0, 180, 360, (50, 50, 60), 4)

        # Default Face placement
        face_x = (target_w - fw) // 2
        face_y = 60

        # Scenario Sequencing:
        # 0 - 75: Normal Attentive Driving
        # 75 - 155: Microsleep / Drowsiness (Eyes closing)
        # 155 - 235: Yawning Fatigue
        # 235 - 315: Off-Road Distraction (Looking right/down)
        # 315 - 390: Cell Phone Distraction
        # 390 - 420: Attentive Recovery

        scenario_name = "Normal Attentive Driving"
        face_copy = face_resized.copy()

        if 0 <= f_idx < 75:
            scenario_name = "Phase 1: Attentive Driving (Road Focused)"
            # Normal micro-blinks every 35 frames
            if f_idx % 35 in [0, 1, 2]:
                # Draw closed eyelids
                cv2.ellipse(face_copy, (140, 175), (22, 6), 0, 0, 360, (140, 120, 110), -1)
                cv2.ellipse(face_copy, (220, 175), (22, 6), 0, 0, 360, (140, 120, 110), -1)

        elif 75 <= f_idx < 155:
            scenario_name = "Phase 2: Microsleep & Eye Closure (Drowsiness Alert)"
            # Persistent closed eyes
            cv2.ellipse(face_copy, (140, 175), (24, 8), 0, 0, 360, (130, 115, 105), -1)
            cv2.ellipse(face_copy, (220, 175), (24, 8), 0, 0, 360, (130, 115, 105), -1)
            cv2.line(face_copy, (116, 175), (164, 175), (60, 50, 45), 2)
            cv2.line(face_copy, (196, 175), (244, 175), (60, 50, 45), 2)

        elif 155 <= f_idx < 235:
            scenario_name = "Phase 3: Yawning & Fatigue (MAR Threshold Exceeded)"
            # Draw wide open mouth
            cv2.ellipse(face_copy, (180, 275), (28, 22), 0, 0, 360, (40, 25, 30), -1)
            cv2.ellipse(face_copy, (180, 275), (28, 22), 0, 0, 360, (110, 80, 90), 2)

        elif 235 <= f_idx < 315:
            scenario_name = "Phase 4: Head Turned Off-Road (Yaw Distraction)"
            # Shift face position and slant to simulate turning right
            face_x += int(40 * np.sin((f_idx - 235) / 25))
            face_y += int(15 * np.sin((f_idx - 235) / 25))
            M = cv2.getRotationMatrix2D((fw / 2, fh / 2), 18, 1.0)
            face_copy = cv2.warpAffine(face_copy, M, (fw, fh))

        elif 315 <= f_idx < 390:
            scenario_name = "Phase 5: External Cell Phone Usage (YOLOv8 Detection)"
            # Overlay phone near driver hand/chest area
            px = face_x + fw - 30
            py = face_y + fh - 100
            ph, pw, _ = phone_sprite.shape
            frame[py:py+ph, px:px+pw] = phone_sprite
            # Hand gripping phone
            cv2.circle(frame, (px + 10, py + 65), 18, (160, 130, 120), -1)
            cv2.circle(frame, (px + pw - 10, py + 65), 16, (160, 130, 120), -1)

        else:
            scenario_name = "Phase 6: Attentive Recovery"

        # Place face into cabin frame
        frame[face_y:face_y+fh, face_x:face_x+fw] = face_copy

        # Top simulation watermark text
        cv2.putText(frame, f"SCENARIO TEST: {scenario_name}", (30, target_h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    print(f"[SUCCESS] Synthetic scenario demo saved to {output_path} ({os.path.getsize(output_path)} bytes)")

if __name__ == "__main__":
    generate_demo_video()

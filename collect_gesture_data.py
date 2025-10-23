"""
Author: Abhishek Kumar Singh / consolecoded
Initials: AKS
Date: 23-10-25
Project: GestureAI
"""
import cv2
import mediapipe as mp
import numpy as np
import os
from initials import signature

# ==== CONFIG ====
GESTURE_NAME = input("Enter gesture name: ").strip()  # e.g., open, fist, thumbs_up
SAMPLES = int(input("How many samples to collect? "))  # e.g., 200
SAVE_DIR = "gesture_data"
# =================

# Make directory
path = os.path.join(SAVE_DIR, GESTURE_NAME)
os.makedirs(path, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,                 # <--- detect both hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
count = 0

print(f"🎥 Collecting data for gesture '{GESTURE_NAME}' (both hands)... Press ESC to quit.")

while count < SAMPLES:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks as (x, y, z) coordinates
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            # Tag with left or right hand if available
            hand_label = result.multi_handedness[hand_idx].classification[0].label  # 'Left' or 'Right'
            filename = f"{GESTURE_NAME}_{hand_label}_{count}.npy"

            # Save each hand’s data separately
            np.save(os.path.join(path, filename), landmarks)
            count += 1

            cv2.putText(frame, f"Samples: {count}/{SAMPLES}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if count >= SAMPLES:
                break

    cv2.imshow("Collect Gesture Data (Both Hands)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Done! {count} samples saved in '{path}'.")

def signature():
    print('GestureAI by Abhishek Kumar Singh (Consolecoded)')

signature()

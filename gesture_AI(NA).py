"""
Author: Abhishek Kumar Singh / consolecoded
Initials: AKS
Date: 23-10-25
Project: GestureAI (Non-Arduino)
"""
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui
from collections import deque
from initials import signature

# ================= CONFIG =================
MODEL_FILE = "gesture_model.pkl"
DELAY = 1.5  # seconds between sending same gesture
SMOOTH_FRAMES = 5
CONF_THRESHOLD = 0.6
# ==========================================

# --- Load gesture recognition model ---
data = joblib.load(MODEL_FILE)
clf = data["model"]
label_map = {v: k for k, v in data["label_map"].items()}

expected_features = clf.n_features_in_  # auto-detect model feature count (42 or 63)
print(f"✅ Model expects {expected_features} features ({'2D' if expected_features == 42 else '3D'}) input")

# --- Gesture → Action mapping ---
gesture_actions = {
    "thumbsup": "LIGHT_ON",
    "thumbsdown": "LIGHT_OFF",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "THUMBS_UP": "FAN_UP",
    "THUMBS_DOWN": "FAN_DOWN",
    "PEACE": "TOGGLE_MUSIC",
    "WAVE": "SPECIAL_ACTION"
}

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# --- Helper functions ---
def perform_pc_action(action):
    """Perform PC action based on gesture"""
    if action == "TOGGLE_MUSIC":
        pyautogui.press("playpause")
    elif action == "FAN_UP":
        pyautogui.press("volumeup")
    elif action == "FAN_DOWN":
        pyautogui.press("volumedown")
    elif action == "SPECIAL_ACTION":
        pyautogui.hotkey("ctrl", "alt", "delete")
        print("⚡ Special PC action triggered!")
    else:
        print(f"Gesture recognized: {action}")  # fallback for gestures not mapped to PC keys

def handle_gesture(gesture):
    action = gesture_actions.get(gesture)
    if not action:
        return
    perform_pc_action(action)

def normalize_landmarks(landmarks):
    """Normalize to [-1,1], handling both 2D (42) and 3D (63) input automatically."""
    landmarks = np.array(landmarks, dtype=float)

    if landmarks.size == 63:  # 3D: x, y, z
        x, y, z = landmarks[::3], landmarks[1::3], landmarks[2::3]
        x -= x[0]; y -= y[0]; z -= z[0]
        max_val = max(max(abs(x)), max(abs(y)), max(abs(z)))
        if max_val > 0:
            x /= max_val; y /= max_val; z /= max_val
        landmarks[::3], landmarks[1::3], landmarks[2::3] = x, y, z

    elif landmarks.size == 42:  # 2D: x, y
        x, y = landmarks[::2], landmarks[1::2]
        x -= x[0]; y -= y[0]
        max_val = max(max(abs(x)), max(abs(y)))
        if max_val > 0:
            x /= max_val; y /= max_val
        landmarks[::2], landmarks[1::2] = x, y

    return landmarks.reshape(1, -1)

# --- Video capture ---
cap = cv2.VideoCapture(0)
last_gesture = None
last_time = 0
pred_queue = deque(maxlen=SMOOTH_FRAMES)

print("🎥 Starting AI gesture recognition (PC-only). Press ESC to exit.")

# --- Main loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    gesture_text = "UNKNOWN"

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if expected_features == 63:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        else:
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()

        landmarks = normalize_landmarks(landmarks)

        if landmarks.shape[1] != expected_features:
            continue

        probs = clf.predict_proba(landmarks)[0]
        pred_index = np.argmax(probs)
        confidence = probs[pred_index]

        if confidence >= CONF_THRESHOLD:
            pred_queue.append(pred_index)
        else:
            pred_queue.append(-1)

        valid_preds = [p for p in pred_queue if p != -1]
        if valid_preds:
            smoothed_pred = max(set(valid_preds), key=valid_preds.count)
            gesture_text = label_map[smoothed_pred]

            current_time = time.time()
            if gesture_text != last_gesture or (current_time - last_time > DELAY):
                handle_gesture(gesture_text)
                last_gesture = gesture_text
                last_time = current_time

    cv2.putText(frame, gesture_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow("AI Gesture Recognition (PC-only)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Exiting")

signature()

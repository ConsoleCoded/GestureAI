import os
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import datetime
import json

# ==== CONFIG ====
DATA_DIR = "gesture_data"
MODEL_FILE = "gesture_model.pkl"
RANDOM_STATE = 42
AUGMENTATIONS_PER_SAMPLE = 5
# =================

# --- Helper functions ---
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks, dtype=float)
    if len(landmarks) == 42:  # 2D
        landmarks = landmarks.reshape(21, 2)
        z = np.zeros((21, 1))
        landmarks = np.hstack([landmarks, z]).flatten()
    elif len(landmarks) != 63:
        raise ValueError(f"Invalid landmarks length {len(landmarks)}")

    x, y, z = landmarks[::3], landmarks[1::3], landmarks[2::3]
    x -= x[0]; y -= y[0]; z -= z[0]
    max_val = max(max(abs(x)), max(abs(y)), max(abs(z)))
    if max_val > 0:
        x /= max_val; y /= max_val; z /= max_val
    landmarks[::3], landmarks[1::3], landmarks[2::3] = x, y, z
    return landmarks

def augment_landmarks(landmarks):
    landmarks = landmarks.copy()
    # Random rotation
    angle = np.deg2rad(np.random.uniform(-10, 10))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    for i in range(21):
        x, y = landmarks[i*3], landmarks[i*3+1]
        landmarks[i*3] = x * cos_a - y * sin_a
        landmarks[i*3+1] = x * sin_a + y * cos_a
    # Shift
    shift = np.random.uniform(-0.05, 0.05, 3)
    landmarks[::3] += shift[0]
    landmarks[1::3] += shift[1]
    landmarks[2::3] += shift[2]
    # Scale
    factor = np.random.uniform(0.9, 1.1)
    landmarks *= factor
    # Noise
    landmarks += np.random.normal(0, 0.005, landmarks.shape)
    # Mirror version
    mirrored = landmarks.copy()
    mirrored[::3] *= -1
    return landmarks, mirrored

# --- Load all gesture data ---
def load_gesture_data():
    X, y, label_map = [], [], {}
    label_counter = 0

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"No directory found: {DATA_DIR}")

    for gesture_name in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, gesture_name)
        if not os.path.isdir(path):
            continue
        label_map[gesture_name] = label_counter
        files = [f for f in os.listdir(path) if f.endswith(".npy")]

        for f in tqdm(files, desc=f"Processing {gesture_name}", leave=False):
            data = np.load(os.path.join(path, f))
            data = normalize_landmarks(data)
            X.append(data)
            y.append(label_counter)

            # Augment
            for _ in range(AUGMENTATIONS_PER_SAMPLE):
                aug, mir = augment_landmarks(data)
                X.append(aug); y.append(label_counter)
                X.append(mir); y.append(label_counter)

        label_counter += 1

    return np.array(X), np.array(y), label_map


# --- Train model ---
print("📦 Loading and training gesture data...")
X, y, label_map = load_gesture_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {acc * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=list(label_map.keys())))
print("\n🌀 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Save model ---
joblib.dump({"model": clf, "label_map": label_map}, MODEL_FILE)
print(f"\n💾 Model saved as: {MODEL_FILE}")

# --- Save metadata ---
meta = {
    "timestamp": datetime.datetime.now().isoformat(),
    "num_classes": len(label_map),
    "classes": list(label_map.keys()),
    "accuracy": acc
}
with open("gesture_meta.json", "w") as f:
    json.dump(meta, f, indent=4)

print("📝 Metadata saved: gesture_meta.json")
print("\n✅ Training complete!")

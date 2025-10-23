# GestureAI

GestureAI is a high-accuracy static hand gesture recognition AI trained on **2 million+ samples**. It supports multiple versions for different use cases: full Arduino + PC integration, PC-only, and a lightweight demo version.

---

## Features by Version

| Version | File | Features |
|---------|------|---------|
| **Full (Arduino + PC)** | `gesture_ai.py` | ✅ Recognizes static gestures in real-time <br> ✅ Sends commands to Arduino devices <br> ✅ Performs PC actions (music, volume, special commands) <br> ✅ Auto-detects 2D/3D gestures <br> ⚠️ Requires Arduino COM port setup |
| **PC-only** | `gesture_ai_web.py` | ✅ Recognizes gestures in real-time <br> ✅ Prints recognized gestures to console <br> ✅ Auto-detects 2D/3D gestures <br> ❌ No Arduino or PC automation |
| **Gesture Collection** | `collect_gesture_data.py` | ✅ Collects static gestures via webcam <br> ✅ Saves gestures per hand in `.npy` files <br> ✅ Supports multiple samples per gesture |
| **Training New Model** | `train_gesture_model.py` | ✅ Trains a RandomForest model on collected gestures <br> ✅ Applies data augmentation (rotation, scaling, mirroring, noise) <br> ✅ Outputs model (`gesture_model.pkl`) and metadata (`gesture_meta.json`) <br> ✅ Prints accuracy, classification report, and confusion matrix |

---

## Setup Instructions

1. Download `gesture_model.zip` from the repository.  
2. Extract to get `gesture_model.pkl`.  
3. Install dependencies:  `pip install -r requirement.txt`
4. Full Version (Arduino + PC): `python gesture_ai.py`
5. Note: Set your Arduino COM port in the script: `ARDUINO_PORT = "COM14"  # Replace with your port`
6. PC-only Version: `python gesture_ai_web.py`
7. PC-only Version but PC gestures: `gesture_AI(NA).py`
8. Gesture Collection: `python collect_gesture_data.py`
9. Training Model: `python train_gesture_model.py`
Notes:
Static gestures only; dynamic gestures may reduce accuracy.

All Python scripts import initials.py for project signature.

PC-only and demo versions do not require hardwar


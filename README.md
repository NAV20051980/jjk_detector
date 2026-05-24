# JJK Detector — Jujutsu Kaisen Hand Sign Detector

A real-time hand gesture recognition system that detects iconic hand signs from **Jujutsu Kaisen** using your webcam, MediaPipe, and a trained Random Forest Classifier.

## Gestures Supported
| Gesture | Description | Color |
|---|---|---|
| `infinity` | Gojo's two-finger V sign | Cyan |
| `domain_expansion` | Both palms open, fingers spread wide | Purple |
| `cleave` | Only index finger pointing | Red |
| `black_flash` | Tight closed fist | Orange |
| `idle` | Relaxed open palm | Green |

## Project Structure
```
jjk_detector/
├── data/               # Collected landmark CSVs
├── model/              # Trained model and encoder
├── collect_data.py     # Webcam-based data collection
├── train_model.py      # Model training script
├── app.py              # Real-time gesture detection app
└── requirements.txt    # Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Collect Training Data
```bash
python collect_data.py
```
Press keys `1`–`5` to collect 200 samples per gesture. Press `Q` to quit.

### 2. Train the Model
```bash
python train_model.py
```

### 3. Run the App
```bash
python app.py
```
Press `Q` to quit.

## How It Works
- **MediaPipe Tasks HandLandmarker** detects 21 hand landmarks (x, y, z) per frame.
- Landmarks are **normalized relative to the wrist** for position invariance.
- A **RandomForestClassifier (200 trees)** predicts the gesture from 63 features.
- Predictions are stabilized via a **10-frame majority vote buffer** (7/10 threshold).
- Gesture-specific confidence thresholds: **90%** for `infinity`/`domain_expansion`, **85%** for others.
- A **0.8-second display cooldown** prevents rapid flickering.

## Model Performance
- Train Accuracy: **100%**
- Test Accuracy: **100%**
- 3,000 total samples (600 per gesture)

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import csv
import os
import ssl
import urllib.request

# Define paths
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
MODEL_PATH = "hand_landmarker.task"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Define gesture mapping
GESTURES = {
    '1': 'infinity',
    '2': 'domain_expansion',
    '3': 'cleave',
    '4': 'black_flash',
    '5': 'idle'
}

TOTAL_SAMPLES = 200

# 1. Download hand landmarker model if not present (bypassing SSL verification if needed)
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    try:
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=context) as response, open(MODEL_PATH, 'wb') as out_file:
            out_file.write(response.read())
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        exit(1)

# 2. Initialize Hand Landmarker using the Tasks API
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# State variables
is_collecting = False
current_gesture = None
samples_collected = 0

print("JJK Hand Gesture Data Collector Initialized!")
print("===========================================")
print("Key Bindings:")
for key, gesture in GESTURES.items():
    print(f"  Press '{key}' to collect {TOTAL_SAMPLES} samples for: {gesture}")
print("  Press 'q' to quit.")
print("===========================================")

# Write CSV header if the file is new or empty
if not os.path.exists(CSV_PATH) or os.stat(CSV_PATH).st_size == 0:
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"lm_{i}_{coord}" for i in range(21) for coord in ('x', 'y', 'z')] + ['label']
        writer.writerow(header)

# Hand skeleton connections (standard MediaPipe Hands connections)
CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckles
    (5, 9), (9, 13), (13, 17)
]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for natural selfie view
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    # Convert OpenCV image to MediaPipe Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect landmarks
    detector_result = detector.detect(mp_image)
    
    hand_detected = False
    normalized_landmarks = []

    if detector_result.hand_landmarks:
        hand_detected = True
        hand_landmarks = detector_result.hand_landmarks[0]
        
        # 1. Convert landmarks to pixel coordinates for custom drawing
        pixel_coords = []
        for lm in hand_landmarks:
            cx = int(lm.x * width)
            cy = int(lm.y * height)
            pixel_coords.append((cx, cy))
            
        # 2. Draw connections (skeleton lines)
        for start_idx, end_idx in CONNECTIONS:
            if start_idx < len(pixel_coords) and end_idx < len(pixel_coords):
                cv2.line(frame, pixel_coords[start_idx], pixel_coords[end_idx], (0, 255, 0), 2)
                
        # 3. Draw keypoints (circles)
        for coord in pixel_coords:
            cv2.circle(frame, coord, 5, (0, 0, 255), -1)

        # 4. Normalize landmarks relative to wrist (index 0)
        wrist = hand_landmarks[0]
        wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z
        
        for lm in hand_landmarks:
            normalized_landmarks.extend([
                lm.x - wrist_x,
                lm.y - wrist_y,
                lm.z - wrist_z
            ])
            
        # Save sample if collecting
        if is_collecting and samples_collected < TOTAL_SAMPLES:
            with open(CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(normalized_landmarks + [current_gesture])
            samples_collected += 1
            
            if samples_collected >= TOTAL_SAMPLES:
                is_collecting = False
                print(f"Finished collecting {TOTAL_SAMPLES} samples for '{current_gesture}'!")

    # Draw Status and GUI Overlay
    status_text = "Status: Ready"
    status_color = (0, 255, 0)
    
    if is_collecting:
        status_text = f"Collecting: {current_gesture.upper()}"
        status_color = (0, 0, 255)
        
        # Progress bar
        progress_ratio = samples_collected / TOTAL_SAMPLES
        bar_width = int(progress_ratio * 400)
        cv2.rectangle(frame, (100, 80), (500, 100), (100, 100, 100), -1)
        cv2.rectangle(frame, (100, 80), (100 + bar_width, 100), (0, 255, 0), -1)
        cv2.putText(frame, f"{samples_collected}/{TOTAL_SAMPLES}", (510, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if not hand_detected:
            cv2.putText(frame, "HAND NOT DETECTED!", (150, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    
    # Legend
    cv2.putText(frame, "1: infinity | 2: domain_expansion | 3: cleave | 4: black_flash | 5: idle", 
                (20, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'Q' to Quit", 
                (20, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Show window
    cv2.imshow('JJK Gesture Data Collector', frame)
    
    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if chr(key) in GESTURES and not is_collecting:
        current_gesture = GESTURES[chr(key)]
        samples_collected = 0
        is_collecting = True
        print(f"Starting collection for '{current_gesture}'...")
    elif key == ord('q') or key == ord('Q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
detector.close()
print("Data collection session closed.")

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import os
import time


# Define model paths
MODEL_PATH = "model/jjk_classifier.pkl"
ENCODER_PATH = "model/label_encoder.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"

# Colors for gestures (in BGR format)
COLORS = {
    'infinity': (255, 255, 0),          # Cyan
    'domain_expansion': (240, 32, 160),  # Purple
    'cleave': (0, 0, 255),               # Red
    'black_flash': (0, 140, 255),        # Orange
    'idle': (0, 255, 0)                 # Green
}

NEUTRAL_COLOR = (200, 200, 200)         # Gray for low confidence/undetected

# Hand connections for drawing
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

def main():
    # 1. Load the trained classifier and label encoder
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print("Error: Trained model files not found. Please run train_model.py first.")
        return
        
    print("Loading model and encoder...")
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
        
    # 2. Check and initialize MediaPipe Hand Landmarker
    if not os.path.exists(HAND_MODEL_PATH):
        print(f"Error: {HAND_MODEL_PATH} not found. Please run collect_data.py first to download it.")
        return
        
    base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # 3. Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("JJK Real-time Gesture Detector Active!")
    print("Press 'Q' to quit.")

    # State variables for stabilizing predictions
    predictions_buffer = []
    displayed_gesture = None
    displayed_confidence = 0.0
    last_display_time = 0.0
    COOLDOWN_DURATION = 0.8  # seconds

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        current_time = time.time()

        # Flip horizontally for selfie-view
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        
        # Convert OpenCV frame to MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hand landmarks
        detector_result = detector.detect(mp_image)
        
        raw_prediction = None
        frame_confidence = 0.0
        pixel_coords = []

        if detector_result.hand_landmarks:
            hand_landmarks = detector_result.hand_landmarks[0]
            
            # Map landmarks to pixel coordinates
            for lm in hand_landmarks:
                cx = int(lm.x * width)
                cy = int(lm.y * height)
                pixel_coords.append((cx, cy))
                
            # Normalize landmarks relative to wrist
            wrist = hand_landmarks[0]
            wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z
            
            normalized_features = []
            for lm in hand_landmarks:
                normalized_features.extend([
                    lm.x - wrist_x,
                    lm.y - wrist_y,
                    lm.z - wrist_z
                ])
                
            # Run prediction on the 63 normalized features
            probabilities = clf.predict_proba([normalized_features])[0]
            max_idx = np.argmax(probabilities)
            max_prob = probabilities[max_idx]
            temp_gesture = label_encoder.inverse_transform([max_idx])[0]
            
            # Determine threshold: 90% for infinity/domain_expansion, 85% for others
            threshold = 0.90 if temp_gesture in ('infinity', 'domain_expansion') else 0.85
            
            if max_prob >= threshold:
                raw_prediction = temp_gesture
                frame_confidence = max_prob

        # Add to buffer
        predictions_buffer.append(raw_prediction)
        if len(predictions_buffer) > 10:
            predictions_buffer.pop(0)

        # Determine majority voted gesture (needs at least 7/10 majority)
        voted_gesture = None
        if len(predictions_buffer) > 0:
            counts = {}
            for pred in predictions_buffer:
                if pred is not None:
                    counts[pred] = counts.get(pred, 0) + 1
            
            for pred, count in counts.items():
                if count >= 7:
                    voted_gesture = pred
                    break

        # Cooldown update logic
        if displayed_gesture is None:
            if voted_gesture is not None:
                displayed_gesture = voted_gesture
                displayed_confidence = frame_confidence
                last_display_time = current_time
        else:
            # A gesture is currently displayed. Check if cooldown has expired
            if current_time - last_display_time >= COOLDOWN_DURATION:
                if voted_gesture != displayed_gesture:
                    displayed_gesture = voted_gesture
                    if voted_gesture is not None:
                        displayed_confidence = frame_confidence
                        last_display_time = current_time
                    else:
                        displayed_confidence = 0.0
                        last_display_time = 0.0
            else:
                # Cooldown active. Keep the displayed gesture.
                # If current frame matches the displayed gesture, update its confidence
                if raw_prediction == displayed_gesture:
                    displayed_confidence = frame_confidence

        # Determine draw color based on displayed gesture
        draw_color = COLORS.get(displayed_gesture, NEUTRAL_COLOR) if displayed_gesture else NEUTRAL_COLOR

        # Draw hand landmarks and connections if hand is detected
        if detector_result.hand_landmarks:
            # Draw skeleton connections
            for start_idx, end_idx in CONNECTIONS:
                if start_idx < len(pixel_coords) and end_idx < len(pixel_coords):
                    cv2.line(frame, pixel_coords[start_idx], pixel_coords[end_idx], draw_color, 2)
                    
            # Draw landmark keypoints
            for coord in pixel_coords:
                cv2.circle(frame, coord, 5, (0, 0, 255), -1)

        # 4. Display text overlay
        if displayed_gesture:
            # Large bold gesture name
            cv2.putText(frame, displayed_gesture.upper(), (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, draw_color, 3, cv2.LINE_AA)
            # Confidence percentage
            cv2.putText(frame, f"Confidence: {displayed_confidence * 100:.1f}%", (20, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Special overlay for DOMAIN EXPANSION
            if displayed_gesture == 'domain_expansion':
                text = "DOMAIN EXPANSION"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.8
                thickness = 4
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (width - text_size[0]) // 2
                text_y = (height + text_size[1]) // 2
                
                # Black backdrop box with purple border
                box_padding = 20
                cv2.rectangle(frame, 
                              (text_x - box_padding, text_y - text_size[1] - box_padding), 
                              (text_x + text_size[0] + box_padding, text_y + box_padding), 
                              (0, 0, 0), -1)
                cv2.rectangle(frame, 
                              (text_x - box_padding, text_y - text_size[1] - box_padding), 
                              (text_x + text_size[0] + box_padding, text_y + box_padding), 
                              draw_color, 3)
                # Centered text in purple/white
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        else:
            cv2.putText(frame, "SEARCHING...", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, NEUTRAL_COLOR, 2, cv2.LINE_AA)

        # Instructions
        cv2.putText(frame, "Press 'Q' to Quit", (20, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        # Show stream window
        cv2.imshow('JJK Gesture Detector', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Application closed successfully.")

if __name__ == "__main__":
    main()

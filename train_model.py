import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define paths
DATA_PATH = "data/landmarks.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "jjk_classifier.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

def main():
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Read data/landmarks.csv
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Please collect data first.")
        return
        
    print(f"Reading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 2. Print sample count per gesture
    print("\nSamples per gesture found:")
    gesture_counts = df['label'].value_counts()
    for gesture, count in gesture_counts.items():
        print(f"  {gesture}: {count} samples")
        
    # 3. Split into features and labels
    # The last column is 'label', everything else is features (63 columns)
    X = df.drop(columns=['label']).values
    y = df['label'].values
    
    # 4. Encode the labels using LabelEncoder
    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 5. Split into 80% train / 20% test (stratified for balanced splitting)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 6. Train a RandomForestClassifier with 200 trees
    print("Training RandomForestClassifier (200 estimators)...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    
    # 7. Print Train and Test accuracies
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTrain Accuracy: {train_acc:.4f} ({train_acc * 100:.2f}%)")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    
    # 8. Print full classification report
    print("\nClassification Report (Test Set):")
    report = classification_report(
        y_test, 
        y_test_pred, 
        target_names=label_encoder.classes_
    )
    print(report)
    
    # 9. Save the trained model and label encoder
    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
        
    print(f"Saving label encoder to {ENCODER_PATH}...")
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
        
    print("\nModel training workflow completed successfully!")

if __name__ == "__main__":
    main()

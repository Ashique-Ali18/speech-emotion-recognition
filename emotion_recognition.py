import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# ✅ Set correct folder path here
DATA_DIR = r"C:\Users\ashiq\CodeAlpha_CreditScoringModel\CodeAlpha_CreditScoringModel\emotion_data"

# Emotion label mapping
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

features = []
labels = []

print("🔄 Extracting MFCC features from audio files...")

for actor_folder in os.listdir(DATA_DIR):
    actor_path = os.path.join(DATA_DIR, actor_folder)
    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                try:
                    emotion_code = file.split("-")[2]
                    emotion = EMOTIONS.get(emotion_code)
                    if emotion:
                        file_path = os.path.join(actor_path, file)
                        mfccs = extract_features(file_path)
                        features.append(mfccs)
                        labels.append(emotion)
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")

print("✅ Feature extraction completed.")

X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))
print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=list(EMOTIONS.values()))
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(np.arange(len(EMOTIONS)), list(EMOTIONS.values()), rotation=45)
plt.yticks(np.arange(len(EMOTIONS)), list(EMOTIONS.values()))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("emotion_confusion_matrix.png")
print("📁 Confusion matrix saved as emotion_confusion_matrix.png")

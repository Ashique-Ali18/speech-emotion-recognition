import os
import librosa
import numpy as np

DATA_PATH = r"C:\Users\ashiq\CodeAlpha_CreditScoringModel\CodeAlpha_CreditScoringModel\emotion_data"
X = []
y = []

# Emotion labels based on RAVDESS dataset
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

print("🔍 Extracting features from audio files...")
for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            try:
                data, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
                mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
                mfccs_scaled = np.mean(mfccs.T, axis=0)

                emotion_code = file.split("-")[2]
                label = emotion_map.get(emotion_code)

                if label:
                    X.append(mfccs_scaled)
                    y.append(label)

            except Exception as e:
                print(f"❌ Failed on {file}: {e}")

X = np.array(X)
y = np.array(y)

np.save("features.npy", X)
np.save("labels.npy", y)

print("✅ Feature extraction completed. Saved as features.npy and labels.npy")

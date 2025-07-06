import librosa
import numpy as np
import joblib
import sys

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfcc
    except Exception as e:
        print("❌ Error extracting features:", e)
        return None

def predict_emotion(file_path):
    model = joblib.load("emotion_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    features = extract_features(file_path)
    if features is not None:
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        emotion = label_encoder.inverse_transform(prediction)
        print(f"🎤 Detected Emotion: {emotion[0]}")
    else:
        print("❌ Couldn't predict emotion due to feature extraction error.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Usage: python predict_emotion.py <path_to_audio.wav>")
    else:
        file_path = sys.argv[1]
        predict_emotion(file_path)

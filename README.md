# Speech Emotion Recognition

This project performs **emotion recognition from audio speech files** using machine learning. It uses MFCC features extracted from `.wav` files to classify emotions such as happy, sad, angry, fearful, etc.

# Dataset
- Source: [RAVDESS Emotional Speech Audio Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- Total Audio Files: 1440 `.wav` files
- Stored in: `/emotion_data/`

# Features Extracted
- **MFCCs (Mel-Frequency Cepstral Coefficients)**
- Feature arrays saved as `features.npy` and `labels.npy`

# Model
- Algorithm: **MLPClassifier (Multi-layer Perceptron)**
- Trained using: `train_model.py`
- Accuracy: 90%+

# Main Files
- `extract_features.py`: Extracts MFCC features
- `train_model.py`: Trains and saves the model (`emotion_model.pkl`)
- `predict_emotion.py`: Predicts emotion from new audio input
- `emotion_recognition.py`: Combined script for loading model and testing

# How to Run

```bash
# Extract features
python extract_features.py

# Train the model
python train_model.py

# Predict emotion
python predict_emotion.py

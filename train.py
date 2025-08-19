import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Parameters
DATA_DIR = "data"
SAMPLES_TO_CONSIDER = 22050  # 1 second at 22kHz

# Function to extract MFCC
def preprocess_audio(file_path, n_mfcc=13):
    signal, sr = librosa.load(file_path, sr=22050)
    if len(signal) > SAMPLES_TO_CONSIDER:
        signal = signal[:SAMPLES_TO_CONSIDER]
    mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # time x features

# Load dataset
X, y = [], []
labels = {"hey_ai": 1, "other": 0}

for label, idx in labels.items():
    folder = os.path.join(DATA_DIR, label)
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        mfcc = preprocess_audio(file_path)
        X.append(mfcc)
        y.append(idx)

# Pad sequences
X = tf.keras.preprocessing.sequence.pad_sequences(X, dtype="float32")
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build RNN model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/hey_ai_rnn.h5")


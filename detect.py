import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/hey_ai_rnn.h5")

# Parameters
SAMPLE_RATE = 22050
DURATION = 1  # 1 second
SAMPLES = SAMPLE_RATE * DURATION

def extract_mfcc(audio, sr=22050, n_mfcc=13):
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

print("Listening... Say 'Hey AI' 🚀")

while True:
    recording = sd.rec(SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    audio = np.squeeze(recording)

    mfcc = extract_mfcc(audio)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)

    prediction = model.predict(mfcc)[0][0]
    if prediction > 0.8:
        print("Wake word detected: 'Hey AI' 🎤")


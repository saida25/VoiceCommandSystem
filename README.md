# Voice Command System 🎤

A **wake-word detection system** ("Hey AI") using **MFCC features + RNN (LSTM)**.  

## 🚀 Features
- Detects custom wake word in real-time
- Uses MFCC audio features
- RNN (LSTM) for sequence classification

## 🛠️ Tech Stack
- Python
- TensorFlow/Keras
- Librosa (MFCC extraction)
- SoundDevice (real-time audio input)

## 📂 Project Structure

voice-command-system/
│── data/ # dataset
│ ├── hey_ai/ # "Hey AI" samples
│ ├── other/ # background/noise
│── models/ # trained model
│── train.py # training script
│── detect.py # real-time detection
│── requirements.txt # dependencies
│── README.md # docs


## 📊 Dataset
- Record samples of yourself saying **"Hey AI"** (~100+ samples).
- Collect background/noise/random speech (~200+ samples).
- Save in `data/hey_ai/` and `data/other/`.

## 📦 Installation
```bash
git clone https://github.com/your-username/voice-command-system.git
cd voice-command-system
pip install -r requirements.txt
🔧 Usage

Train the model:
python train.py
(model will be saved in models/hey_ai_rnn.h5)


Run detection:

python detect.py
Say "Hey AI" into your mic 🎤

🙌 Acknowledgments

Librosa:for audio processing
TensorFlow: for deep learning



# 📁 File: predict_once.py

import sounddevice as sd
import librosa
import torch
import numpy as np
import torch.nn.functional as F
from models.model import CNNNet

def record_audio(seconds=3, fs=16000):
    print("🎙️ Recording for {} seconds...".format(seconds))
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return recording.flatten()

def preprocess_audio(audio, sr=16000, n_mels=64, max_len=256):
    # Compute Mel spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Convert to tensor [1, 64, T]
    mel_tensor = torch.tensor(mel_db).unsqueeze(0).float()

    # Pad or truncate to fixed number of time frames
    time_dim = mel_tensor.shape[-1]
    if time_dim > max_len:
        mel_tensor = mel_tensor[:, :, :max_len]
    else:
        mel_tensor = F.pad(mel_tensor, (0, max_len - time_dim))

    # Normalize
    mel_tensor = (mel_tensor - mel_tensor.mean()) / (mel_tensor.std() + 1e-9)

    # Final shape: [1, 1, 64, 256]
    return mel_tensor.unsqueeze(0)

def predict(model, recording):
    input_tensor = preprocess_audio(recording)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    return "✅ Real Voice" if pred == 0 else "❌ Spoofed Voice"

# Load trained model
model = CNNNet()
model.load_state_dict(torch.load('models/best_model.pth', map_location='cpu'))
model.eval()

# Record and predict
audio = record_audio()
result = predict(model, audio)
print("🧠 Prediction:", result)

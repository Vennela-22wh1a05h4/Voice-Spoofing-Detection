import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from models.model import CNNNet
import librosa
import torch.nn.functional as F

# Constants
DURATION = 5
SAMPLE_RATE = 16000
THRESHOLD = 0.01
MAX_LEN = 256
n_mels = 64

# Record audio
print("🎙️ Listening...")
recording = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()

# Silence detection
volume = np.linalg.norm(recording)
print(f"Volume: {volume:.4f}")
if volume < THRESHOLD:
    print("😶 Too quiet. Exiting.")
    exit()

write("input.wav", SAMPLE_RATE, recording)

# Load audio
try:
    y, sr = librosa.load("input.wav", sr=SAMPLE_RATE)
except Exception as e:
    print("⚠️ Error loading audio:", e)
    exit()

# Mel-spectrogram
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=160, win_length=400, n_mels=n_mels)
log_mel = librosa.power_to_db(mel, ref=np.max)
mel_tensor = torch.tensor(log_mel).unsqueeze(0).float()

# Pad/truncate
if mel_tensor.shape[-1] > MAX_LEN:
    mel_tensor = mel_tensor[:, :, :MAX_LEN]
else:
    mel_tensor = F.pad(mel_tensor, (0, MAX_LEN - mel_tensor.shape[-1]))

mel_tensor = (mel_tensor - mel_tensor.mean()) / (mel_tensor.std() + 1e-9)
input_tensor = mel_tensor.unsqueeze(0)  # Shape: [1, 1, 64, 256]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNNet().to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

# Predict
with torch.no_grad():
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    predicted = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted].item()

label_map = {0: "✅ Real Voice", 1: "❌ Spoofed Voice"}
print(f"🧠 Prediction: {label_map[predicted]} (Confidence: {confidence:.2f})")

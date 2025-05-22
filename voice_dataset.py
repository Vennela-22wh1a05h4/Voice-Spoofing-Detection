import torch
import os
import librosa
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

class ASVspoofDataset(Dataset):
    def __init__(self, file_list, label_dict, root_path, sr=16000, n_mels=64, max_len=256, augment=True):
        self.file_list = file_list
        self.label_dict = label_dict
        self.root_path = root_path
        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len  # Number of time frames
        self.augment = augment

        # ✅ Match mel settings from predict_live.py
        self.n_fft = 512
        self.hop_length = 160
        self.win_length = 400

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        label = 0 if self.label_dict[filename] == 'bonafide' else 1
        filepath = os.path.join(self.root_path, filename + '.flac')

        # Load audio
        y, _ = librosa.load(filepath, sr=self.sr)

        # ===== 🔄 AUGMENTATION =====
        if self.augment:
            if random.random() < 0.5:
                noise = np.random.normal(0, 0.005, y.shape)
                y += noise
            if random.random() < 0.5:
                shift = int(0.1 * len(y))
                y = np.roll(y, shift)

        # ===== 🎵 MEL-SPECTROGRAM (MATCH TRAINING & LIVE PREDICTION) =====
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_tensor = torch.tensor(mel_db).unsqueeze(0).float()  # Shape: [1, n_mels, T]

        # ===== ⏳ Padding or truncating =====
        time_dim = mel_tensor.shape[-1]
        if time_dim > self.max_len:
            mel_tensor = mel_tensor[:, :, :self.max_len]
        else:
            mel_tensor = F.pad(mel_tensor, (0, self.max_len - time_dim))

        # ✅ Normalize like in prediction
        mel_tensor = (mel_tensor - mel_tensor.mean()) / (mel_tensor.std() + 1e-9)

        return mel_tensor, label

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from models.model import CNNNet
from voice_dataset import ASVspoofDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# === ✅ Paths ===
protocol_file = r'C:\Voice_Project\data\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.txt'
root_audio_path = r'C:\Voice_Project\data\LA\ASVspoof2019_LA_eval\flac'

# === ✅ Load file list and label mapping ===
file_list = []
label_dict = {}
with open(protocol_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        file_id, label = parts[1], parts[-1].lower()  # Normalize label to lowercase
        file_list.append(file_id)
        label_dict[file_id] = 0 if label == 'bonafide' else 1

# === ✅ Create Dataset and DataLoader ===
test_dataset = ASVspoofDataset(file_list, label_dict, root_audio_path, augment=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# === ✅ Load Trained Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNNet().to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

# === ✅ Evaluate ===
all_preds = []
all_labels = []
with torch.no_grad():
    for mel, label in tqdm(test_loader, desc="🔍 Testing"):
        mel, label = mel.to(device), label.to(device)
        output = model(mel)
        _, predicted = torch.max(output, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

# === ✅ Accuracy ===
correct = np.sum(np.array(all_preds) == np.array(all_labels))
total = len(all_labels)
accuracy = 100 * correct / total
print(f"\n✅ Test Accuracy: {accuracy:.2f}%")

# === ✅ Confusion Matrix and Classification Report ===
cm = confusion_matrix(all_labels, all_preds)
print("\n🧾 Confusion Matrix:")
print(cm)

print("\n📊 Classification Report (Precision / Recall / F1):")
print(classification_report(all_labels, all_preds, target_names=["Bonafide", "Spoof"]))

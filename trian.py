import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models.model import CNNNet
from voice_dataset import ASVspoofDataset
from tqdm import tqdm
import copy
import random
import numpy as np

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def main():
    # Paths
    train_protocol = r'C:\Voice_Project\data\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt'
    root_audio_path = r'C:\Voice_Project\data\LA\ASVspoof2019_LA_train\flac'
    val_protocol = r'C:\Voice_Project\data\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt'
    val_audio_path = r'C:\Voice_Project\data\LA\ASVspoof2019_LA_dev\flac'

    # Load train data
    file_list, label_dict = [], {}
    with open(train_protocol, 'r') as f:
        for line in f:
            parts = line.strip().split()
            file_id, label = parts[1], parts[-1]
            file_list.append(file_id)
            label_dict[file_id] = label

    # Load validation data
    val_file_list, val_label_dict = [], {}
    with open(val_protocol, 'r') as f:
        for line in f:
            parts = line.strip().split()
            file_id, label = parts[1], parts[-1]
            val_file_list.append(file_id)
            val_label_dict[file_id] = label

    # Datasets and Loaders
    train_loader = DataLoader(
        ASVspoofDataset(file_list, label_dict, root_audio_path),
        batch_size=16, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        ASVspoofDataset(val_file_list, val_label_dict, val_audio_path, augment=False),
        batch_size=16, shuffle=False, num_workers=2
    )

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Early stopping
    patience, best_loss, best_model, epochs_no_improve = 3, float('inf'), None, 0

    # Training loop
    for epoch in range(20):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for mel, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            mel, label = mel.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(mel)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        acc = 100 * correct / total
        print(f"🟢 Epoch {epoch+1} | Train Loss: {running_loss:.4f} | Accuracy: {acc:.2f}%")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mel, label in val_loader:
                mel, label = mel.to(device), label.to(device)
                output = model(mel)
                val_loss += criterion(output, label).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"🔵 Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print("✅ Validation improved. Saving model.")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print("⛔ Early stopping.")
            break

    if best_model:
        torch.save(best_model, "models/best_model.pth")
        print("💾 Best model saved to models/best_model.pth")

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()

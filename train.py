import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.models.ast_models import ASTModel

# =============== CONFIG ===============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETRAINED_MODEL_PATH = "C:/Users/C3I LAB_A09/Desktop/sedd/audioset_10_10_0.4593.pth"
ESC50_CSV_PATH = "C:/Users/C3I LAB_A09/Desktop/sedd/ESC-50-master/meta/esc50.csv"
ESC50_AUDIO_DIR = "C:/Users/C3I LAB_A09/Desktop/sedd/ESC-50-master/audio"
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
SAVE_MODEL_PATH = 'fine_tuned_ast_esc50.pth'

# =============== Dataset ===============
def get_log_mel_spectrogram(audio_path):
    try:
        waveform, sr = librosa.load(audio_path, sr=16000)
        spectrogram = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_mels=128, n_fft=1024, hop_length=160
        )
        log_spectrogram = librosa.power_to_db(spectrogram)
        return log_spectrogram
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

class ESC50Dataset(Dataset):
    def __init__(self, csv_path, audio_dir, mode='train'):
        self.meta = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.file_paths = []
        self.labels = []

        if mode == 'train':
            self.meta = self.meta[self.meta['fold'].isin([1, 2, 3, 4])]
        elif mode == 'test':
            self.meta = self.meta[self.meta['fold'].isin([5])]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for idx, row in self.meta.iterrows():
            file_path = os.path.join(audio_dir, row['filename'])
            label = row['target']
            self.file_paths.append(file_path)
            self.labels.append(label)

    def __getitem__(self, index):
        spec = get_log_mel_spectrogram(self.file_paths[index])  # (128, time)

        # Pad or crop to 1000 time steps
        if spec.shape[1] < 1000:
            pad_width = 1000 - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            spec = spec[:, :1000]

        spec = torch.tensor(spec).unsqueeze(0)  # (1, 128, 1000)

        # 🔥🔥 Fix: Now permute to (channel=1, frequency=128, time=1000)
        spec = spec.permute(0, 1, 2)  # Actually unnecessary, already correct after unsqueeze!

        # 🚨 CRITICAL: Now Squeeze any wrong dimension
        spec = spec.squeeze(0)  # Now (1, 128, 1000)

        return spec, self.labels[index]


    def __len__(self):
        return len(self.file_paths)

# =============== Load Model ===============
def load_ast_model(pretrained_path):
    model = ASTModel(
        label_dim=50,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=1000,
        model_size='base384'
    )
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    return model.to(DEVICE)


# =============== Training and Validation ===============
def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        # Check the shape of the inputs tensor to ensure it's as expected
        print(f"Input shape before squeeze: {inputs.shape}")
        
        # Fix input dimensions: [16, 1, 128, 1000] -> [16, 1, 128, 1000]
        # if inputs.dim() == 4:  # Ensure it has a fourth dimension
        #     inputs = inputs.squeeze(3)  # Remove 4th dimension (if present)

        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(train_loader), correct / total


def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            # inputs = inputs.squeeze(3)  # Remove 4th dimension
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(val_loader), correct / total
# =============== MAIN ===============
if __name__ == "__main__":
    print("🚀 Preparing data...")
    train_dataset = ESC50Dataset(ESC50_CSV_PATH, ESC50_AUDIO_DIR, mode='train')
    val_dataset = ESC50Dataset(ESC50_CSV_PATH, ESC50_AUDIO_DIR, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("🚀 Loading pretrained AST model...")
    model = load_ast_model(PRETRAINED_MODEL_PATH)

    # Replace final classifier layer
    in_features = model.mlp_head[-1].in_features
    model.mlp_head = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Linear(in_features, 50)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("🚀 Starting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

    # Save the fine-tuned model
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"\n✅ Fine-tuned model saved to {SAVE_MODEL_PATH}")

    # Plot loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1,2,2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.show()
# import os
# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# import numpy as np
# import librosa
# from torch.utils.data import Dataset, DataLoader
# from src.models.ast_models import ASTModel
# import pandas as pd
# from tqdm import tqdm

# # ==== Step 1: Configuration ====
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# PRETRAINED_MODEL_PATH = "C:/Users/C3I LAB_A09/Desktop/sedd/fine_tuned_ast_esc50.pth"  # <-- Change this
# ESC50_CSV_PATH = "C:/Users/C3I LAB_A09/Desktop/sedd/ESC-50-master/meta/esc50.csv"  # <-- Change this
# ESC50_AUDIO_DIR = "C:/Users/C3I LAB_A09/Desktop/sedd/ESC-50-master/audio"  # <-- Change this
# BATCH_SIZE = 16

# # ==== Step 2: Helper - Audio Preprocessing ====
# def get_log_mel_spectrogram(audio_path):
#     waveform, sr = librosa.load(audio_path, sr=16000)
#     spectrogram = librosa.feature.melspectrogram(
#         y=waveform, sr=sr, n_mels=128, n_fft=1024, hop_length=160
#     )
#     log_spectrogram = librosa.power_to_db(spectrogram)
#     return log_spectrogram

# # ==== Step 3: Dataset Loader ====
# class ESC50Dataset(Dataset):
#     def __init__(self, csv_path, audio_dir):
#         self.meta = pd.read_csv(csv_path)
#         self.audio_dir = audio_dir
#         self.file_paths = []
#         self.labels = []

#         for idx, row in self.meta.iterrows():
#             file_path = os.path.join(audio_dir, row['filename'])
#             label = row['target']
#             self.file_paths.append(file_path)
#             self.labels.append(label)

#     def __getitem__(self, index):
#         spec = get_log_mel_spectrogram(self.file_paths[index])
#         # Ensure consistent size (pad or trim)
#         if spec.shape[1] < 1000:
#             pad_width = 1000 - spec.shape[1]
#             spec = np.pad(spec, ((0,0), (0,pad_width)), mode='constant')
#         else:
#             spec = spec[:, :1000]

#         spec = torch.tensor(spec).unsqueeze(0)  # (1, 128, 1000)
#         spec = spec.permute(0, 2, 1)  # (1, Time, Frequency)
#         label = self.labels[index]
#         return spec, label

#     def __len__(self):
#         return len(self.file_paths)

# class ESC50Dataset(Dataset):
#     def __init__(self, csv_path, audio_dir):
#         self.meta = pd.read_csv(csv_path)
#         self.audio_dir = audio_dir
#         self.file_paths = []
#         self.labels = []

#         for idx, row in self.meta.iterrows():
#             file_path = os.path.join(audio_dir, row['filename'])
#             label = row['target']
#             self.file_paths.append(file_path)
#             self.labels.append(label)

#     def __getitem__(self, index):
#         spec = get_log_mel_spectrogram(self.file_paths[index])  # (128, time)

#         # Pad or crop to 1000 time steps
#         if spec.shape[1] < 1000:
#             pad_width = 1000 - spec.shape[1]
#             spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
#         else:
#             spec = spec[:, :1000]

#         spec = torch.tensor(spec).unsqueeze(0)  # (1, 128, 1000)

#         # 🔥🔥 Fix: Now permute to (channel=1, frequency=128, time=1000)
#         spec = spec.permute(0, 1, 2)  # Actually unnecessary, already correct after unsqueeze!

#         # 🚨 CRITICAL: Now Squeeze any wrong dimension
#         spec = spec.squeeze(0)  # Now (1, 128, 1000)

#         return spec, self.labels[index]

#     def __len__(self):
#         return len(self.file_paths)


# def load_ast_model(pretrained_path):
#     model = ASTModel(
#         label_dim=50,   # ESC-50 has 50 classes
#         fstride=10,
#         tstride=10,
#         input_fdim=128,
#         input_tdim=1000,
#         model_size='base384'
#     )
#     checkpoint = torch.load(pretrained_path, map_location='cpu')

#     # 🔥 CHANGE THIS LINE
#     model.load_state_dict(checkpoint, strict=False)

#     return model.to(DEVICE)


# # ==== Step 5: Main Evaluation ====
# def evaluate(model, dataloader):
#     model.eval()
#     total = 0
#     correct = 0

#     with torch.no_grad():
#         for inputs, labels in tqdm(dataloader, desc="Evaluating"):
#             inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
#             outputs = model(inputs)
#             preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#     accuracy = correct / total
#     print(f"\n✅ Final Test Accuracy: {accuracy*100:.2f}%")

# # ==== Step 6: Run All ====
# if __name__ == "__main__":
#     print("🚀 Loading ESC-50 Dataset...")
#     dataset = ESC50Dataset(ESC50_CSV_PATH, ESC50_AUDIO_DIR)
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

#     print("🚀 Loading Pretrained AST Model...")
#     model = load_ast_model(PRETRAINED_MODEL_PATH)

#     print("🚀 Starting Evaluation...")
#     evaluate(model, dataloader)


import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from src.models.ast_models import ASTModel
import pandas as pd
from tqdm import tqdm

# ==== Step 1: Configuration ====
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETRAINED_MODEL_PATH = "C:/Users/C3I LAB_A09/Desktop/sedd/fine_tuned_ast_esc50.pth"  # <-- Change this
ESC50_CSV_PATH = "C:/Users/C3I LAB_A09/Desktop/sedd/ESC-50-master/meta/esc50.csv"  # <-- Change this
ESC50_AUDIO_DIR = "C:/Users/C3I LAB_A09/Desktop/sedd/ESC-50-master/audio"  # <-- Change this
BATCH_SIZE = 16

# ==== Step 2: Helper - Audio Preprocessing ====
def get_log_mel_spectrogram(audio_path):
    waveform, sr = librosa.load(audio_path, sr=16000)
    spectrogram = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=128, n_fft=1024, hop_length=160
    )
    log_spectrogram = librosa.power_to_db(spectrogram)
    return log_spectrogram

# ==== Step 3: Dataset Loader ====
class ESC50Dataset(Dataset):
    def __init__(self, csv_path, audio_dir):
        self.meta = pd.read_csv(csv_path)
        self.meta = self.meta[self.meta['fold'] == 5]  # 🔥 Only use fold 5 for testing
        self.audio_dir = audio_dir
        self.file_paths = []
        self.labels = []

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

        # 🔥 Already correct shape
        spec = spec.squeeze(0)  # Now (1, 128, 1000)

        return spec, self.labels[index]

    def __len__(self):
        return len(self.file_paths)

# ==== Step 4: Load AST Model ====
def load_ast_model(pretrained_path):
    model = ASTModel(
        label_dim=50,   # ESC-50 has 50 classes
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=1000,
        model_size='base384'
    )
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    model.load_state_dict(checkpoint, strict=False)
    return model.to(DEVICE)

# ==== Step 5: Main Evaluation ====
def evaluate(model, dataloader):
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"\n✅ Final Test Accuracy (Fold 5 only): {accuracy*100:.2f}%")

# ==== Step 6: Run All ====
if __name__ == "__main__":
    print("🚀 Loading ESC-50 Dataset (Fold 5 only)...")
    dataset = ESC50Dataset(ESC50_CSV_PATH, ESC50_AUDIO_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("🚀 Loading Pretrained AST Model...")
    model = load_ast_model(PRETRAINED_MODEL_PATH)

    print("🚀 Starting Evaluation...")
    evaluate(model, dataloader)

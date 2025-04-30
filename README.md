# Audio-Spectrogram-Transformer (AST)

## Setup Instructions

### 1. Clone the Repository

Start by cloning this repository to your local machine:

```bash
git clone https://github.com/asmisriva/AST-Audio-Spectrogram-Transformer_SED.git
cd AST-Audio-Spectrogram-Transformer_SED
```

---

### 2. Install Python Requirements

Make sure you have **Python 3.8 or higher** (Python 3.11 recommended). Then install all required packages:

```bash
pip install torch torchvision torchaudio timm librosa tqdm pandas scikit-learn numpy
```

---

### 3. Download ESC-50 Dataset

Download the [ESC-50 dataset](https://github.com/karoldvl/ESC-50) and organize it as follows:

```
ESC-50-master/
├── audio/               # Contains all .wav files
└── meta/
    └── esc50.csv        # Metadata CSV file
```

Place the `ESC-50-master` directory in the root of this project.

---


### 5. Train the AST Model

To train the AST model on the ESC-50 dataset:

```bash
python train.py
```

---

### 6. Evaluate the AST Model

After training is complete, run:

```bash
python test.py
```

This will evaluate the model’s performance on the test set and report metrics like accuracy.

---

### Folder Structure Overview

```
AST-Audio-Classification/
├── train.py
├── test.py
├── src/
│   └── models/
│       └── ast_models.py
├── model_after_training_ast.pth
├── ESC-50-master/
│   ├── audio/
│   └── meta/
│       └── esc50.csv
```

---

### Result Example (ESC-50)

| Dataset | Accuracy (ours) | Accuracy (original paper) |
|---------|------------------|----------------------------|
| ESC-50  | 82.5%            | 88%                      |

---


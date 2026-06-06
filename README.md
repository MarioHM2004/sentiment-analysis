# Build Your Dream

Final project for DL2026 course.
Binary sentiment classifier (positive/negative) for movie reviews
using DistilBERT fine-tuned on the IMDB dataset.

## Team
- Titouan Donin — Project Lead
- Loup Thomas — Data Lead
- Mario Herranz — Model Lead
- Ziane Badarou — Evaluation & Writing Lead
- Julien Renard — Reproducibility Lead

## Goal
Achieve ≥92% accuracy on binary sentiment classification
(positive/negative) on the IMDB dataset (50,000 reviews).

## Model Architecture
- Base model: `distilbert-base-uncased-finetuned-sst-2-english`
- Classification head: `Dropout(0.3) → Linear(768, 2)`
- Total parameters: 66,364,418

## Project Structure
```
build-your-dream/
├── data/              ← IMDB dataset (auto-downloaded)
├── models/            ← model checkpoints
├── notebooks/         ← evaluation and inference examples
├── src/
│   ├── model.py       ← model architecture
│   ├── train.py       ← training loop
│   └── utils.py       ← inference utilities
├── EXPERIMENTS.md     ← hyperparameter sweep results
├── .gitignore
└── README.md
```

## Installation

### 1. Install Miniconda
Download and install Miniconda from the official site:
- Official: https://docs.conda.io/en/latest/miniconda.html
- China mirror: https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/

### 2. Create the environment
```bash
conda create -n dl2026 python=3.11 -y
conda activate dl2026
```

### 3. Install PyTorch
**NVIDIA GPU (CUDA):**
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**CPU only:**
```bash
pip install torch torchvision torchaudio
```

**Apple Silicon (M1/M2/M3):**
```bash
pip install torch torchvision torchaudio
```

### 4. Install project dependencies
```bash
pip install transformers datasets accelerate
```

### 5. Verify installation
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Training
```bash
conda activate dl2026
python src/train.py
```

## Inference
```python
from src.utils import load_model, predict

model, tokenizer = load_model()

label, confidence = predict(
    "This movie was absolutely amazing!",
    model,
    tokenizer
)
print(f"{label} ({confidence:.2%})")
# positive (96.3%)
```

## Results
See [EXPERIMENTS.md](EXPERIMENTS.md) for the full experiment
log, hyperparameter sweeps, and final results.

**Best result:** 92.13% test accuracy (Experiment 9)

## Dataset
- **Name:** Large Movie Review Dataset (IMDB)
- **Source:** [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)
- **Size:** 25,000 train / 25,000 test reviews
- **Task:** Binary sentiment classification (0=negative, 1=positive)
- **License:** Academic/research use — cite ACL 2011 paper

## Status
- Model trained — 92.13% test accuracy
- Inference functions ready
- Evaluation and error analysis in progress
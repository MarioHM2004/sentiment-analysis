# Sentiment Analysis on Movie Reviews

> DistilBERT fine-tuned on IMDB achieves **92.89 % test accuracy** on binary sentiment classification (positive / negative).

## Setup

```bash
git clone git@github.com:MarioHM2004/sentiment-analysis.git
cd build-your-dream-ai
pip install -r requirements.txt
```

## Goal

Achieve ≥92% accuracy on binary sentiment classification (positive/negative) on the IMDB dataset (50,000 reviews).

## Model Architecture

- **Base model:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Classification head:** Dropout(0.3) → Linear(768, 2)
- **Total parameters:** 66,364,418

## Project Structure

```
build-your-dream/
├── EXPERIMENTS.md             # full experiment log with hyperparameter sweeps
├── LICENSE                    # MIT
├── README.md
├── checkpoints/
│   └── best.pt                # best checkpoint (92.13% test accuracy)
├── configs/
│   └── default.yaml           # all hyperparameters
├── evaluate.py                # loads a checkpoint, prints the headline number
├── notebooks/
│   ├── 01-eda.ipynb           # exploratory data analysis
│   ├── 02-train.ipynb         # the training run, with seeds pinned
│   └── 03-ablations.ipynb     # ablation studies
├── requirements.txt           # exact deps
├── results/
│   ├── figures/
│   │   └── training_curves.png  # loss & accuracy curves
│   └── metrics.json           # headline numbers and ablations
├── src/
│   ├── __init__.py
│   ├── data.py                # dataset loading and preprocessing
│   ├── model.py               # model definition and classification head
│   └── utils.py               # inference helpers (load_model, predict)
|   └── App.py                 # Live demo
└── train.py                   # the entry point
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

## Reproducing our headline result

```bash
# Train from scratch (downloads IMDB automatically, saves checkpoints/best.pt)
python train.py --config configs/default.yaml

# Evaluate a saved checkpoint — prints test accuracy
python evaluate.py --checkpoint checkpoints/best.pt
```

## Demo

```bash
python - <<'EOF'
    from src.utils import load_model, predict
    model, tok = load_model("checkpoints/best.pt", "distilbert-base-uncased-finetuned-sst-2-english")
    label, conf = predict("This movie was absolutely amazing!", model, tok)
    print(f"{label} ({conf:.2%})")
    # positive (99.84%)
EOF
```

# Run life demo
```
python app.py
```

## Results

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full experiment log, hyperparameter sweeps, and final results.

**Best result: 92.13% test accuracy (Experiment 9)**

## Dataset

| Field   | Value |
|---------|-------|
| Name    | Large Movie Review Dataset (IMDB) |
| Source  | `stanfordnlp/imdb` |
| Size    | 25,000 train / 25,000 test reviews |
| Task    | Binary sentiment classification (0=negative, 1=positive) |
| License | Academic/research use — cite ACL 2011 paper |

## Status

- [x] Model trained — 92.89% test accuracy
- [x] Inference functions ready
- [x] Evaluation and error analysis ready

## Authors

- Titouan Donin — titouan.donin@epitech.eu
- Mario Alessandro Herranz Machado — mario-alessandro.herranz-machado@epitech.eu
- Loup Thomas — loup.thomas@epitech.eu
- Ziane Badarou — badaroucedene@icloud.com
- Julien Renard — julien.renard@epitech.eu

## AI Disclosure

Claude (Anthropic) was used for brainstorming, code structure guidance,
and debugging assistance during development. All implementation decisions,
training runs, hyperparameter experiments, and analysis are the authors' own work.

MIT License — see [LICENSE](LICENSE).

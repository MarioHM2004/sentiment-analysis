# Sentiment Analysis on Movie Reviews - build-your-dream-ai

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
- Base model: DistilBERT (`distilbert-base-uncased`)
- Classification head: `Linear(768, 256) → ReLU → Dropout → Linear(256, 2)`
- Total parameters: 66,560,258

## Project Structure
```
├── data/              ← IMDB dataset (auto-downloaded)
├── models/            ← model checkpoints
├── src/
│   ├── model.py       ← model architecture
│   ├── train.py       ← training loop
│   └── utils.py       ← inference utilities
├── EXPERIMENTS.md     ← hyperparameter sweep results
├── .gitignore
└── README.md
```

## Installation
```bash
conda activate dl2026
pip install transformers datasets accelerate
```

## Training
```bash
python src/train.py
```

## Results
See [EXPERIMENTS.md](EXPERIMENTS.md) for the full experiment
log, hyperparameter sweeps, and final results.

## Dataset
- **Name:** Large Movie Review Dataset (IMDB)
- **Source:** [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)
- **Size:** 25,000 train / 25,000 test reviews
- **Labels:** binary (0 = negative, 1 = positive)

## Status
- Hyperparameter sweeps in progress
- Target: ≥92% test accuracy
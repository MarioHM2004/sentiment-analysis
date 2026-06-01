# Experiments — Model Lead (Mario A Herranz)

## Objective
Achieve ≥92% accuracy on IMDB sentiment classification
using DistilBERT fine-tuned on the IMDB dataset.

## Dataset splits
- Train:      20,000 reviews
- Validation:  5,000 reviews
- Test:       25,000 reviews (only used for final evaluation)

> Note: Experiments 1 and 2 used train/test only (no validation split).
> From Experiment 3 onwards: train/val for tuning, test for final evaluation only.

## Key findings
- `MAX_LENGTH=512` was the critical factor — truncating to 256 capped accuracy at ~91%
- `distilbert-base-uncased-finetuned-sst-2-english` consistently outperformed the base checkpoint
- Optimal learning rate: `2e-5` — higher values destroyed pre-trained knowledge
- Overfitting starts after epoch 1-3 depending on configuration
- Best configuration: `lr=2e-5`, `dropout=0.3`, `epochs=3`, `batch=16`, `MAX_LENGTH=512`

## Results

| Exp | lr   | dropout | epochs | batch | max_len | checkpoint | val acc | test acc | notes |
|-----|------|---------|--------|-------|---------|------------|---------|----------|-------|
| 1   | 2e-5 | 0.3     | 3      | 32    | 256     | base       | —       | 90.89%   | no val split — peak epoch 1 |
| 2   | 1e-5 | 0.2     | 4      | 32    | 256     | base       | —       | 89.56%   | no val split — peak epoch 2 |
| 3   | 2e-5 | 0.4     | 2      | 64    | 256     | base       | 89.50%  | —        | first val split |
| 4   | 2e-5 | 0.3     | 5      | 32    | 256     | base       | 91.14%  | 90.67%   | peak epoch 3, overfitting after |
| 5   | 2e-5 | 0.3     | 3      | 32    | 256     | sst-2      | 90.48%  | 91.19%   | switched to sentiment checkpoint |
| 6   | 3e-5 | 0.2     | 2      | 32    | 256     | sst-2      | 89.30%  | 90.37%   | lr too high, destroyed knowledge |
| 7   | 1e-5 | 0.1     | 3      | 32    | 256     | sst-2      | 90.06%  | 90.78%   | lr too low, slow convergence |
| 8   | 2e-5 | 0.3     | 3      | 32    | 256     | sst-2      | 90.08%  | 91.10%   | simplified head: Dropout→Linear |
| 9   | 2e-5 | 0.3     | 3      | 16    | 512     | sst-2      | 91.58%  | **92.13%** | MAX_LENGTH=512 — target reached |

## Best configuration
| Parameter    | Value                                          |
|-------------|------------------------------------------------|
| Checkpoint  | distilbert-base-uncased-finetuned-sst-2-english |
| lr          | 2e-5                                           |
| dropout     | 0.3                                            |
| epochs      | 3                                              |
| batch_size  | 16                                             |
| max_length  | 512                                            |

## Final model
- Saved at: `models/sentiment_classifier.pt`
- Val accuracy:  91.58%
- Test accuracy: **92.13%**
- Parameters:    66,364,418
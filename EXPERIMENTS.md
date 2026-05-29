# Experiments — Model Lead (Mario)

## Objective
Achieve ≥92% accuracy on IMDB sentiment classification.

## Changes since Experiment 1
- Added proper train/val/test split (20k/5k/25k)
- Added final evaluation on test set
- Experiments 1 and 2 used train/test only (no validation split)
- From Experiment 3 onwards: train/val for tuning, test for final evaluation

## Results

| Exp | lr   | dropout | epochs | batch | val acc | test acc | notes |
|-----|------|---------|--------|-------|---------|----------|-------|
| 1   | 2e-5 | 0.3     | 3      | 32    | —       | 90.89%   | no val split, peak epoch 1 |
| 2   | 1e-5 | 0.2     | 4      | 32    | —       | 89.56%   | no val split, peak epoch 2 |
| 3   | 2e-5 | 0.4     | 2      | 64    | ?       | ?        | first run with val split |

## Best configuration so far
TBD

## Final model
- Saved at: models/sentiment_classifier.pt
- Test accuracy: TBD
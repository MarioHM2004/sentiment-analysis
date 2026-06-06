import argparse
import json

import torch
import yaml

from src.data import load_dataloaders
from src.model import SentimentClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            preds = model(input_ids, attention_mask).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path from config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt_path = args.checkpoint or cfg["paths"]["checkpoint"]

    print(f"Loading checkpoint: {ckpt_path}")
    model = SentimentClassifier(
        checkpoint=cfg["model"]["checkpoint"],
        dropout=cfg["model"]["dropout"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    _, _, test_dl = load_dataloaders(cfg)

    test_acc = evaluate(model, test_dl)
    print(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    metrics_path = cfg["paths"]["metrics"]
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["test_acc"] = round(test_acc, 4)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics updated at {metrics_path}")
    except FileNotFoundError:
        pass

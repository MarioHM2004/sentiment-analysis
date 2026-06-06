import argparse
import json
import os

import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt

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


def train(cfg: dict):
    print(f"Device: {DEVICE}")

    train_dl, val_dl, test_dl = load_dataloaders(cfg)

    model = SentimentClassifier(
        checkpoint=cfg["model"]["checkpoint"],
        dropout=cfg["model"]["dropout"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()
    epochs = cfg["training"]["epochs"]

    losses, val_accs = [], []
    log_path = cfg["paths"]["training_log"]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            losses.append(loss.item())

        val_acc = evaluate(model, val_dl)
        val_accs.append(val_acc)
        avg_loss = epoch_loss / len(train_dl)
        print(f"Epoch {epoch+1}/{epochs} | loss: {avg_loss:.4f} | val acc: {val_acc:.4f}")

        with open(log_path, "a") as f:
            f.write(f"Epoch {epoch+1} | lr={cfg['training']['learning_rate']} "
                    f"| dropout={cfg['model']['dropout']} | val acc={val_acc:.4f}\n")

    print(f"\nBest val acc: {max(val_accs):.4f}")

    test_acc = evaluate(model, test_dl)
    print(f"Test acc:     {test_acc:.4f}")

    ckpt_path = cfg["paths"]["checkpoint"]
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    metrics = {
        "val_acc": round(max(val_accs), 4),
        "test_acc": round(test_acc, 4),
        "parameters": total_params,
        "epochs": epochs,
        "learning_rate": cfg["training"]["learning_rate"],
        "dropout": cfg["model"]["dropout"],
        "batch_size": cfg["training"]["batch_size"],
        "max_length": cfg["data"]["max_length"],
    }
    metrics_path = cfg["paths"]["metrics"]
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    figures_dir = cfg["paths"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.plot(losses)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("CE loss")
    ax2.plot(val_accs, marker="o")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "training_curves.png")
    plt.savefig(fig_path)
    print(f"Figures saved to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from datasets import load_dataset
from model import SentimentClassifierModel

# Configuration

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 1e-5
MAX_LENGTH = 256
DROPOUT = 0.2

print(f"Using device: {DEVICE}")

def load_data():
    print("Loading dataset IMDB...")
    dataset = load_dataset("stanfordnlp/imdb")

    # load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    # divide training in train + validation
    train_val = dataset["train"].train_test_split(test_size=0.2, seed=42)

    train_data = train_val["train"]
    val_data = train_val["test"]
    test_data = dataset["test"]

    # tokenize the dataset
    train_data = train_data.map(tokenize, batched=True)
    val_data = val_data.map(tokenize, batched=True)
    test_data = test_data.map(tokenize, batched=True)

    # PyTorch format
    cols = ["input_ids", "attention_mask", "label"]
    train_data.set_format(type="torch", columns=cols)
    val_data.set_format(type="torch", columns=cols)
    test_data.set_format(type="torch", columns=cols)

    # data loaders
    train_dl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train:      {len(train_data)} reviews")
    print(f"Validation: {len(val_data)} reviews")
    print(f"Test:       {len(test_data)} reviews")

    return train_dl, val_dl, test_dl

def evaluate(model, test_dl):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for batch in test_dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            preds =logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

def train(model, train_dl, val_dl):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    losses, accs = [], []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for batch in train_dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            # Forward
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            losses.append(loss.item())

        # Evaluate
        acc = evaluate(model, val_dl)
        accs.append(acc)

        avg_loss = epoch_loss / len(train_dl)
        print(f"Epoch {epoch+1}/{EPOCHS} "
              f"| loss: {avg_loss:.4f} "
              f"| val acc: {acc:.4f}")

    return losses, accs

def main():
    # 1. Load data
    print("Loading data...")
    train_dl, val_dl, test_dl = load_data()

    # 2. Create model
    print("\nCreating model...")
    model = SentimentClassifierModel(dropout=DROPOUT).to(DEVICE)

    # 3. Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters.")

    # 4. Train
    print("\nTraining...")
    losses, accs = train(model, train_dl, val_dl)

    # 5. Validation accuracy
    print(f"\nBest val accuracy: {max(accs):.4f}")
    print(f"Final val accuracy: {accs[-1]:.4f}")
    assert accs[-1] > 0.92, "Expected accuracy to be above 92%"

    # 6. Finak evaluation on test set
    print("\nFinal evaluation on test set...")
    test_acc = evaluate(model, test_dl)
    print(f"Test accuracy: {test_acc:.4f}")

    # 7. Save model
    torch.save(model.state_dict(), "models/sentiment_classifier.pt")
    print("Model saved to models/sentiment_classifier.pt")

    # 8. Graphs
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.5))

    a.plot(losses)
    a.set_title("Training Loss")
    a.set_xlabel("step")
    a.set_ylabel("CE loss")

    b.plot(accs, marker="o")
    b.set_title("Test Accuracy")
    b.set_xlabel("epoch")
    b.set_ylabel("accuracy")

    plt.tight_layout()
    plt.savefig("models/training_curves.png")
    plt.show()
    print("Graphs saved to models/training_curves.png")

if __name__ == "__main__":
    main()
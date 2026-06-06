import torch
from transformers import DistilBertTokenizer
from src.model import SentimentClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: str, model_checkpoint: str, dropout: float = 0.3):
    tokenizer = DistilBertTokenizer.from_pretrained(model_checkpoint)
    model = SentimentClassifier(checkpoint=model_checkpoint, dropout=dropout).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=False))
    model.eval()
    return model, tokenizer


def predict(text: str, model, tokenizer, max_length: int = 512) -> tuple[str, float]:
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).item()
    label = "positive" if pred == 1 else "negative"
    return label, probs[0][pred].item()


def predict_batch(texts: list[str], model, tokenizer, max_length: int = 512) -> list[tuple[str, float]]:
    inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    return [
        ("positive" if preds[i].item() == 1 else "negative", probs[i][preds[i].item()].item())
        for i in range(len(texts))
    ]

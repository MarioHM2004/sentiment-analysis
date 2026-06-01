import torch
from transformers import DistilBertTokenizer
from src.model import SentimentClassifierModel

CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTH = 512
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path="models/sentiment_classifier.pt"):
    """
    Loads the trained sentiment classifier from disk.
    Returns the model and tokenizer ready for inference.
    """
    tokenizer = DistilBertTokenizer.from_pretrained(CHECKPOINT)

    model = SentimentClassifierModel().to(DEVICE)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )
    model.eval()

    return model, tokenizer


def predict(text, model, tokenizer):
    """
    Predicts the sentiment of a single text.

    Args:
        text:      string - the review to classify
        model:     trained SentimentClassifier
        tokenizer: DistilBertTokenizer

    Returns:
        label:      "positive" or "negative"
        confidence: float between 0 and 1
    """
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).item()
    confidence = probs[0][pred].item()

    label = "positive" if pred == 1 else "negative"

    return label, confidence


def predict_batch(texts, model, tokenizer):
    """
    Predicts sentiment for a list of texts.

    Args:
        texts:     list of strings
        model:     trained SentimentClassifier
        tokenizer: DistilBertTokenizer

    Returns:
        list of (label, confidence) tuples
    """
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)

    results = []
    for i in range(len(texts)):
        pred = preds[i].item()
        confidence = probs[i][pred].item()
        label = "positive" if pred == 1 else "negative"
        results.append((label, confidence))

    return results
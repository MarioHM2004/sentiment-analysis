import torch
import torch.nn as nn
from transformers import DistilBertModel

class SentimentClassifierModel(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        # DistilBERT pre-trained
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 2),
        )

    def forward(self, input_ids, attention_mask):
        # run text through DistilBERT
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        # take the [CLS] token output
        cls_output = distilbert_output.last_hidden_state[:, 0, :]

        # clasify
        logits = self.classifier(cls_output)
        return logits
import torch.nn as nn
from transformers import DistilBertModel


class SentimentClassifier(nn.Module):
    def __init__(self, checkpoint: str, dropout: float = 0.3):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(checkpoint)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 2),
        )

    def forward(self, input_ids, attention_mask):
        out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import torch


class HateSpeechClassifier(nn.Module):
    def __init__(self, encoder: AutoModel, num_labels: int = 2):
        super().__init__()
        self.encoder = encoder
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        logits = self.classifier(cls_emb)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return loss, logits


def create_encoder_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    return encoder, tokenizer
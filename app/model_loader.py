import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, List
from transformers import AutoTokenizer, AutoConfig, AutoModel

# Path to model folder
MODEL_DIR = (Path(__file__).resolve().parent.parent / "outputs" / "models" / "best_model")
CLASSIFIER_PATH = MODEL_DIR / "classifier_head.pt"
NUM_LABELS = 2

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    # base encoder (no classifier head)
    encoder = AutoModel.from_pretrained(MODEL_DIR)

    hidden_size = encoder.config.hidden_size

    classifier = nn.Linear(hidden_size, NUM_LABELS)

    # load the saved weights from training
    state_dict = torch.load(CLASSIFIER_PATH, map_location="cpu")
    classifier.load_state_dict(state_dict)

    encoder.eval()
    classifier.eval()

    return tokenizer, encoder, classifier


def predict(text: str, tokenizer, encoder, classifier, max_length: int = 256):
    """
    Run a forward pass on given text and return:
    - predicted label id
    - list of probabilities per class
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    with torch.no_grad():
        outputs = encoder(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]
        logits = classifier(emb)
        probs = torch.softmax(logits, dim=-1)[0].tolist()
        pred_id = int(torch.argmax(logits, dim=-1)[0])

    return pred_id, probs

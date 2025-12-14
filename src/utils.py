import os
import random
import re
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(log_dir: str, name: str = "train") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        ch.setFormatter(ch_fmt)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(ch_fmt)
        logger.addHandler(fh)

    return logger


def load_alay_dict(path: str) -> Dict[str, str]:
    """
    Load new_kamusalay.csv as a dictionary:
    key = alay word (lowercased)
    val = normalized phrase (as-is)
    """
    if not os.path.exists(path):
        print(f"[WARN] Alay dict file not found: {path}. Skipping alay normalization.")
        return {}

    df_alay = pd.read_csv(path, header=None, names=["alay", "normal"], encoding="latin-1")
    df_alay["alay"] = df_alay["alay"].astype(str).str.strip().str.lower()
    df_alay["normal"] = df_alay["normal"].astype(str).str.strip()

    alay_dict = dict(zip(df_alay["alay"], df_alay["normal"]))
    print(f"[INFO] Loaded {len(alay_dict)} alay entries from {path}")

    return alay_dict


def normalize_text(text: str, alay_dict: Dict[str, str]) -> str:
    """
    Preprocess and normalize a single tweet.
    """
    text = text.replace("USER", "")  # Remove USER
    text = re.sub(r"^RT\s+", "", text)  # Remove starting "RT "
    text = re.sub(r"http\S+|www\.\S+", "", text)  # Remove links
    text = " ".join(text.split())  # Normalize whitespace
    text = re.sub(r"\\x\S*", "", text)  # Remove emojis encoded as \xf0...

    # Alay normalization
    tokens = text.split()
    normalized_tokens: List[str] = []

    for tok in tokens:
        key = tok.lower()
        normalized_tokens.append(alay_dict.get(key, tok))

    return " ".join(normalized_tokens)

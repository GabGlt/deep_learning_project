import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Tuple
from .utils import normalize_text, load_alay_dict


class SimCSEDataset(Dataset):
    """
    Dataset for unsupervised SimCSE pretraining.
    Only needs tokenized input (no labels).
    """
    def __init__(self, texts: List[str], tokenizer: AutoTokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


class HateSpeechDataset(Dataset):
    """
    Supervised dataset for binary classification.
    label: 0 = neutral, 1 = hate/abusive
    """
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_and_split_data(csv_path: str, alay_dict: Dict[str, str], seed: int, logger=None):
    """
    Load CSV, create Toxic label (from multi-class to binary class), normalize text, and split into train/val/test.
    """
    df = pd.read_csv(csv_path, encoding="latin-1")

    df["HS"] = df["HS"].astype(int)
    df["Abusive"] = df["Abusive"].astype(int)
    df["Toxic"] = ((df["HS"] == 1) | (df["Abusive"] == 1)).astype(int)

    df = df.dropna(subset=["Tweet"])
    count_toxic = df["Toxic"].sum()

    msg = (
        f"Found {len(df)} rows \n"
        f"{count_toxic} ({(count_toxic / len(df) * 100):.2f}%) toxic \n"
        f"{len(df) - count_toxic} ({((len(df) - count_toxic) / len(df) * 100):.2f}%) neutral"
    )
    if logger:
        logger.info(msg)
    else:
        print(msg)

    texts = [normalize_text(text, alay_dict) for text in df["Tweet"].tolist()]
    labels = df["Toxic"].tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=seed,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=seed,
    )

    if logger:
        logger.info(f"Train size: {len(X_train)} ({len(X_train) / len(df) * 100:.2f}%)")
        logger.info(f"Val size: {len(X_val)} ({len(X_val) / len(df) * 100:.2f}%)")
        logger.info(f"Test size: {len(X_test)} ({len(X_test) / len(df) * 100:.2f}%)")
    else:
        print(f"Train size: {len(X_train)} ({len(X_train) / len(df) * 100:.2f}%)")
        print(f"Val size: {len(X_val)} ({len(X_val) / len(df) * 100:.2f}%)")
        print(f"Test size: {len(X_test)} ({len(X_test) / len(df) * 100:.2f}%)")

    return texts, labels, X_train, X_val, X_test, y_train, y_val, y_test


def create_dataloaders(X_train: List[str], X_val: List[str], X_test: List[str], y_train: List[int], y_val: List[int], y_test: List[int], tokenizer: AutoTokenizer, max_len: int, batch_size_cls: int):
    train_dataset = HateSpeechDataset(X_train, y_train, tokenizer, max_len)
    val_dataset = HateSpeechDataset(X_val, y_val, tokenizer, max_len)
    test_dataset = HateSpeechDataset(X_test, y_test, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_cls, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_cls, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_cls, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
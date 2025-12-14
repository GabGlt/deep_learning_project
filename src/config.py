# src/config.py
import os
from dataclasses import dataclass
import torch


@dataclass
class Config:
    data_csv_path: str = "dataset/re_dataset.csv"
    wordmap_csv_path: str = "dataset/new_kamusalay.csv"
    model_name: str = "indolem/indobert-base-uncased"
    # class_names: list = ["Neutral", "Hate_Abusive"]
    max_len: int = 256

    batch_size_simcse: int = 128
    batch_size_cls: int = 64

    num_epochs_simcse: int = 2
    num_epochs_cls: int = 3

    simcse_temperature: float = 0.05

    lr_simcse: float = 5e-5
    lr_cls: float = 3e-5
    weight_decay: float = 0.01

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    outputs_dir: str = "outputs"
    models_dir: str = os.path.join(outputs_dir, "models")
    logs_dir: str = os.path.join(outputs_dir, "logs")
    plots_dir: str = os.path.join(outputs_dir, "plots")
    best_model_dir: str = os.path.join(models_dir, "best_model")

    use_simcse: bool = True

config = Config()

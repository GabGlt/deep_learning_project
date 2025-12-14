import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def compute_metrics(y_true: List[int], y_pred: List[int], class_names: List[str]) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    return {"accuracy": acc, "confusion_matrix": cm, "report": report}


def save_metrics(metrics: Dict, logs_dir: str, prefix: str = "test"):
    os.makedirs(logs_dir, exist_ok=True)
    report_path = os.path.join(logs_dir, f"{prefix}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(metrics["report"])
        f.write(f"\nAccuracy: {metrics['accuracy']:.4f}\n")

    cm_path = os.path.join(logs_dir, f"{prefix}_confusion_matrix.npy")
    np.save(cm_path, metrics["confusion_matrix"])


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]),
                    ha="center", va="center")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: Dict[str, List[float]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Loss curves
    fig, ax = plt.subplots()
    ax.plot(history["train_loss"], label="Train loss")
    ax.plot(history["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), bbox_inches="tight")
    plt.close(fig)

    # Accuracy curve
    fig, ax = plt.subplots()
    ax.plot(history["val_acc"], label="Val accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_accuracy.png"), bbox_inches="tight")
    plt.close(fig)

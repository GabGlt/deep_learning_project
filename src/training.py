import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, AutoModel
from typing import Dict, List, Tuple
from .data_loader import SimCSEDataset, HateSpeechDataset
from .evaluation import compute_metrics, save_metrics, plot_confusion_matrix, plot_training_curves


def simcse_loss(encoder: AutoModel, batch: Dict[str, torch.Tensor], temperature: float = 0.05) -> torch.Tensor:
    """
    Unsupervised SimCSE loss with two dropout-masked views.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    outputs1 = encoder(input_ids=input_ids, attention_mask=attention_mask)
    outputs2 = encoder(input_ids=input_ids, attention_mask=attention_mask)

    z1 = outputs1.last_hidden_state[:, 0, :]  # (B, H)
    z2 = outputs2.last_hidden_state[:, 0, :]  # (B, H)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temperature  # (2B, 2B)

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(mask, -1e12)

    labels = torch.arange(2 * batch_size, device=sim.device)
    labels = (labels + batch_size) % (2 * batch_size)

    loss = F.cross_entropy(sim, labels)
    return loss


def train_simcse(encoder: AutoModel, dataloader: DataLoader, num_epochs: int, lr: float, temperature: float, device: str, logger=None):
    encoder.to(device)
    encoder.train()

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)
    num_training_steps = num_epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    for epoch in range(num_epochs):
        encoder.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"[SimCSE] Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            loss = simcse_loss(encoder, batch, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            avg_loss = total_loss / (pbar.n + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        epoch_loss = total_loss / len(dataloader)
        msg = f"[SimCSE] Epoch {epoch+1}/{num_epochs} - avg loss: {epoch_loss:.4f}"
        if logger:
            logger.info(msg)
        else:
            print(msg)


def train_and_evaluate_classifier(model, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, num_epochs: int, lr: float, device: str, best_model_dir: str, weight_decay: float, tokenizer, logs_dir: str, plots_dir: str, class_names: List[str], logger=None):
    model.to(device)
    os.makedirs(best_model_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    best_val_acc = 0.0
    best_state_dict = None

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        # Train 
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"[CLS] Epoch {epoch+1}/{num_epochs} (train)")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix({"loss": f"{total_train_loss/train_steps:.4f}"})

        avg_train_loss = total_train_loss / max(1, train_steps)

        # Validation 
        model.eval()
        total_val_loss = 0.0
        val_steps = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[CLS] Epoch {epoch+1}/{num_epochs} (val)"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += loss.item()
                val_steps += 1

                preds = torch.argmax(logits, dim=1)
                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())

        avg_val_loss = total_val_loss / max(1, val_steps)
        val_acc = sum(int(a == b) for a, b in zip(all_labels, all_preds)) / len(all_labels)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        msg = (
            f"[CLS] Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )
        if logger:
            logger.info(msg)
        else:
            print(msg)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()
            torch.save(best_state_dict, os.path.join(best_model_dir, "best_model.pt"))
            if logger:
                logger.info(f"[CLS] New best model saved with val_acc={best_val_acc:.4f}")
            else:
                print(f"[CLS] New best model saved with val_acc={best_val_acc:.4f}")

    # Load best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    else:
        if logger:
            logger.warning("[WARN] No best model state dict found, using last epoch model.")
        else:
            print("[WARN] No best model state dict found, using last epoch model.")

    # Save encoder + tokenizer + head in HF-style
    if hasattr(model.encoder, "save_pretrained"):
        model.encoder.save_pretrained(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
        torch.save(model.classifier.state_dict(), os.path.join(best_model_dir, "classifier_head.pt"))

    # Test Evaluation
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[TEST]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_preds, class_names)
    save_metrics(metrics, logs_dir, prefix="test")

    if logger:
        logger.info(f"[TEST] Accuracy: {metrics['accuracy']:.4f}")
        logger.info("\n" + metrics["report"])
    else:
        print(f"\n[TEST] Accuracy: {metrics['accuracy']:.4f}\n")
        print(metrics["report"])

    # Confusion matrix plot
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, cm_path)

    # Training curves
    plot_training_curves(history, plots_dir)

    return history, metrics

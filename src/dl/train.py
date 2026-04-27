"""
Deep learning training loop with:
    - Early stopping
    - Learning rate scheduling (ReduceLROnPlateau)
    - Model checkpoint saving
    - Training log CSV export
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"  Early stopping triggered after {self.counter} epochs without improvement.")
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None, total_epochs=None, model_tag="model"):
    """Run one training epoch, return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    batch_iterator = tqdm(
        dataloader,
        desc=f"{model_tag} e{epoch}/{total_epochs} batches",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

    for batch_index, (images, labels) in enumerate(batch_iterator, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        batch_iterator.set_postfix(
            batch=batch_index,
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total:.4f}",
        )

    batch_iterator.close()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Run validation, return average loss, accuracy, all predictions and labels."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train_model(model, train_dataset, val_dataset, training_cfg, results_dir,
                model_tag="ResNet18", random_seed=42):
    """
    Full training loop with early stopping, LR scheduling, and checkpointing.

    Args:
        model: PyTorch model.
        train_dataset: Training Dataset.
        val_dataset: Validation Dataset.
        training_cfg: Dict with training hyperparameters from dl.yaml.
        results_dir: Base directory for saving results.
        model_tag: Tag for naming saved files.
        random_seed: Random seed.

    Returns:
        model: Trained model (best checkpoint loaded).
        log_path: Path to the training log CSV.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    results_dir = Path(results_dir)
    models_dir = results_dir / "models"
    logs_dir = results_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"  Using device: {device}")
    model = model.to(device)

    # DataLoaders
    batch_size = training_cfg.get("batch_size", 32)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=pin_memory)

    # Loss, optimizer
    criterion = nn.CrossEntropyLoss()
    lr = training_cfg.get("learning_rate", 0.001)
    wd = training_cfg.get("weight_decay", 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Scheduler
    sched_cfg = training_cfg.get("scheduler", {})
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 3),
        min_lr=sched_cfg.get("min_lr", 1e-5),
    )

    # Early stopping
    es_cfg = training_cfg.get("early_stopping", {})
    early_stopping = EarlyStopping(
        patience=es_cfg.get("patience", 7),
        min_delta=es_cfg.get("min_delta", 0.001),
    )

    epochs = training_cfg.get("epochs", 30)
    best_val_loss = float("inf")
    checkpoint_path = models_dir / f"{model_tag}_best.pt"
    log_records = []

    logger.info(f"  Starting training for up to {epochs} epochs...")

    epoch_progress = tqdm(range(1, epochs + 1), desc=f"{model_tag} epochs", unit="epoch", dynamic_ncols=True, leave=True)
    for epoch in epoch_progress:
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=epochs,
            model_tag=model_tag,
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | {elapsed:.1f}s"
        )
        epoch_progress.set_postfix(train_loss=f"{train_loss:.4f}", val_acc=f"{val_acc:.4f}", lr=f"{current_lr:.2e}")

        log_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        })

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"    ✓ Best model saved (val_loss={val_loss:.4f})")

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.should_stop:
            epoch_progress.close()
            break

    if not early_stopping.should_stop:
        epoch_progress.close()

    # Save training log
    log_path = logs_dir / f"{model_tag}_training_log.csv"
    pd.DataFrame(log_records).to_csv(log_path, index=False)
    logger.info(f"  Training log saved to {log_path}")

    # Load best checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    logger.info(f"  Loaded best checkpoint from {checkpoint_path}")

    return model, log_path

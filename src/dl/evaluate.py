"""
Deep learning model evaluation on test set.
"""

import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dl.train import validate
from src.evaluation.metrics import compute_metrics, save_classification_report, save_confusion_matrix_plot

logger = logging.getLogger(__name__)


def evaluate_model(model, test_dataset, class_names, results_dir,
                   model_tag="ResNet18", split_name="test", batch_size=32):
    """
    Evaluate a trained DL model on a dataset split.

    Saves classification report, confusion matrix, and metrics CSV.

    Args:
        model: Trained PyTorch model.
        test_dataset: PyTorch Dataset.
        class_names: List of class name strings.
        results_dir: Base results directory.
        model_tag: Tag for naming saved files.
        split_name: Name of the split (e.g., "test", "val").
        batch_size: Batch size for evaluation.

    Returns:
        metrics_dict: Dictionary of metrics.
    """
    results_dir = Path(results_dir)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    pin_memory = device.type == "cuda"
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=pin_memory)

    criterion = torch.nn.CrossEntropyLoss()
    _, accuracy, y_pred, y_true = validate(model, test_loader, criterion, device)

    metrics = compute_metrics(y_true, y_pred, class_names)

    logger.info(f"  {model_tag} on {split_name}:")
    logger.info(f"    Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"    Macro F1:  {metrics['macro_f1']:.4f}")

    # Classification report
    report_path = results_dir / "metrics" / f"{model_tag}_{split_name}_report.csv"
    save_classification_report(y_true, y_pred, class_names, report_path)

    # Confusion matrix
    cm_path = results_dir / "confusion_matrices" / f"{model_tag}_{split_name}_cm.png"
    save_confusion_matrix_plot(y_true, y_pred, class_names, cm_path,
                               title=f"{model_tag} — {split_name}")

    # Summary metrics
    metrics_path = results_dir / "metrics" / f"{model_tag}_{split_name}_metrics.csv"
    summary = pd.DataFrame([{
        "model": model_tag,
        "split": split_name,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
    }])
    summary.to_csv(metrics_path, index=False)

    return metrics

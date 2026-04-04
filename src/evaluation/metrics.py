"""
Evaluation metrics computation and saving utilities.

Used by both ML and DL pipelines.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, class_names):
    """
    Compute classification metrics.

    Args:
        y_true: True labels (1D array).
        y_pred: Predicted labels (1D array).
        class_names: List of class name strings.

    Returns:
        Dictionary with:
            accuracy, macro_precision, macro_recall, macro_f1,
            per_class_f1 (dict keyed by class name).
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    per_class_f1 = {}
    for i, name in enumerate(class_names):
        if i < len(per_class):
            per_class_f1[name] = per_class[i]

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
    }


def save_classification_report(y_true, y_pred, class_names, output_path):
    """
    Save a sklearn classification report as CSV.

    Args:
        y_true, y_pred: Label arrays.
        class_names: List of class names.
        output_path: Path to save CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df.to_csv(output_path)
    logger.info(f"  Classification report saved to {output_path}")


def save_confusion_matrix_plot(y_true, y_pred, class_names, output_path, title="Confusion Matrix"):
    """
    Save a confusion matrix heatmap as PNG.

    Args:
        y_true, y_pred: Label arrays.
        class_names: List of class names.
        output_path: Path to save PNG.
        title: Plot title string.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Confusion matrix saved to {output_path}")

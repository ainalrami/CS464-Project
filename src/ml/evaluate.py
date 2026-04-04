"""
Evaluation utilities for the classical ML pipeline.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics, save_classification_report, save_confusion_matrix_plot

logger = logging.getLogger(__name__)


def evaluate_model(pipeline, X, y, class_names, split_name, model_name, results_dir):
    """
    Evaluate a trained ML pipeline on features X with labels y.

    Saves:
        - Classification report CSV
        - Confusion matrix heatmap PNG
        - Metrics summary CSV

    Args:
        pipeline: Fitted sklearn Pipeline.
        X: Feature matrix (N, D).
        y: True labels (N,).
        class_names: List of class name strings.
        split_name: Name of the split (e.g., "val", "test").
        model_name: Name of the model.
        results_dir: Base results directory.

    Returns:
        metrics_dict: Dictionary with accuracy, precision, recall, f1, per_class_f1.
    """
    y_pred = pipeline.predict(X)

    metrics = compute_metrics(y, y_pred, class_names)

    logger.info(f"  {model_name} on {split_name}:")
    logger.info(f"    Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"    Macro F1:  {metrics['macro_f1']:.4f}")

    # Save results
    results_dir = Path(results_dir)

    # Classification report
    report_path = results_dir / "metrics" / f"{model_name}_{split_name}_report.csv"
    save_classification_report(y, y_pred, class_names, report_path)

    # Confusion matrix
    cm_path = results_dir / "confusion_matrices" / f"{model_name}_{split_name}_cm.png"
    save_confusion_matrix_plot(y, y_pred, class_names, cm_path, title=f"{model_name} — {split_name}")

    # Summary metrics CSV
    metrics_path = results_dir / "metrics" / f"{model_name}_{split_name}_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([{
        "model": model_name,
        "split": split_name,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
    }])
    summary_df.to_csv(metrics_path, index=False)

    return metrics

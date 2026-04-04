"""
Publication-style plotting utilities.

Generates:
    - Training curves (loss and accuracy)
    - Robustness comparison bar/line charts
    - Per-class F1 grouped bar charts
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Use a clean style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def plot_training_curves(log_csv_path, output_dir):
    """
    Plot training and validation loss/accuracy curves from a CSV log.

    Expected CSV columns: epoch, train_loss, val_loss, train_acc, val_acc

    Args:
        log_csv_path: Path to training log CSV.
        output_dir: Directory to save plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_csv_path)

    # Loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df["epoch"], df["train_loss"], "o-", label="Train Loss", markersize=4)
    ax1.plot(df["epoch"], df["val_loss"], "s-", label="Val Loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()

    ax2.plot(df["epoch"], df["train_acc"], "o-", label="Train Accuracy", markersize=4)
    ax2.plot(df["epoch"], df["val_acc"], "s-", label="Val Accuracy", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)
    logger.info(f"  Training curves saved to {output_dir / 'training_curves.png'}")


def plot_robustness_comparison(results_df, output_path):
    """
    Plot a grouped bar chart comparing model accuracy across degradation conditions.

    Args:
        results_df: DataFrame with columns: model, condition, accuracy.
        output_path: Path to save the plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    models = results_df["model"].unique()
    conditions = results_df["condition"].unique()
    n_models = len(models)
    n_conditions = len(conditions)

    x = np.arange(n_conditions)
    width = 0.8 / n_models

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, model in enumerate(models):
        model_data = results_df[results_df["model"] == model]
        accs = []
        for cond in conditions:
            row = model_data[model_data["condition"] == cond]
            accs.append(row["accuracy"].values[0] if len(row) > 0 else 0)
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=model, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness Comparison: Clean vs. Degraded Test Sets")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Robustness comparison plot saved to {output_path}")


def plot_per_class_f1(per_class_results, class_names, output_path):
    """
    Plot per-class F1 comparison across models.

    Args:
        per_class_results: Dict of {model_name: {class_name: f1_score}}.
        class_names: List of class name strings.
        output_path: Path to save the plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = list(per_class_results.keys())
    n_models = len(models)
    n_classes = len(class_names)

    x = np.arange(n_classes)
    width = 0.8 / n_models
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, model in enumerate(models):
        f1s = [per_class_results[model].get(cn, 0) for cn in class_names]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, f1s, width, label=model, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Comparison")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Per-class F1 plot saved to {output_path}")


def plot_model_comparison_bar(summary_df, output_path):
    """
    Plot a bar chart comparing overall metrics (accuracy, macro F1) across models.

    Args:
        summary_df: DataFrame with columns: model, accuracy, macro_f1.
        output_path: Path to save the plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    models = summary_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width / 2, summary_df["accuracy"], width, label="Accuracy", color="#4C72B0")
    ax.bar(x + width / 2, summary_df["macro_f1"], width, label="Macro F1", color="#55A868")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Test Set")
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Model comparison plot saved to {output_path}")

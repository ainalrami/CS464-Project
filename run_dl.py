#!/usr/bin/env python3
"""
Deep Learning Pipeline Entry Point
====================================

Usage:
    python run_dl.py --config configs/dl.yaml

Runs the complete DL pipeline:
    1. Load dataset & create stratified splits
    2. Build ResNet18 with transfer learning
    3. Train with early stopping & LR scheduling
    4. Evaluate on validation and test sets
    5. Optionally compare with/without augmentation
    6. Save metrics, confusion matrices, training curves, and model checkpoint
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import (
    load_dataset, create_splits, save_split_metadata,
    load_split_metadata, EuroSATDataset, get_dl_transforms,
)
from src.dl.model import build_model
from src.dl.train import train_model
from src.dl.evaluate import evaluate_model
from src.evaluation.plots import plot_training_curves

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_dl")


def run_training(cfg, augmentation_enabled, results_dir, model_tag):
    """
    Run one full DL training + evaluation cycle.

    Args:
        cfg: Full config dict.
        augmentation_enabled: Whether to use data augmentation.
        results_dir: Output directory.
        model_tag: Tag for naming saved files.
    """
    random_seed = cfg.get("random_seed", 42)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    dataset_root = cfg["dataset"]["root"]
    image_size = tuple(cfg["dataset"].get("image_size", [224, 224]))
    split_cfg = cfg["split"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    aug_cfg = cfg.get("augmentation", {})

    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Load or create splits
    split_file = logs_dir / "split_train.csv"
    if split_file.exists():
        logger.info("Loading existing split metadata...")
        splits, class_names = load_split_metadata(logs_dir)
    else:
        logger.info("Loading dataset and creating new splits...")
        data, class_names = load_dataset(dataset_root)
        splits = create_splits(
            data,
            train_ratio=split_cfg["train_ratio"],
            val_ratio=split_cfg["val_ratio"],
            test_ratio=split_cfg["test_ratio"],
            random_seed=split_cfg["random_seed"],
        )
        save_split_metadata(splits, logs_dir, class_names)

    logger.info(f"Classes: {class_names}")

    # Build transforms
    if augmentation_enabled:
        train_transform = get_dl_transforms(image_size, aug_cfg, is_train=True)
    else:
        train_transform = get_dl_transforms(image_size, None, is_train=False)

    eval_transform = get_dl_transforms(image_size, None, is_train=False)

    # Build datasets
    train_dataset = EuroSATDataset(splits["train"], transform=train_transform)
    val_dataset = EuroSATDataset(splits["val"], transform=eval_transform)
    test_dataset = EuroSATDataset(splits["test"], transform=eval_transform)

    logger.info(f"Dataset sizes — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Build model
    model = build_model(
        num_classes=model_cfg.get("num_classes", 10),
        pretrained=model_cfg.get("pretrained", True),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    )

    # Train
    logger.info(f"\nTraining {model_tag} (augmentation={'ON' if augmentation_enabled else 'OFF'})...")
    model, log_path = train_model(
        model, train_dataset, val_dataset,
        training_cfg, results_dir,
        model_tag=model_tag,
        random_seed=random_seed,
    )

    # Plot training curves
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(log_path, plots_dir)
    # Rename to include model tag
    default_curve = plots_dir / "training_curves.png"
    tagged_curve = plots_dir / f"{model_tag}_training_curves.png"
    if default_curve.exists():
        default_curve.rename(tagged_curve)

    # Evaluate on val
    logger.info("Evaluating on validation set...")
    val_metrics = evaluate_model(
        model, val_dataset, class_names, results_dir,
        model_tag=model_tag, split_name="val",
        batch_size=training_cfg.get("batch_size", 32),
    )

    # Evaluate on test
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(
        model, test_dataset, class_names, results_dir,
        model_tag=model_tag, split_name="test",
        batch_size=training_cfg.get("batch_size", 32),
    )

    return {
        "model": model_tag,
        "augmentation": augmentation_enabled,
        "val_accuracy": val_metrics["accuracy"],
        "val_f1": val_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["macro_f1"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run deep learning pipeline on EuroSAT RGB.")
    parser.add_argument("--config", type=str, default="configs/dl.yaml",
                        help="Path to DL config YAML file.")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg.get("results_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Train WITH augmentation
    result = run_training(cfg, augmentation_enabled=True, results_dir=results_dir,
                          model_tag="ResNet18_aug")
    all_results.append(result)

    # Optionally train WITHOUT augmentation for comparison
    if cfg.get("compare_augmentation", False):
        logger.info("\n" + "=" * 60)
        logger.info("Running comparison: WITHOUT augmentation")
        logger.info("=" * 60)
        result_no_aug = run_training(cfg, augmentation_enabled=False, results_dir=results_dir,
                                     model_tag="ResNet18_noaug")
        all_results.append(result_no_aug)

    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_path = results_dir / "metrics" / "dl_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    logger.info(f"\n{'=' * 60}")
    logger.info("DL Pipeline Complete!")
    logger.info(f"{'=' * 60}")
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"\n{summary_df.to_string(index=False)}")


if __name__ == "__main__":
    main()

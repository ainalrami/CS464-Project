#!/usr/bin/env python3
"""
Classical ML Pipeline Entry Point
==================================

Usage:
    python run_ml.py --config configs/ml.yaml

Runs the complete classical ML pipeline:
    1. Load dataset & create stratified splits
    2. Extract handcrafted features (HOG, color histogram, LBP)
    3. Train & tune SVM, Random Forest, and XGBoost/LogReg
    4. Evaluate on validation and test sets
    5. Save metrics, confusion matrices, and model files
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.dataset import load_dataset, create_splits, save_split_metadata, load_split_metadata
from src.features.extractors import extract_features_batch
from src.ml.train import train_model, save_model
from src.ml.evaluate import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_ml")


def main():
    parser = argparse.ArgumentParser(description="Run classical ML pipeline on EuroSAT RGB.")
    parser.add_argument("--config", type=str, default="configs/ml.yaml",
                        help="Path to ML config YAML file.")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    random_seed = cfg.get("random_seed", 42)
    np.random.seed(random_seed)

    results_dir = Path(cfg.get("results_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Load dataset & create splits
    # ----------------------------------------------------------------
    dataset_root = cfg["dataset"]["root"]
    split_cfg = cfg["split"]
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Check if splits already exist (for reproducibility across runs)
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
    logger.info(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    # ----------------------------------------------------------------
    # 2. Feature extraction (for each ablation mode)
    # ----------------------------------------------------------------
    feature_cfg = cfg.get("features", {})
    image_size = tuple(cfg["dataset"].get("image_size", [64, 64]))
    hog_cfg = feature_cfg.get("hog", {})
    color_cfg = feature_cfg.get("color_histogram", {})
    lbp_cfg = feature_cfg.get("lbp", {})
    feature_modes = feature_cfg.get("modes", ["hog_color_texture"])

    # ----------------------------------------------------------------
    # 3 & 4. Train and evaluate for each feature mode × model
    # ----------------------------------------------------------------
    model_configs = cfg.get("models", [])
    all_results = []

    for feat_mode in feature_modes:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Feature mode: {feat_mode}")
        logger.info(f"{'=' * 60}")

        # Extract features
        logger.info("Extracting training features...")
        X_train, y_train = extract_features_batch(
            splits["train"], image_size=image_size, mode=feat_mode,
            hog_cfg=hog_cfg, color_cfg=color_cfg, lbp_cfg=lbp_cfg,
        )

        logger.info("Extracting validation features...")
        X_val, y_val = extract_features_batch(
            splits["val"], image_size=image_size, mode=feat_mode,
            hog_cfg=hog_cfg, color_cfg=color_cfg, lbp_cfg=lbp_cfg,
        )

        logger.info("Extracting test features...")
        X_test, y_test = extract_features_batch(
            splits["test"], image_size=image_size, mode=feat_mode,
            hog_cfg=hog_cfg, color_cfg=color_cfg, lbp_cfg=lbp_cfg,
        )

        for model_cfg in model_configs:
            model_name_cfg = model_cfg["name"]
            tag = f"{model_name_cfg}_{feat_mode}"

            logger.info(f"\n--- Training {tag} ---")
            t0 = time.time()

            pipeline, best_params, cv_score, actual_name = train_model(
                X_train, y_train, model_cfg, random_seed=random_seed
            )
            train_time = time.time() - t0
            tag = f"{actual_name}_{feat_mode}"

            logger.info(f"  Training time: {train_time:.1f}s")

            # Save model
            save_model(pipeline, tag, results_dir / "models")

            # Evaluate on val
            logger.info("  Evaluating on validation set...")
            val_metrics = evaluate_model(
                pipeline, X_val, y_val, class_names,
                split_name="val", model_name=tag, results_dir=results_dir,
            )

            # Evaluate on test
            logger.info("  Evaluating on test set...")
            test_metrics = evaluate_model(
                pipeline, X_test, y_test, class_names,
                split_name="test", model_name=tag, results_dir=results_dir,
            )

            all_results.append({
                "model": actual_name,
                "feature_mode": feat_mode,
                "cv_score": cv_score,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1": test_metrics["macro_f1"],
                "train_time_s": train_time,
            })

    # ----------------------------------------------------------------
    # 5. Save summary table
    # ----------------------------------------------------------------
    summary_df = pd.DataFrame(all_results)
    summary_path = results_dir / "metrics" / "ml_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    logger.info(f"\n{'=' * 60}")
    logger.info("ML Pipeline Complete!")
    logger.info(f"{'=' * 60}")
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"\n{summary_df.to_string(index=False)}")


if __name__ == "__main__":
    main()

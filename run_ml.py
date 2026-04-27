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
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

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
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model: SVM | RandomForest | XGBoost  (default: all)")
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
    # 2. Feature extraction (for each ablation mode × image size)
    # ----------------------------------------------------------------
    feature_cfg = cfg.get("features", {})
    hog_cfg = feature_cfg.get("hog", {})
    color_cfg = feature_cfg.get("color_histogram", {})
    lbp_cfg = feature_cfg.get("lbp", {})
    feature_modes = feature_cfg.get("modes", ["hog_color_texture"])

    # Build list of (size_label, image_size, modes_to_run) triples
    native_size = tuple(cfg["dataset"].get("image_size", [64, 64]))
    # upscaled_modes: subset of modes to run at larger size (avoids running all 3 modes × 2 sizes)
    upscaled_modes = cfg.get("upscaled_modes", feature_modes)

    experiments = []
    if not cfg.get("only_upscaled", False):
        experiments.append(("native", native_size, feature_modes))
    if cfg.get("compare_upscaled", False):
        up_size = tuple(cfg.get("image_size_upscaled", [128, 128]))
        experiments.append(("upscaled", up_size, upscaled_modes))

    if not experiments:
        logger.error("No experiments selected. Check 'only_upscaled' and 'compare_upscaled' in config.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 3 & 4. Train and evaluate for each image size × feature mode × model
    # ----------------------------------------------------------------
    model_configs = cfg.get("models", [])
    if args.model:
        model_configs = [m for m in model_configs if m["name"].lower() == args.model.lower()]
        if not model_configs:
            logger.error(f"Model '{args.model}' not found in config. Available: {[m['name'] for m in cfg.get('models', [])]}")
            sys.exit(1)
        logger.info(f"Running only: {args.model}")
    total_experiments = sum(len(modes_for_size) for _, _, modes_for_size in experiments) * len(model_configs)
    all_results = []

    completed_experiments = 0
    with tqdm(total=total_experiments, desc="ML progress", unit="model", dynamic_ncols=True) as progress_bar:
        for size_label, image_size, modes_for_size in experiments:
            for feat_mode in modes_for_size:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Image size: {image_size} ({size_label}) | Feature mode: {feat_mode}")
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
                    completed_experiments += 1
                    progress_bar.set_postfix(stage=f"{model_name_cfg} | {feat_mode} | {size_label}")

                    logger.info(
                        f"\n--- Training {model_name_cfg} [{feat_mode}, {size_label}] "
                        f"({completed_experiments}/{total_experiments}) ---"
                    )
                    t0 = time.time()

                    pipeline, best_params, cv_score, actual_name = train_model(
                        X_train, y_train, model_cfg, random_seed=random_seed
                    )
                    train_time = time.time() - t0

                    # Tag includes size_label so upscaled and native models don't overwrite each other
                    size_suffix = "" if size_label == "native" else f"_{size_label}"
                    tag = f"{actual_name}_{feat_mode}{size_suffix}"

                    logger.info(f"  Training time: {train_time:.1f}s  |  tag={tag}")

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
                        "image_size": f"{image_size[0]}x{image_size[1]}",
                        "size_label": size_label,
                        "cv_score": cv_score,
                        "val_accuracy": val_metrics["accuracy"],
                        "val_f1": val_metrics["macro_f1"],
                        "test_accuracy": test_metrics["accuracy"],
                        "test_f1": test_metrics["macro_f1"],
                        "train_time_s": train_time,
                    })
                    progress_bar.update(1)

                # Release large matrices before next feature mode / image size block.
                del X_train, y_train, X_val, y_val, X_test, y_test
                gc.collect()

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

#!/usr/bin/env python3
"""
Robustness Analysis Entry Point
=================================

Usage:
    python run_robustness.py --config configs/robustness.yaml

Evaluates all trained models (ML + DL) on degraded test sets:
    - Gaussian blur (multiple severity levels)
    - Gaussian noise (multiple severity levels)
    - Downsampling (multiple factors)

Produces:
    - robustness_results.csv: Comparison table
    - robustness_comparison.png: Bar chart
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

from src.data.dataset import load_split_metadata, get_dl_transforms
from src.dl.model import build_model
from src.robustness.evaluate import run_robustness_evaluation
from src.evaluation.plots import plot_robustness_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_robustness")


def main():
    parser = argparse.ArgumentParser(description="Run robustness analysis on EuroSAT RGB.")
    parser.add_argument("--config", type=str, default="configs/robustness.yaml",
                        help="Path to robustness config YAML file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    random_seed = cfg.get("random_seed", 42)
    np.random.seed(random_seed)

    results_dir = Path(cfg.get("results_dir", "./results"))
    logs_dir = results_dir / "logs"

    # ----------------------------------------------------------------
    # 1. Load split metadata
    # ----------------------------------------------------------------
    if not (logs_dir / "split_train.csv").exists():
        logger.error("Split metadata not found. Run run_ml.py or run_dl.py first.")
        sys.exit(1)

    splits, class_names = load_split_metadata(logs_dir)
    test_samples = splits["test"]
    logger.info(f"Test set: {len(test_samples)} samples")

    # ----------------------------------------------------------------
    # 2. Load trained ML models
    # ----------------------------------------------------------------
    models_dir = results_dir / "models"
    ml_models = {}

    feature_mode = cfg.get("feature_mode", "hog_color_texture")

    # Discover ML model files
    if models_dir.exists():
        for pkl_file in sorted(models_dir.glob("*.pkl")):
            model_name = pkl_file.stem
            # Only load models matching the feature mode
            if feature_mode in model_name:
                logger.info(f"Loading ML model: {model_name}")
                ml_models[model_name] = joblib.load(pkl_file)

    if not ml_models:
        logger.warning("No ML models found. ML robustness evaluation will be skipped.")

    # ----------------------------------------------------------------
    # 3. Load trained DL model
    # ----------------------------------------------------------------
    dl_model = None
    image_size_dl = tuple(cfg["dataset"].get("image_size_dl", [224, 224]))
    dl_transform = get_dl_transforms(image_size_dl, None, is_train=False)

    # Try to load the best DL checkpoint
    dl_checkpoint = models_dir / "ResNet18_aug_best.pt"
    if not dl_checkpoint.exists():
        dl_checkpoint = models_dir / "ResNet18_noaug_best.pt"

    dl_model_name = "ResNet18"
    if dl_checkpoint.exists():
        dl_model_name = dl_checkpoint.stem.replace("_best", "")
        logger.info(f"Loading DL model from {dl_checkpoint}")
        dl_model = build_model(num_classes=len(class_names), pretrained=False)
        dl_model.load_state_dict(
            torch.load(dl_checkpoint, map_location="cpu", weights_only=True)
        )
        dl_model.eval()
    else:
        logger.warning("No DL model checkpoint found. DL robustness evaluation will be skipped.")

    # ----------------------------------------------------------------
    # 4. Run robustness evaluation
    # ----------------------------------------------------------------
    # Feature extraction config (from ml.yaml defaults)
    ml_yaml_path = Path("configs/ml.yaml")
    hog_cfg = {}
    color_cfg = {}
    lbp_cfg = {}
    image_size_ml = tuple(cfg["dataset"].get("image_size_ml", [64, 64]))

    if ml_yaml_path.exists():
        with open(ml_yaml_path, "r") as f:
            ml_cfg = yaml.safe_load(f)
        feat_cfg = ml_cfg.get("features", {})
        hog_cfg = feat_cfg.get("hog", {})
        color_cfg = feat_cfg.get("color_histogram", {})
        lbp_cfg = feat_cfg.get("lbp", {})

    degradations_cfg = cfg.get("degradations", {})

    results_df = run_robustness_evaluation(
        ml_models=ml_models,
        dl_model=dl_model,
        dl_transform=dl_transform,
        test_samples=test_samples,
        class_names=class_names,
        degradations_cfg=degradations_cfg,
        image_size_ml=image_size_ml,
        feature_mode=feature_mode,
        hog_cfg=hog_cfg,
        color_cfg=color_cfg,
        lbp_cfg=lbp_cfg,
        results_dir=results_dir,
        dl_model_name=dl_model_name,
    )

    # ----------------------------------------------------------------
    # 5. Plot results
    # ----------------------------------------------------------------
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_robustness_comparison(results_df, plots_dir / "robustness_comparison.png")

    logger.info(f"\n{'=' * 60}")
    logger.info("Robustness Analysis Complete!")
    logger.info(f"{'=' * 60}")
    logger.info(f"\n{results_df.to_string(index=False)}")


if __name__ == "__main__":
    main()

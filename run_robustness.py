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
from src.dl.model import build_model, build_cnn
from src.robustness.evaluate import run_robustness_evaluation
from src.evaluation.plots import plot_robustness_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_robustness")


def _load_dl_checkpoint(checkpoint_path, num_classes, model_tag):
    """
    Load a DL model checkpoint, auto-detecting architecture from the tag name.

    Returns the loaded model or None on failure.
    """
    tag_lower = model_tag.lower()
    try:
        if "cnn" in tag_lower:
            model = build_cnn(num_classes=num_classes)
        else:
            model = build_model(num_classes=num_classes, pretrained=False)
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        model.eval()
        logger.info(f"Loaded DL model '{model_tag}' from {checkpoint_path}")
        return model
    except Exception as e:
        logger.warning(f"Could not load checkpoint {checkpoint_path}: {e}")
        return None


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
    #    Only load models whose filename ends with _{feature_mode}
    #    to guarantee the feature extractor matches exactly.
    # ----------------------------------------------------------------
    models_dir = results_dir / "models"
    ml_models = {}

    feature_mode = cfg.get("feature_mode", "hog_color_texture")

    if models_dir.exists():
        for pkl_file in sorted(models_dir.glob("*.pkl")):
            model_name = pkl_file.stem
            # Exact suffix match: model must have been trained with this feature mode
            if model_name.endswith(f"_{feature_mode}"):
                logger.info(f"Loading ML model: {model_name}")
                try:
                    ml_models[model_name] = joblib.load(pkl_file)
                except Exception as e:
                    logger.warning(f"Could not load {pkl_file}: {e}")

    if not ml_models:
        logger.warning(
            f"No ML models found with suffix '_{feature_mode}'. "
            f"Run run_ml.py with feature mode '{feature_mode}' first, "
            f"or change feature_mode in robustness.yaml."
        )

    # ----------------------------------------------------------------
    # 3. Load all trained DL model checkpoints (ResNet18 and/or CNN)
    # ----------------------------------------------------------------
    image_size_dl = tuple(cfg["dataset"].get("image_size_dl", [224, 224]))
    dl_transform = get_dl_transforms(image_size_dl, None, is_train=False)

    # Collect all *_best.pt files in models_dir
    dl_models = {}   # {model_tag: pytorch_model}
    if models_dir.exists():
        for pt_file in sorted(models_dir.glob("*_best.pt")):
            model_tag = pt_file.stem.replace("_best", "")
            model = _load_dl_checkpoint(pt_file, num_classes=len(class_names),
                                        model_tag=model_tag)
            if model is not None:
                dl_models[model_tag] = model

    if not dl_models:
        logger.warning("No DL checkpoints found. DL robustness evaluation will be skipped.")

    # ----------------------------------------------------------------
    # 4. Feature extraction config (read from ml.yaml for consistency)
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # 5. Run robustness evaluation for each DL model separately
    #    (ML results are identical across DL model runs so we merge)
    # ----------------------------------------------------------------
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if dl_models:
        all_results_list = []
        for dl_model_name, dl_model in dl_models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Robustness evaluation with DL model: {dl_model_name}")
            logger.info(f"{'='*60}")
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
            all_results_list.append(results_df)

        import pandas as pd
        # Merge: keep ML rows from first run only (they're the same), add all DL rows
        first_df = all_results_list[0]
        ml_rows = first_df[~first_df["model"].str.contains("ResNet|CNN", case=False, na=False)]
        dl_rows_list = [
            df[df["model"].str.contains("ResNet|CNN", case=False, na=False)]
            for df in all_results_list
        ]
        combined = pd.concat([ml_rows] + dl_rows_list, ignore_index=True)
    else:
        # Only ML models
        import pandas as pd
        combined = run_robustness_evaluation(
            ml_models=ml_models,
            dl_model=None,
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
        )

    # Save merged results
    out_path = results_dir / "metrics" / "robustness_results.csv"
    combined.to_csv(out_path, index=False)
    logger.info(f"Robustness results saved to {out_path}")

    # Plot
    plot_robustness_comparison(combined, plots_dir / "robustness_comparison.png")

    logger.info(f"\n{'=' * 60}")
    logger.info("Robustness Analysis Complete!")
    logger.info(f"{'=' * 60}")
    logger.info(f"\n{combined.to_string(index=False)}")


if __name__ == "__main__":
    main()

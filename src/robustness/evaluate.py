"""
Robustness evaluation: evaluate all trained models on degraded test sets.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.robustness.degradations import apply_degradation
from src.features.extractors import extract_features_single, extract_features_batch
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


def evaluate_ml_on_degraded(pipeline, test_samples, degradation_type, deg_params,
                            image_size=(64, 64), feature_mode="hog_color_texture",
                            hog_cfg=None, color_cfg=None, lbp_cfg=None):
    """
    Evaluate an ML model on a degraded version of the test set.

    Args:
        pipeline: Trained sklearn pipeline.
        test_samples: List of (image_path, label) tuples.
        degradation_type: Type of degradation to apply.
        deg_params: Dict of degradation parameters.
        image_size: Resize dimensions before feature extraction.
        feature_mode: Feature ablation mode.
        hog_cfg, color_cfg, lbp_cfg: Feature extraction configs.

    Returns:
        y_true, y_pred: Arrays of true and predicted labels.
    """
    features_list = []
    labels = []

    for img_path, label in tqdm(test_samples,
                                 desc=f"ML degraded eval ({degradation_type})", unit="img"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (image_size[1], image_size[0]))
            # Apply degradation
            img = apply_degradation(img, degradation_type, **deg_params)
            feat = extract_features_single(img, mode=feature_mode,
                                           hog_cfg=hog_cfg, color_cfg=color_cfg, lbp_cfg=lbp_cfg)
            features_list.append(feat)
            labels.append(label)
        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")

    X = np.array(features_list, dtype=np.float64)
    y_true = np.array(labels, dtype=np.int64)
    y_pred = pipeline.predict(X)
    return y_true, y_pred


@torch.no_grad()
def evaluate_dl_on_degraded(model, test_samples, degradation_type, deg_params,
                            transform, device=None):
    """
    Evaluate a DL model on a degraded version of the test set.

    Args:
        model: Trained PyTorch model.
        test_samples: List of (image_path, label) tuples.
        degradation_type: Type of degradation.
        deg_params: Dict of degradation parameters.
        transform: Torchvision transform to apply after degradation.
        device: Torch device.

    Returns:
        y_true, y_pred: Arrays of true and predicted labels.
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    for img_path, label in tqdm(test_samples,
                                 desc=f"DL eval ({degradation_type})", unit="img"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = apply_degradation(img, degradation_type, **deg_params)
            pil_img = Image.fromarray(img)
            tensor_img = transform(pil_img).unsqueeze(0).to(device)
            output = model(tensor_img)
            pred = output.argmax(dim=1).item()
            all_preds.append(pred)
            all_labels.append(label)
        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")

    return np.array(all_labels), np.array(all_preds)


def run_robustness_evaluation(ml_models, dl_model, dl_transform,
                              test_samples, class_names, degradations_cfg,
                              image_size_ml=(64, 64), feature_mode="hog_color_texture",
                              hog_cfg=None, color_cfg=None, lbp_cfg=None,
                              results_dir="./results", dl_model_name="ResNet18"):
    """
    Run the full robustness evaluation across all models and degradation conditions.

    Args:
        ml_models: Dict of {model_name: sklearn_pipeline}.
        dl_model: Trained PyTorch model (or None to skip).
        dl_transform: Torchvision transform for DL evaluation.
        test_samples: List of (image_path, label) tuples.
        class_names: List of class name strings.
        degradations_cfg: Dict from robustness.yaml with degradation types and levels.
        image_size_ml: For ML feature extraction.
        feature_mode: Feature ablation mode for ML.
        hog_cfg, color_cfg, lbp_cfg: Feature extraction configs.
        results_dir: Base results directory.
        dl_model_name: Name to use for the DL model in results (e.g. "ResNet18_aug").

    Returns:
        results_df: DataFrame with columns [model, condition, accuracy, macro_f1].
    """
    results_dir = Path(results_dir)
    records = []

    # ----------------------------------------------------------------
    # Clean test set evaluation
    # ----------------------------------------------------------------
    logger.info("Evaluating on CLEAN test set...")

    # Extract ML features once and reuse across all ML models
    if ml_models:
        X_test, y_test = extract_features_batch(
            test_samples, image_size=image_size_ml, mode=feature_mode,
            hog_cfg=hog_cfg, color_cfg=color_cfg, lbp_cfg=lbp_cfg,
        )
        for model_name, pipeline in ml_models.items():
            y_pred = pipeline.predict(X_test)
            metrics = compute_metrics(y_test, y_pred, class_names)
            records.append({
                "model": model_name, "condition": "clean",
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            })

    if dl_model is not None:
        y_true, y_pred = evaluate_dl_on_degraded(
            dl_model, test_samples, "none", {}, dl_transform,
        )
        if len(y_true) > 0:
            metrics = compute_metrics(y_true, y_pred, class_names)
            records.append({
                "model": dl_model_name, "condition": "clean",
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            })

    # ----------------------------------------------------------------
    # Degraded test set evaluation
    # ----------------------------------------------------------------
    all_degradation_items = []
    for deg_type, levels in degradations_cfg.items():
        for level_cfg in levels:
            label = level_cfg["label"]
            params = {k: v for k, v in level_cfg.items() if k != "label"}
            all_degradation_items.append((deg_type, label, params))

    for deg_type, label, params in all_degradation_items:
        logger.info(f"Evaluating on degraded test set: {label} ({deg_type}, params={params})")

        for model_name, pipeline in ml_models.items():
            y_true, y_pred = evaluate_ml_on_degraded(
                pipeline, test_samples, deg_type, params,
                image_size=image_size_ml, feature_mode=feature_mode,
                hog_cfg=hog_cfg, color_cfg=color_cfg, lbp_cfg=lbp_cfg,
            )
            metrics = compute_metrics(y_true, y_pred, class_names)
            records.append({
                "model": model_name, "condition": label,
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            })

        if dl_model is not None:
            y_true, y_pred = evaluate_dl_on_degraded(
                dl_model, test_samples, deg_type, params, dl_transform,
            )
            metrics = compute_metrics(y_true, y_pred, class_names)
            records.append({
                "model": dl_model_name, "condition": label,
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            })

    results_df = pd.DataFrame(records)

    out_path = results_dir / "metrics" / "robustness_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    logger.info(f"Robustness results saved to {out_path}")

    return results_df

"""
Handcrafted feature extractors for the classical ML pipeline.

Supports three feature types:
    - HOG (Histogram of Oriented Gradients)
    - Color Histogram (per-channel in HSV space)
    - LBP (Local Binary Pattern)

Feature ablation modes:
    - "hog"               : HOG only
    - "hog_color"         : HOG + Color Histogram
    - "hog_color_texture"  : HOG + Color Histogram + LBP
"""

import logging

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm

logger = logging.getLogger(__name__)

VALID_MODES = {"hog", "hog_color", "hog_color_texture"}


def extract_hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extract HOG features from a grayscale image.

    Args:
        img_gray: 2D numpy array (grayscale).
        orientations: Number of gradient orientations.
        pixels_per_cell: Size of a cell in pixels.
        cells_per_block: Number of cells per block.

    Returns:
        1D numpy array of HOG features.
    """
    features = hog(
        img_gray,
        orientations=orientations,
        pixels_per_cell=tuple(pixels_per_cell),
        cells_per_block=tuple(cells_per_block),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features


def extract_color_histogram(img_bgr, bins=32):
    """
    Extract color histogram features in HSV space.

    Computes per-channel histograms and concatenates them.

    Args:
        img_bgr: 3-channel BGR image (as loaded by OpenCV).
        bins: Number of histogram bins per channel.

    Returns:
        1D numpy array of histogram features.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    features = []
    # H channel: range [0, 180]
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    # S channel: range [0, 256]
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    # V channel: range [0, 256]
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()

    # Normalize each histogram
    for hist in [hist_h, hist_s, hist_v]:
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum
        features.append(hist)

    return np.concatenate(features)


def extract_lbp(img_gray, radius=3, n_points=24):
    """
    Extract Local Binary Pattern histogram features.

    Args:
        img_gray: 2D numpy array (grayscale).
        radius: Radius of the LBP circle.
        n_points: Number of surrounding points.

    Returns:
        1D numpy array of LBP histogram features.
    """
    lbp = local_binary_pattern(img_gray, n_points, radius, method="uniform")
    # Uniform LBP produces exactly n_points + 2 distinct patterns
    hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True)
    return hist.astype(np.float32)


def extract_features_single(img_bgr, mode="hog_color_texture", hog_cfg=None, color_cfg=None, lbp_cfg=None):
    """
    Extract features from a single BGR image based on the ablation mode.

    Args:
        img_bgr: BGR image (numpy array).
        mode: One of "hog", "hog_color", "hog_color_texture".
        hog_cfg: Dict with HOG parameters (orientations, pixels_per_cell, cells_per_block).
        color_cfg: Dict with color histogram parameters (bins).
        lbp_cfg: Dict with LBP parameters (radius, n_points).

    Returns:
        1D numpy array of concatenated features.
    """
    hog_cfg = hog_cfg or {}
    color_cfg = color_cfg or {}
    lbp_cfg = lbp_cfg or {}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    parts = []

    # Always extract HOG
    hog_feat = extract_hog(
        gray,
        orientations=hog_cfg.get("orientations", 9),
        pixels_per_cell=hog_cfg.get("pixels_per_cell", (8, 8)),
        cells_per_block=hog_cfg.get("cells_per_block", (2, 2)),
    )
    parts.append(hog_feat)

    # Add color histogram
    if mode in ("hog_color", "hog_color_texture"):
        color_feat = extract_color_histogram(
            img_bgr,
            bins=color_cfg.get("bins", 32),
        )
        parts.append(color_feat)

    # Add LBP texture
    if mode == "hog_color_texture":
        lbp_feat = extract_lbp(
            gray,
            radius=lbp_cfg.get("radius", 3),
            n_points=lbp_cfg.get("n_points", 24),
        )
        parts.append(lbp_feat)

    return np.concatenate(parts)


def extract_features_batch(samples, image_size=(64, 64), mode="hog_color_texture",
                           hog_cfg=None, color_cfg=None, lbp_cfg=None):
    """
    Extract features from a batch of image samples.

    Args:
        samples: List of (image_path, label) tuples.
        image_size: (H, W) to resize images to before feature extraction.
        mode: Feature ablation mode. One of "hog", "hog_color", "hog_color_texture".
        hog_cfg, color_cfg, lbp_cfg: Feature-specific config dicts.

    Returns:
        X: 2D numpy array of shape (N, D).
        y: 1D numpy array of labels.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown feature mode '{mode}'. Must be one of {VALID_MODES}.")

    features_list = []
    labels = []
    skipped = 0

    for img_path, label in tqdm(samples, desc=f"Extracting features ({mode})", unit="img"):
        try:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Cannot read image: {img_path}")
                skipped += 1
                continue
            img = cv2.resize(img, (image_size[1], image_size[0]))
            feat = extract_features_single(img, mode=mode,
                                           hog_cfg=hog_cfg, color_cfg=color_cfg, lbp_cfg=lbp_cfg)
            features_list.append(feat)
            labels.append(label)
        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")
            skipped += 1

    if skipped > 0:
        logger.warning(f"Skipped {skipped} images due to errors.")

    if not features_list:
        raise RuntimeError("No features could be extracted — all images failed to load.")

    # float32 halves RAM usage versus float64 and is sufficient for these features.
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    logger.info(f"Feature matrix shape: {X.shape}")
    return X, y

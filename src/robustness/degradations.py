"""
Image degradation functions for robustness analysis.

Supports:
    - Gaussian blur
    - Gaussian noise
    - Downsampling (with upsample back to original size)
"""

import cv2
import numpy as np


def apply_gaussian_blur(img, sigma=1.0):
    """
    Apply Gaussian blur to an image.

    Args:
        img: Input image (numpy array, uint8 or float).
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Blurred image (same dtype as input).
    """
    # Kernel size must be odd and large enough for the sigma
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(ksize, 3)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_gaussian_noise(img, std=0.05):
    """
    Add Gaussian noise to an image.

    Args:
        img: Input image (numpy array, uint8).
        std: Standard deviation of noise relative to [0, 1] range.
              e.g., std=0.05 means noise std is 0.05*255 ≈ 12.75 when applied to uint8.

    Returns:
        Noisy image (uint8).
    """
    img_float = img.astype(np.float64) / 255.0
    noise = np.random.normal(0, std, img_float.shape)
    noisy = np.clip(img_float + noise, 0, 1)
    return (noisy * 255).astype(np.uint8)


def apply_downsample(img, factor=2):
    """
    Downsample an image by a factor and then upsample back to original size.

    This simulates lower-resolution capture.

    Args:
        img: Input image (numpy array, uint8).
        factor: Downsampling factor (e.g., 2 means half resolution in each dimension).

    Returns:
        Degraded image at original resolution (uint8).
    """
    h, w = img.shape[:2]
    small_h = max(h // factor, 1)
    small_w = max(w // factor, 1)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored


def apply_degradation(img, degradation_type, **kwargs):
    """
    Apply a named degradation to an image.

    Args:
        img: Input image (numpy array).
        degradation_type: One of "gaussian_blur", "gaussian_noise", "downsample".
        **kwargs: Parameters for the degradation function.

    Returns:
        Degraded image.
    """
    if degradation_type == "gaussian_blur":
        return apply_gaussian_blur(img, sigma=kwargs.get("sigma", 1.0))
    elif degradation_type == "gaussian_noise":
        return apply_gaussian_noise(img, std=kwargs.get("std", 0.05))
    elif degradation_type == "downsample":
        return apply_downsample(img, factor=kwargs.get("factor", 2))
    else:
        raise ValueError(f"Unknown degradation type: {degradation_type}")

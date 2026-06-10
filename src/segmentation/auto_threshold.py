"""
Automated ECT image segmentation replacing manual thresholding.

The thesis used a manually tuned threshold of 110 on normalized [0,255]
grayscale images derived from the real part of ECT impedance data.
This module provides automatic alternatives: Otsu's method, adaptive
thresholding, and a multi-method ensemble approach.
"""

import numpy as np


def otsu_threshold(image: np.ndarray) -> tuple[int, np.ndarray]:
    """Compute the optimal global threshold using Otsu's method.

    Maximises the inter-class variance between foreground (solid) and
    background (void) pixel populations, removing the need for manual
    threshold tuning.

    Args:
        image: 2D array of grayscale values in [0, 255] (uint8 or float).

    Returns:
        (threshold, binary_image) where binary_image has 1 = void, 0 = solid,
        matching the convention used in the thesis.
    """
    hist = np.zeros(256, dtype=np.float64)
    img_flat = np.clip(np.round(image), 0, 255).astype(int).ravel()
    for val in img_flat:
        hist[val] += 1

    total = img_flat.size
    hist_norm = hist / total

    best_thresh = 0
    best_variance = 0.0

    cum_sum_0 = 0.0
    cum_weight_0 = 0.0

    total_mean = np.sum(np.arange(256) * hist_norm)

    for t in range(256):
        cum_weight_0 += hist_norm[t]
        if cum_weight_0 == 0:
            continue
        cum_weight_1 = 1.0 - cum_weight_0
        if cum_weight_1 == 0:
            break

        cum_sum_0 += t * hist_norm[t]
        mean_0 = cum_sum_0 / cum_weight_0
        mean_1 = (total_mean - cum_sum_0) / cum_weight_1

        variance = cum_weight_0 * cum_weight_1 * (mean_0 - mean_1) ** 2

        if variance > best_variance:
            best_variance = variance
            best_thresh = t

    binary = (image < best_thresh).astype(np.uint8)
    return best_thresh, binary


def adaptive_threshold(
    image: np.ndarray,
    block_size: int = 51,
    offset: float = 5.0,
) -> np.ndarray:
    """Local adaptive thresholding using a block-wise mean.

    For each pixel, the threshold is the mean intensity within a local
    neighbourhood minus an offset.  This handles non-uniform illumination
    or sensor sensitivity variations across the ECT scan.

    Args:
        image: 2D grayscale array in [0, 255].
        block_size: Side length of the local neighbourhood (must be odd).
        offset: Constant subtracted from the local mean to form the threshold.

    Returns:
        Binary image (1 = void, 0 = solid).
    """
    if block_size % 2 == 0:
        block_size += 1

    rows, cols = image.shape
    binary = np.zeros_like(image, dtype=np.uint8)
    pad = block_size // 2

    padded = np.pad(image.astype(np.float64), pad, mode="reflect")

    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)

    for i in range(rows):
        for j in range(cols):
            r1, c1 = i, j
            r2, c2 = i + block_size - 1, j + block_size - 1

            area = (r2 - r1 + 1) * (c2 - c1 + 1)
            s = integral[r2, c2]
            if r1 > 0:
                s -= integral[r1 - 1, c2]
            if c1 > 0:
                s -= integral[r2, c1 - 1]
            if r1 > 0 and c1 > 0:
                s += integral[r1 - 1, c1 - 1]

            local_mean = s / area
            binary[i, j] = 1 if image[i, j] < (local_mean - offset) else 0

    return binary


def adaptive_threshold_fast(
    image: np.ndarray,
    block_size: int = 51,
    offset: float = 5.0,
) -> np.ndarray:
    """Vectorised adaptive thresholding using integral images."""
    if block_size % 2 == 0:
        block_size += 1

    img = image.astype(np.float64)
    rows, cols = img.shape
    pad = block_size // 2

    padded = np.pad(img, pad, mode="reflect")
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)

    r1 = np.arange(rows)
    c1 = np.arange(cols)
    R1, C1 = np.meshgrid(r1, c1, indexing="ij")
    R2 = R1 + block_size - 1
    C2 = C1 + block_size - 1

    area = block_size * block_size
    s = integral[R2, C2].copy()
    s -= np.where(R1 > 0, integral[R1 - 1, C2], 0)
    s -= np.where(C1 > 0, integral[R2, C1 - 1], 0)
    s += np.where((R1 > 0) & (C1 > 0), integral[R1 - 1, C1 - 1], 0)

    local_mean = s / area
    binary = (img < (local_mean - offset)).astype(np.uint8)
    return binary


def multi_method_segment(
    image: np.ndarray,
    block_size: int = 51,
    offset: float = 5.0,
    agreement: int = 2,
) -> tuple[dict, np.ndarray]:
    """Ensemble segmentation combining Otsu and adaptive thresholding.

    Pixels are classified as void only if at least `agreement` out of the
    three methods (Otsu, adaptive, and the median of Otsu+adaptive thresholds)
    agree.

    Args:
        image: 2D grayscale array [0, 255].
        block_size: Block size for adaptive method.
        offset: Offset for adaptive method.
        agreement: Minimum number of methods that must classify a pixel as void.

    Returns:
        (info_dict, consensus_binary) where info_dict contains per-method
        results and the Otsu threshold.
    """
    otsu_val, binary_otsu = otsu_threshold(image)
    binary_adaptive = adaptive_threshold_fast(image, block_size, offset)

    median_thresh = int(np.round((otsu_val + (image.mean() - offset)) / 2))
    binary_median = (image < median_thresh).astype(np.uint8)

    votes = binary_otsu.astype(int) + binary_adaptive.astype(int) + binary_median.astype(int)
    consensus = (votes >= agreement).astype(np.uint8)

    info = {
        "otsu_threshold": otsu_val,
        "median_threshold": median_thresh,
        "binary_otsu": binary_otsu,
        "binary_adaptive": binary_adaptive,
        "binary_median": binary_median,
    }
    return info, consensus

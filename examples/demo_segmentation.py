"""
Demo: Automated ECT Image Segmentation (Item 1)

Generates a synthetic ECT-like image resembling the thesis's black box
cross-sections (with voids), then compares manual threshold (110) against
Otsu, adaptive, and ensemble methods.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.segmentation import otsu_threshold, adaptive_threshold, multi_method_segment
from src.segmentation import clean_binary_image
from src.segmentation.auto_threshold import adaptive_threshold_fast


def create_synthetic_ect_image(rows=430, cols=300, noise_level=15):
    """Create a synthetic image mimicking ECT real-part data of a beam with voids."""
    image = np.ones((rows, cols)) * 160  # Solid material = high value

    # Add rectangular voids (low intensity)
    image[100:200, 80:120] = 40
    image[100:200, 180:220] = 40
    image[250:350, 50:100] = 40
    image[250:350, 200:250] = 40

    # Add a thin wall (challenging for ECT)
    image[150:300, 148:152] = 120  # Barely above typical threshold

    # Add gradient (simulating non-uniform sensor response)
    gradient = np.linspace(0, 20, cols)[np.newaxis, :]
    image += gradient

    # Add noise
    noise = np.random.normal(0, noise_level, image.shape)
    image = np.clip(image + noise, 0, 255)

    return image


def main():
    print("=" * 60)
    print("Demo: Automated ECT Image Segmentation")
    print("=" * 60)

    np.random.seed(42)
    image = create_synthetic_ect_image()

    # --- Method 1: Manual threshold (thesis approach) ---
    manual_thresh = 110
    binary_manual = (image < manual_thresh).astype(np.uint8)

    # --- Method 2: Otsu's method ---
    otsu_val, binary_otsu = otsu_threshold(image)
    print(f"Manual threshold: {manual_thresh}")
    print(f"Otsu threshold:   {otsu_val}")

    # --- Method 3: Adaptive threshold ---
    binary_adaptive = adaptive_threshold_fast(image, block_size=51, offset=5.0)

    # --- Method 4: Multi-method ensemble ---
    info, binary_ensemble = multi_method_segment(image)
    print(f"Median threshold: {info['median_threshold']}")

    # --- Morphological cleanup ---
    binary_cleaned = clean_binary_image(binary_ensemble, min_component_size=50)

    # --- Metrics: compare against ground truth ---
    gt = create_ground_truth(image.shape)
    for name, binary in [
        ("Manual (110)", binary_manual),
        ("Otsu", binary_otsu),
        ("Adaptive", binary_adaptive),
        ("Ensemble", binary_ensemble),
        ("Ensemble+Clean", binary_cleaned),
    ]:
        tp = np.sum((binary == 1) & (gt == 1))
        fp = np.sum((binary == 1) & (gt == 0))
        fn = np.sum((binary == 0) & (gt == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {name:20s}  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    # --- Visualisation ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("ECT Image Segmentation Comparison", fontsize=14)

    panels = [
        (axes[0, 0], image, "Original (grayscale)", "gray"),
        (axes[0, 1], binary_manual, f"Manual (thresh={manual_thresh})", "gray_r"),
        (axes[0, 2], binary_otsu, f"Otsu (thresh={otsu_val})", "gray_r"),
        (axes[1, 0], binary_adaptive, "Adaptive", "gray_r"),
        (axes[1, 1], binary_ensemble, "Ensemble (2/3 vote)", "gray_r"),
        (axes[1, 2], binary_cleaned, "Ensemble + Morphology", "gray_r"),
    ]

    for ax, data, title, cmap in panels:
        ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("x points")
        ax.set_ylabel("z layers")

    plt.tight_layout()
    plt.savefig("segmentation_comparison.png", dpi=150)
    plt.show()
    print("\nSaved: segmentation_comparison.png")


def create_ground_truth(shape):
    """Ground truth binary image matching the synthetic voids."""
    gt = np.zeros(shape, dtype=np.uint8)
    gt[100:200, 80:120] = 1
    gt[100:200, 180:220] = 1
    gt[250:350, 50:100] = 1
    gt[250:350, 200:250] = 1
    return gt


if __name__ == "__main__":
    main()

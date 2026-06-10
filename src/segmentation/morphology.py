"""Morphological operations for cleaning binary ECT segmentation results."""

import numpy as np


def _binary_dilate(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode="constant", constant_values=0)
    result = np.zeros_like(image)
    for di in range(kernel_size):
        for dj in range(kernel_size):
            result |= padded[di : di + image.shape[0], dj : dj + image.shape[1]]
    return result


def _binary_erode(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode="constant", constant_values=0)
    result = np.ones_like(image)
    for di in range(kernel_size):
        for dj in range(kernel_size):
            result &= padded[di : di + image.shape[0], dj : dj + image.shape[1]]
    return result


def _connected_components(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """Simple flood-fill connected-component labelling (4-connectivity)."""
    labels = np.zeros_like(binary, dtype=int)
    current_label = 0
    rows, cols = binary.shape

    for i in range(rows):
        for j in range(cols):
            if binary[i, j] == 1 and labels[i, j] == 0:
                current_label += 1
                stack = [(i, j)]
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= rows or c < 0 or c >= cols:
                        continue
                    if binary[r, c] != 1 or labels[r, c] != 0:
                        continue
                    labels[r, c] = current_label
                    stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])

    return labels, current_label


def clean_binary_image(
    binary: np.ndarray,
    close_kernel: int = 3,
    open_kernel: int = 3,
    min_component_size: int = 50,
) -> np.ndarray:
    """Clean a binary segmentation result.

    Applies morphological closing (fill small holes), opening (remove
    small noise), and removes connected components smaller than a minimum
    size.

    Args:
        binary: Input binary image (1 = void, 0 = solid).
        close_kernel: Kernel size for closing operation.
        open_kernel: Kernel size for opening operation.
        min_component_size: Components with fewer pixels are removed.

    Returns:
        Cleaned binary image.
    """
    # Closing: dilate then erode — fills small gaps in void regions
    closed = _binary_erode(_binary_dilate(binary, close_kernel), close_kernel)

    # Opening: erode then dilate — removes small noise specks
    opened = _binary_dilate(_binary_erode(closed, open_kernel), open_kernel)

    # Remove small connected components
    labels, n_labels = _connected_components(opened)
    result = np.zeros_like(opened)
    for label_id in range(1, n_labels + 1):
        component = labels == label_id
        if np.sum(component) >= min_component_size:
            result[component] = 1

    return result

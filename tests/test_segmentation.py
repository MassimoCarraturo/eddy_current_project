"""Tests for the automated ECT segmentation module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.segmentation.auto_threshold import (
    otsu_threshold,
    adaptive_threshold_fast,
    multi_method_segment,
)
from src.segmentation.morphology import clean_binary_image


class TestOtsuThreshold:
    def test_bimodal_image(self):
        """Otsu should find the valley between two clear peaks."""
        image = np.zeros((100, 100))
        image[:50, :] = 200  # Bright region
        image[50:, :] = 50   # Dark region
        noise = np.random.normal(0, 5, image.shape)
        image = np.clip(image + noise, 0, 255)

        thresh, binary = otsu_threshold(image)
        assert 55 < thresh < 195, f"Threshold {thresh} not between peaks 50 and 200"
        assert binary.shape == image.shape
        assert np.mean(binary[:50, :]) < 0.1  # Bright = solid = 0
        assert np.mean(binary[50:, :]) > 0.9   # Dark = void = 1

    def test_uniform_image(self):
        """Otsu on a uniform image should still return valid binary output."""
        image = np.ones((50, 50)) * 128
        thresh, binary = otsu_threshold(image)
        assert binary.shape == (50, 50)

    def test_output_is_binary(self):
        image = np.random.rand(30, 30) * 255
        _, binary = otsu_threshold(image)
        assert set(np.unique(binary)).issubset({0, 1})


class TestAdaptiveThreshold:
    def test_gradient_image(self):
        """Adaptive threshold should handle non-uniform intensity."""
        rows, cols = 100, 100
        gradient = np.tile(np.linspace(50, 200, cols), (rows, 1))
        # Add void in center
        gradient[30:70, 30:70] -= 80
        gradient = np.clip(gradient, 0, 255)

        binary = adaptive_threshold_fast(gradient, block_size=31, offset=10)
        assert binary.shape == gradient.shape
        # Center should be detected as void
        center_void_rate = np.mean(binary[35:65, 35:65])
        assert center_void_rate > 0.5


class TestMultiMethodSegment:
    def test_consensus(self):
        image = np.zeros((100, 100))
        image[:, :] = 180
        image[20:40, 20:40] = 30  # Clear void
        noise = np.random.normal(0, 3, image.shape)
        image = np.clip(image + noise, 0, 255)

        info, consensus = multi_method_segment(image)
        assert "otsu_threshold" in info
        # The clear void should be detected
        void_rate = np.mean(consensus[22:38, 22:38])
        assert void_rate > 0.8


class TestMorphology:
    def test_removes_small_components(self):
        binary = np.zeros((50, 50), dtype=np.uint8)
        binary[10:12, 10:12] = 1  # Small component (4 pixels)
        binary[30:45, 30:45] = 1  # Large component (225 pixels)

        cleaned = clean_binary_image(binary, min_component_size=10)
        assert np.sum(cleaned[10:12, 10:12]) == 0  # Small removed
        assert np.sum(cleaned[30:45, 30:45]) > 0    # Large kept

    def test_preserves_large_regions(self):
        binary = np.zeros((100, 100), dtype=np.uint8)
        binary[20:80, 20:80] = 1

        cleaned = clean_binary_image(binary, min_component_size=50)
        assert np.sum(cleaned) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

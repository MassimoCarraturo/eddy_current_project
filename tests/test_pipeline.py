"""Tests for the unified pipeline."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.pipeline import UnifiedECTPipeline
from src.utils.common import ECTCoilParams, MaterialParams


@pytest.fixture
def simple_pipeline():
    coil = ECTCoilParams(
        r_inner=0.00107 / 2,
        r_outer=0.00262 / 2,
        length=0.00293,
        n_turns=235,
        liftoff=0.00056,
        frequency=240e3,
    )
    material = MaterialParams(sigma=17.7e6)
    return UnifiedECTPipeline(coil=coil, material=material, kappa=-0.326)


class TestGeometryReconstruction:
    def test_basic_reconstruction(self, simple_pipeline):
        """Pipeline should produce a 3D binary volume from layer data."""
        n_layers = 10
        n_x = 50
        for i in range(n_layers):
            real = np.ones(n_x) * 0.5
            imag = np.ones(n_x) * (-0.3)
            # Add void in middle layers
            if 3 <= i <= 6:
                real[20:30] = 0.1
                imag[20:30] = -0.05
            simple_pipeline.add_layer(i, real, imag)

        recon = simple_pipeline.run_geometry_reconstruction()
        assert recon.binary_volume.ndim == 3
        assert recon.binary_volume.shape[0] == n_layers
        assert "x_mm" in recon.spacing

    def test_empty_pipeline_raises(self, simple_pipeline):
        with pytest.raises(ValueError):
            simple_pipeline.run()


class TestDICOMMetadata:
    def test_metadata_fields(self, simple_pipeline):
        for i in range(5):
            simple_pipeline.add_layer(i, np.ones(50) * 0.5, np.ones(50) * -0.3)

        result = simple_pipeline.run(run_strain=False)
        meta = simple_pipeline.export_dicom_metadata(result)
        assert "StudyInstanceUID" in meta
        assert "PixelSpacing" in meta
        assert "Modality" in meta
        assert meta["Modality"] == "CT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

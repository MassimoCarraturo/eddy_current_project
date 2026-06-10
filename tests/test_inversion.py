"""Tests for the Dodd-Deeds model and optimisation-based inversion."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.inversion.dodd_deeds import DoddDeedsModel
from src.inversion.optimizer import ImpedanceInverter
from src.utils.common import ECTCoilParams, MaterialParams


@pytest.fixture
def thesis_coil():
    """Coil from thesis whole_pipeline.py."""
    return ECTCoilParams(
        r_inner=0.00107 / 2,
        r_outer=0.00262 / 2,
        length=0.00293,
        n_turns=235,
        liftoff=0.00056,
        frequency=240e3,
    )


@pytest.fixture
def steel_316l():
    return MaterialParams(sigma=17.7e6)


class TestDoddDeedsModel:
    def test_z0_is_purely_imaginary_dominant(self, thesis_coil):
        """Self-impedance of an air coil should be dominated by imaginary part."""
        model = DoddDeedsModel(thesis_coil)
        z0 = model.z0()
        assert abs(z0.imag) > abs(z0.real)

    def test_z_total_differs_from_z0(self, thesis_coil, steel_316l):
        """Placing a conductor below the coil must change the impedance."""
        model = DoddDeedsModel(thesis_coil)
        z0 = model.z0()
        z_total = model.z_total(steel_316l)
        assert abs(z0 - z_total) > 0

    def test_normalized_impedance_range(self, thesis_coil, steel_316l):
        """Normalised impedance components should be in a reasonable range."""
        model = DoddDeedsModel(thesis_coil)
        z_norm = model.z_normalized(steel_316l)
        assert -2 < z_norm.real < 2
        assert -2 < z_norm.imag < 2

    def test_impedance_monotonic_with_sigma(self, thesis_coil):
        """Increasing conductivity should monotonically change impedance."""
        model = DoddDeedsModel(thesis_coil)
        sigmas = np.array([1e6, 5e6, 10e6, 20e6])
        z_vals = model.impedance_vs_sigma(sigmas)
        # The imaginary part should change monotonically
        diffs = np.diff(z_vals.imag)
        assert np.all(diffs > 0) or np.all(diffs < 0)


class TestImpedanceInverter:
    def test_round_trip_single(self, thesis_coil, steel_316l):
        """Forward then inverse should recover the original conductivity."""
        model = DoddDeedsModel(thesis_coil)
        z_meas = model.z_normalized(steel_316l)

        inverter = ImpedanceInverter(
            thesis_coil,
            sigma_bounds=(1e6, 50e6),
        )
        sigma_inv = inverter.invert_single(z_meas)
        rel_err = abs(sigma_inv - steel_316l.sigma) / steel_316l.sigma
        assert rel_err < 0.01, f"Round-trip error {rel_err:.4f} > 1%"

    def test_invert_to_strain(self, thesis_coil):
        """Full inversion pipeline should recover strain from impedance."""
        sigma_0 = 17.7e6
        kappa = -0.326
        strain_true = 0.01  # 1%

        sigma_strained = sigma_0 * (1 + kappa * strain_true)
        model = DoddDeedsModel(thesis_coil)
        z_meas = model.z_normalized(MaterialParams(sigma=sigma_strained))

        inverter = ImpedanceInverter(
            thesis_coil,
            sigma_bounds=(0.5 * sigma_0, 2 * sigma_0),
        )
        sigma_arr, strain_arr = inverter.invert_to_strain(
            np.array([z_meas]), sigma_0, kappa
        )
        assert abs(strain_arr[0] - strain_true) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

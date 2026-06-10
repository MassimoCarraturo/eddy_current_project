"""Tests for regularised tensor-strain inversion -- Step 4."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.tensor_model import (ElastoResistivityModel, direction,
                              optimize_probe_set, TensorStrainInverter,
                              volumetric_strain, to_voigt)


@pytest.fixture
def model():
    return ElastoResistivityModel(kappa_long=-0.40, kappa_trans=-0.08)


@pytest.fixture
def strain():
    return np.array([[0.010, 0.003, 0.001],
                     [0.003, -0.004, -0.0005],
                     [0.001, -0.0005, 0.006]])


class TestFullConfig:
    def test_noiseless_recovery(self, model, strain):
        dirs = optimize_probe_set(model, n_probes=6, n_restarts=8)["directions"]
        inv = TensorStrainInverter(model, dirs, prior_weight=1e-8)
        y = inv.forward(strain)
        rec = inv.invert(y, prior_strain=np.zeros((3, 3)), as_tensor=True)
        assert np.allclose(rec, strain, atol=1e-5)

    def test_all_components_data_resolved(self, model):
        dirs = optimize_probe_set(model, n_probes=6, n_restarts=8)["directions"]
        inv = TensorStrainInverter(model, dirs, prior_weight=1e-6)
        assert np.all(inv.data_resolved_fraction() > 0.99)


class TestRosetteWithPrior:
    def test_inplane_resolved_outofplane_from_prior(self, model):
        dirs = [direction(a) for a in (0, 60, 120)]
        inv = TensorStrainInverter(model, dirs, prior_weight=1e-3)
        frac = inv.data_resolved_fraction()
        # in-plane xx, yy, xy resolved; out-of-plane yz, xz from prior
        assert frac[0] > 0.8 and frac[1] > 0.8 and frac[5] > 0.8
        assert frac[3] < 0.2 and frac[4] < 0.2

    def test_prior_fills_unobserved(self, model, strain):
        dirs = [direction(a) for a in (0, 60, 120)]
        inv = TensorStrainInverter(model, dirs, prior_weight=1e-3)
        prior = np.zeros((3, 3))
        prior[0, 2] = prior[2, 0] = 0.0123  # an xz value only the prior knows
        y = inv.forward(strain)
        rec = inv.invert(y, prior_strain=prior, as_tensor=True)
        # recovered xz should track the prior (unobserved by in-plane probes)
        assert abs(rec[0, 2] - 0.0123) < 1e-4

    def test_data_overrides_prior_bias_inplane(self, model, strain):
        dirs = [direction(a) for a in (0, 60, 120)]
        inv = TensorStrainInverter(model, dirs, prior_weight=1e-4)
        prior = strain + 0.005  # biased prior
        y = inv.forward(strain)
        rec = inv.invert(y, prior_strain=prior, as_tensor=True)
        # xx is observed -> recovered closer to truth than the biased prior
        assert abs(rec[0, 0] - strain[0, 0]) < abs(prior[0, 0] - strain[0, 0])


class TestFieldAndFallback:
    def test_invert_field_matches_pointwise(self, model, strain):
        dirs = optimize_probe_set(model, n_probes=6, n_restarts=6)["directions"]
        inv = TensorStrainInverter(model, dirs, prior_weight=1e-6)
        Y = np.array([inv.forward(strain), inv.forward(2 * strain)])
        field = inv.invert_field(Y)
        single = inv.invert(Y[0])
        assert np.allclose(field[0], single)

    def test_volumetric_fallback(self, model):
        eps_vol = 0.012
        eps_iso = (eps_vol / 3.0) * np.eye(3)
        dsig = model.delta_sigma_ratio(eps_iso)
        scalar = 0.5 * (dsig[0, 0] + dsig[1, 1])  # normal-coil reading
        assert np.isclose(volumetric_strain(scalar, model), eps_vol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

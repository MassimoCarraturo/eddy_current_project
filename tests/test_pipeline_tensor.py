"""Tests for the tensor strain branch of the unified pipeline."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.pipeline import UnifiedECTPipeline, TensorStrainResult
from src.utils.common import ECTCoilParams, MaterialParams
from src.tensor_model import (ElastoResistivityModel, direction,
                              optimize_probe_set, TensorStrainInverter, to_voigt)


@pytest.fixture
def coil():
    return ECTCoilParams(r_inner=0.535e-3, r_outer=1.31e-3, length=2.93e-3,
                         n_turns=235, liftoff=0.56e-3, frequency=240e3)


@pytest.fixture
def model():
    return ElastoResistivityModel(kappa_long=-0.40, kappa_trans=-0.08)


def _pipeline(coil):
    return UnifiedECTPipeline(coil, MaterialParams(sigma=17.7e6))


class TestTensorPipeline:
    def test_full_config_recovers_tensor_field(self, coil, model):
        dirs = optimize_probe_set(model, n_probes=6, n_restarts=6)["directions"]
        fwd = TensorStrainInverter(model, dirs, prior_weight=1e-8)

        pipe = _pipeline(coil)
        truth = {}
        for li in range(5):
            eps = np.zeros((3, 3))
            eps[0, 0] = 0.002 + 0.001 * li
            eps[2, 2] = 0.001 * li
            eps[0, 1] = eps[1, 0] = 0.0005 * li
            truth[li] = eps
            pipe.add_directional_layer(li, fwd.forward(eps))

        res = pipe.run_tensor_strain_prediction(model, dirs, prior_weight=1e-8)
        assert isinstance(res, TensorStrainResult)
        assert res.strain_voigt.shape == (5, 1, 6)
        # full config -> recovers each layer's tensor
        for li in range(5):
            assert np.allclose(res.strain_tensor(li), truth[li], atol=1e-4)

    def test_resolution_reported(self, coil, model):
        dirs = [direction(a) for a in (0, 60, 120)]
        fwd = TensorStrainInverter(model, dirs, prior_weight=1e-3)
        pipe = _pipeline(coil)
        pipe.add_directional_layer(0, fwd.forward(np.eye(3) * 0.001))
        res = pipe.run_tensor_strain_prediction(model, dirs, prior_weight=1e-3)
        # in-plane rosette: out-of-plane shears come from the prior
        assert res.resolution[3] < 0.2 and res.resolution[4] < 0.2
        assert res.resolution[0] > 0.8

    def test_spatially_resolved(self, coil, model):
        dirs = optimize_probe_set(model, n_probes=6, n_restarts=5)["directions"]
        fwd = TensorStrainInverter(model, dirs, prior_weight=1e-8)
        pipe = _pipeline(coil)
        # 4 x-points per layer
        meas = np.column_stack([fwd.forward(np.eye(3) * (0.001 * k)) for k in range(4)])
        pipe.add_directional_layer(0, meas)
        res = pipe.run_tensor_strain_prediction(model, dirs, prior_weight=1e-8)
        assert res.strain_voigt.shape == (1, 4, 6)

    def test_prior_fills_unobserved_in_pipeline(self, coil, model):
        dirs = [direction(a) for a in (0, 60, 120)]
        fwd = TensorStrainInverter(model, dirs, prior_weight=1e-3)
        pipe = _pipeline(coil)
        eps = np.zeros((3, 3))
        eps[0, 0] = 0.003
        pipe.add_directional_layer(0, fwd.forward(eps))
        prior = np.zeros((1, 6))
        prior[0, 4] = 0.0099  # xz known only to the prior
        res = pipe.run_tensor_strain_prediction(model, dirs, prior_field=prior,
                                                prior_weight=1e-3)
        assert abs(res.strain_voigt[0, 0, 4] - 0.0099) < 1e-3

    def test_requires_directional_data(self, coil, model):
        pipe = _pipeline(coil)
        with pytest.raises(ValueError):
            pipe.run_tensor_strain_prediction(model, [direction(0)])

    def test_equivalent_strain_nonnegative(self, coil, model):
        dirs = optimize_probe_set(model, n_probes=6, n_restarts=5)["directions"]
        fwd = TensorStrainInverter(model, dirs, prior_weight=1e-8)
        pipe = _pipeline(coil)
        pipe.add_directional_layer(0, fwd.forward(np.diag([0.01, -0.005, 0.003])))
        res = pipe.run_tensor_strain_prediction(model, dirs, prior_weight=1e-8)
        assert np.all(res.equivalent_strain() >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

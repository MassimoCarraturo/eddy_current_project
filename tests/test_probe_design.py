"""Tests for probe-configuration (rosette) design -- Step 3."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.tensor_model import (ElastoResistivityModel, direction,
                              design_metrics, evaluate_configuration,
                              max_achievable_rank, optimize_inplane_rosette,
                              optimize_probe_set)


@pytest.fixture
def model():
    return ElastoResistivityModel(kappa_long=-0.40, kappa_trans=-0.08)


class TestDesignMetrics:
    def test_full_rank_identity(self):
        m = design_metrics(np.eye(6))
        assert m["rank"] == 6
        assert np.isclose(m["condition_number"], 1.0)

    def test_rank_deficient(self):
        A = np.ones((3, 6))  # rank 1
        assert design_metrics(A)["rank"] == 1


class TestAchievableRank:
    def test_inplane_caps_at_three(self, model):
        assert max_achievable_rank(model, allow_tilt=False) == 3

    def test_tilt_reaches_six(self, model):
        assert max_achievable_rank(model, allow_tilt=True) == 6

    def test_scalar_model_caps_at_one(self):
        scalar = ElastoResistivityModel.from_scalar_kappa(-0.326)
        assert max_achievable_rank(scalar, allow_tilt=True) == 1


class TestOptimisation:
    def test_inplane_optimum_beats_naive(self, model):
        naive = evaluate_configuration(
            [direction(a) for a in (0, 45, 90)], model)["condition_number"]
        opt = optimize_inplane_rosette(model, n_probes=3, n_restarts=12)
        assert opt["rank"] == 3
        assert opt["condition_number"] <= naive + 1e-6

    def test_full_set_resolves_all_six(self, model):
        opt = optimize_probe_set(model, n_probes=6, n_restarts=10)
        assert opt["rank"] == 6
        assert np.isfinite(opt["condition_number"])

    def test_evenly_spaced_rosette_well_conditioned(self, model):
        """A delta rosette (0/60/120) should be reasonably conditioned."""
        m = evaluate_configuration([direction(a) for a in (0, 60, 120)], model)
        assert m["rank"] == 3
        assert m["condition_number"] < 5.0


class TestElastoResistivityRatio:
    """Observability vs the kappa_perp/kappa_parallel ratio (paper Sec. 5)."""

    def _full_set(self):
        return ([direction(a) for a in (0, 60, 120)]
                + [direction(a, 45) for a in (0, 60, 120)])

    def test_volumetric_limit_collapses_rank(self):
        """kappa_perp == kappa_parallel -> K is rank 1 -> only volumetric DOF."""
        model = ElastoResistivityModel(kappa_long=-0.374, kappa_trans=-0.374)
        assert evaluate_configuration(self._full_set(), model)["rank"] == 1

    def test_conditioning_degrades_toward_volumetric(self):
        """Condition number grows as kappa_perp -> kappa_parallel."""
        full = self._full_set()
        c_lo = evaluate_configuration(
            full, ElastoResistivityModel(-0.374, -0.05))["condition_number"]
        c_hi = evaluate_configuration(
            full, ElastoResistivityModel(-0.374, -0.34))["condition_number"]
        assert c_hi > c_lo

    def test_anchored_kappa_reproduces_thesis_scalar(self):
        """kappa_par = -0.374, kappa_perp = -0.08 -> along-load projection -0.326."""
        model = ElastoResistivityModel(kappa_long=-0.374, kappa_trans=-0.08)
        keff = model.effective_scalar_kappa([1, 0, 0], [1, 0, 0], 0.30)
        assert abs(keff - (-0.326)) < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

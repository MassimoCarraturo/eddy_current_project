"""Tests for the strain-tensor observability analysis."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.tensor_model import (ObservabilityAnalysis, projection_row, direction,
                              NORMAL_COIL_ROW, to_voigt, ElastoResistivityModel)


class TestProjection:
    def test_matches_quadratic_form(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            n = rng.normal(size=3)
            S = rng.normal(size=(3, 3))
            S = S + S.T
            nn = n / np.linalg.norm(n)
            assert np.isclose(projection_row(n) @ to_voigt(S), nn @ S @ nn)

    def test_direction_in_plane(self):
        d = direction(0.0)  # tilt defaults to 90 deg -> in-plane
        assert np.isclose(d[2], 0.0)
        assert np.allclose(d, [1, 0, 0])

    def test_direction_axial(self):
        d = direction(0.0, tilt_deg=0.0)  # along +z
        assert np.allclose(d, [0, 0, 1])


class TestObservability:
    def test_normal_coil_rank_one(self):
        oa = ObservabilityAnalysis([NORMAL_COIL_ROW])
        assert oa.analyse()["rank"] == 1

    def test_inplane_rosette_rank_three(self):
        oa = ObservabilityAnalysis.from_directions(
            [direction(0), direction(45), direction(90)])
        assert oa.analyse()["rank"] == 3

    def test_inplane_rosette_blind_to_out_of_plane(self):
        """In-plane probes (n_z = 0) cannot see zz, yz, xz conductivity."""
        oa = ObservabilityAnalysis.from_directions(
            [direction(0), direction(45), direction(90)])
        # columns for zz(2), yz(3), xz(4) must be identically zero
        assert np.allclose(oa.M[:, 2], 0.0)
        assert np.allclose(oa.M[:, 3], 0.0)
        assert np.allclose(oa.M[:, 4], 0.0)

    def test_tilted_probes_add_z_sensitivity(self):
        oa = ObservabilityAnalysis.from_directions(
            [direction(0, 45), direction(90, 45)])
        # tilted directions have n_z != 0 -> nonzero zz column
        assert not np.allclose(oa.M[:, 2], 0.0)

    def test_full_config_resolves_all_six(self):
        oa = ObservabilityAnalysis.from_directions(
            [direction(0), direction(45), direction(90),
             direction(0, 45), direction(90, 45), direction(45, 45)])
        assert oa.analyse()["rank"] == 6

    def test_strain_map_composition(self):
        """Composing with K maps strain -> measurements with full rank."""
        model = ElastoResistivityModel(kappa_long=-0.4, kappa_trans=-0.08)
        oa = ObservabilityAnalysis.from_directions(
            [direction(0), direction(45), direction(90),
             direction(0, 45), direction(90, 45), direction(45, 45)])
        A = oa.strain_map(model, sigma0=1.0)
        assert A.shape == (6, 6)
        assert np.linalg.matrix_rank(A) == 6

    def test_blind_components_reported(self):
        oa = ObservabilityAnalysis([NORMAL_COIL_ROW])
        blind = oa.blind_components()
        assert len(blind) == 5  # 6 - rank(1) = 5 null directions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

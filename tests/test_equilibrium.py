"""Tests for equilibrium-constrained out-of-plane strain recovery."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.tensor_model import (ElastoResistivityModel, direction,
                              EquilibriumConstrainedInverter,
                              free_surface_operator, close_out_of_plane, to_voigt)

NU = 0.30


@pytest.fixture
def model():
    return ElastoResistivityModel(kappa_long=-0.40, kappa_trans=-0.08)


def plane_stress_strain(exx, eyy, exy, nu=NU):
    """A traction-free (plane-stress) strain tensor from its in-plane part."""
    e = np.zeros((3, 3))
    e[0, 0], e[1, 1] = exx, eyy
    e[0, 1] = e[1, 0] = exy
    e[2, 2] = -nu / (1 - nu) * (exx + eyy)
    return e


class TestClosure:
    def test_operator_coefficients(self):
        G = free_surface_operator(NU)
        c = NU / (1 - NU)
        assert np.allclose(G[0], [c, c, 1, 0, 0, 0])
        assert np.allclose(G[1], [0, 0, 0, 1, 0, 0])
        assert np.allclose(G[2], [0, 0, 0, 0, 1, 0])

    def test_closure_formula(self):
        eps = close_out_of_plane(np.diag([0.01, 0.004, 0.0]), NU)
        assert np.isclose(eps[2, 2], -NU / (1 - NU) * (0.01 + 0.004))
        assert eps[0, 2] == 0 and eps[1, 2] == 0

    def test_operator_annihilates_plane_stress(self):
        """G eps = 0 for any genuine plane-stress state."""
        e = plane_stress_strain(0.012, -0.003, 0.002)
        assert np.allclose(free_surface_operator(NU) @ to_voigt(e), 0.0, atol=1e-12)


class TestEquilibriumInverter:
    def test_inplane_rosette_recovers_zz(self, model):
        """In-plane rosette + equilibrium recovers eps_zz it could not see."""
        dirs = [direction(a) for a in (0, 60, 120)]
        inv = EquilibriumConstrainedInverter(model, dirs, poisson=NU,
                                             prior_weight=1e-4, constraint_weight=1e6)
        eps = plane_stress_strain(0.010, -0.004, 0.002)
        y = inv.forward(eps)
        rec = inv.invert(y, prior_strain=np.zeros((3, 3)), as_tensor=True)
        assert np.allclose(rec, eps, atol=1e-4)

    def test_equilibrium_lifts_zz_resolution(self, model):
        """eps_zz resolution rises from ~0 (data only) to ~1 (data+equilibrium)."""
        dirs = [direction(a) for a in (0, 60, 120)]
        data_only = EquilibriumConstrainedInverter(model, dirs, poisson=NU,
                                                   prior_weight=1e-3,
                                                   constraint_weight=0.0)
        with_eq = EquilibriumConstrainedInverter(model, dirs, poisson=NU,
                                                 prior_weight=1e-3,
                                                 constraint_weight=1e6)
        assert data_only.resolved_fraction()[2] < 0.2      # zz unobserved
        assert with_eq.resolved_fraction()[2] > 0.9        # zz now resolved
        # out-of-plane shears too
        assert with_eq.resolved_fraction()[3] > 0.9
        assert with_eq.resolved_fraction()[4] > 0.9

    def test_data_overrides_biased_prior_inplane(self, model):
        """In-plane components still come from data, not the biased prior."""
        dirs = [direction(a) for a in (0, 60, 120)]
        inv = EquilibriumConstrainedInverter(model, dirs, poisson=NU,
                                             prior_weight=1e-4, constraint_weight=1e6)
        eps = plane_stress_strain(0.010, -0.004, 0.002)
        prior = eps + 0.005
        rec = inv.invert(inv.forward(eps), prior_strain=prior, as_tensor=True)
        assert abs(rec[0, 0] - eps[0, 0]) < abs(prior[0, 0] - eps[0, 0])
        assert abs(rec[2, 2] - eps[2, 2]) < abs(prior[2, 2] - eps[2, 2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

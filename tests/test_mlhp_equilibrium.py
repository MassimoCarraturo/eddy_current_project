"""Tests for field-level out-of-plane strain recovery by mlhp FE equilibrium.

Skipped entirely when the optional ``mlhp`` backend is not installed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

pytest.importorskip("mlhp", reason="mlhp FE backend not installed")

from src.tensor_model.mlhp_equilibrium import (
    EigenstrainEquilibrium, plane_strain_stiffness)

NU = 0.30


@pytest.fixture(scope="module")
def solved():
    """One solve shared across the (read-only) checks."""
    eq = EigenstrainEquilibrium(E=1.0, nu=NU, eps0=-2.0e-3,
                                ncells=(10, 10), degree=4).solve()
    X, Z, EPS, SIG = eq.sample_grid(nx=21, nz=21)
    return eq, X, Z, EPS, SIG


def test_stiffness_symmetry_and_ratio():
    C = plane_strain_stiffness(1.0, NU)
    assert np.allclose(C, C.T)
    # uniaxial-strain ratio sigma_xx/sigma_zz = (1-nu)/nu under eps=(1,0,0)
    s = C @ np.array([1.0, 0.0, 0.0])
    assert s[1] / s[0] == pytest.approx(NU / (1 - NU), rel=1e-12)


def test_traction_free_top(solved):
    """sigma_zz and sigma_xz vanish on the free surface z = H."""
    eq, X, Z, EPS, SIG = solved
    scale = np.abs(SIG[..., 1]).max()
    assert np.abs(SIG[:, -1, 1]).max() < 5e-3 * scale     # sigma_zz @ top
    assert np.abs(SIG[:, -1, 2]).max() < 5e-3 * scale     # sigma_xz @ top


def test_interior_equilibrium(solved):
    """div(sigma) ~ 0 in the interior (finite-difference of the sampled field)."""
    eq, X, Z, EPS, SIG = solved
    dx = X[1, 0] - X[0, 0]
    divx = np.gradient(SIG[..., 0], dx, axis=0) + np.gradient(SIG[..., 2], dx, axis=1)
    divz = np.gradient(SIG[..., 2], dx, axis=0) + np.gradient(SIG[..., 1], dx, axis=1)
    inner = (slice(2, -2), slice(2, -2))
    grad_scale = np.abs(np.gradient(SIG[..., 1], dx, axis=1)).max()
    resid = max(np.abs(divx[inner]).max(), np.abs(divz[inner]).max())
    assert resid < 0.1 * grad_scale


def test_closure_exact_at_surface(solved):
    """The free-surface closure matches eps_zz at z = H (where it is valid)."""
    eq, X, Z, EPS, SIG = solved
    clos = eq.closure(EPS[:, -1, 0])
    scale = np.abs(EPS[..., 1]).max()
    assert np.abs(EPS[:, -1, 1] - clos).max() < 1e-2 * scale


def test_closure_fails_in_interior(solved):
    """The closure departs from the true eps_zz with depth -- the motivation."""
    eq, X, Z, EPS, SIG = solved
    clos = eq.closure(EPS[..., 0])
    err = np.abs(EPS[..., 1] - clos)
    scale = np.abs(EPS[..., 1]).max()
    # near the clamped base the per-column closure is grossly wrong
    assert err[:, :5].max() > 0.3 * scale


def test_sample_requires_solve():
    eq = EigenstrainEquilibrium(ncells=(4, 4), degree=2)
    with pytest.raises(RuntimeError):
        eq.sample(0.5, 0.5)

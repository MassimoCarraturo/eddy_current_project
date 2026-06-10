"""Tests for the tensorial elastoresistivity model."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.tensor_model import ElastoResistivityModel, to_voigt, from_voigt
from src.tensor_model.elastoresistivity import GeneralElastoResistivityModel


class TestVoigt:
    def test_roundtrip(self):
        t = np.array([[1.0, 4.0, 5.0], [4.0, 2.0, 6.0], [5.0, 6.0, 3.0]])
        assert np.allclose(from_voigt(to_voigt(t)), t)

    def test_ordering(self):
        t = np.array([[1.0, 9.0, 8.0], [9.0, 2.0, 7.0], [8.0, 7.0, 3.0]])
        v = to_voigt(t)
        # [xx, yy, zz, yz, xz, xy]
        assert np.allclose(v, [1, 2, 3, 7, 8, 9])


class TestElastoResistivity:
    def test_symmetry(self):
        m = ElastoResistivityModel(kappa_long=-0.4, kappa_trans=-0.08)
        eps = np.array([[0.01, 0.002, 0.001],
                        [0.002, -0.003, 0.0005],
                        [0.001, 0.0005, 0.004]])
        ds = m.delta_sigma_ratio(eps)
        assert np.allclose(ds, ds.T)

    def test_longitudinal_constant(self):
        """Uniaxial normal strain along x -> dsig_xx/sig0 = kappa_L * eps."""
        m = ElastoResistivityModel(kappa_long=-0.4, kappa_trans=-0.08)
        eps = np.zeros((3, 3))
        eps[0, 0] = 0.01
        ds = m.delta_sigma_ratio(eps)
        assert np.isclose(ds[0, 0], -0.4 * 0.01)
        assert np.isclose(ds[1, 1], -0.08 * 0.01)  # transverse response

    def test_shear_coupling(self):
        m = ElastoResistivityModel(kappa_long=-0.4, kappa_trans=-0.08)
        eps = np.zeros((3, 3))
        eps[0, 1] = eps[1, 0] = 0.005
        ds = m.delta_sigma_ratio(eps)
        assert np.isclose(ds[0, 1], (-0.4 - (-0.08)) * 0.005)

    def test_scalar_limit_is_volumetric(self):
        """kappa_L = kappa_T reduces to dsig/sig0 = kappa * tr(eps) * I."""
        m = ElastoResistivityModel.from_scalar_kappa(-0.326)
        eps = np.array([[0.01, 0.003, 0.0],
                        [0.003, -0.002, 0.0],
                        [0.0, 0.0, 0.005]])
        ds = m.delta_sigma_ratio(eps)
        assert np.allclose(np.diag(ds), -0.326 * np.trace(eps))
        assert np.isclose(ds[0, 1], 0.0)  # no shear response in scalar limit

    def test_effective_kappa_along_load(self):
        """kappa_eff sensed along the load axis = kappa_L - 2*nu*kappa_T."""
        m = ElastoResistivityModel(kappa_long=-0.4, kappa_trans=-0.08)
        nu = 0.3
        keff = m.effective_scalar_kappa([1, 0, 0], [1, 0, 0], nu)
        assert np.isclose(keff, -0.4 - 2 * nu * (-0.08))

    def test_effective_kappa_direction_dependent(self):
        """Along-load and transverse projections must differ (tensor nature)."""
        m = ElastoResistivityModel(kappa_long=-0.4, kappa_trans=-0.08)
        k_par = m.effective_scalar_kappa([1, 0, 0], [1, 0, 0], 0.3)
        k_perp = m.effective_scalar_kappa([1, 0, 0], [0, 1, 0], 0.3)
        assert not np.isclose(k_par, k_perp)

    def test_conductivity_tensor_anisotropic(self):
        m = ElastoResistivityModel(kappa_long=-0.4, kappa_trans=-0.08)
        eps = np.zeros((3, 3))
        eps[0, 0] = 0.01  # uniaxial -> breaks isotropy
        sig = m.conductivity_tensor(17.7e6, eps)
        vals = np.linalg.eigvalsh(sig)
        assert vals.max() - vals.min() > 0  # anisotropic


class TestGeneralModel:
    def test_voigt_matrix_application(self):
        K = np.eye(6) * -0.1
        gm = ElastoResistivityModel.from_voigt_matrix(K)
        assert isinstance(gm, GeneralElastoResistivityModel)
        eps = np.array([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.03]])
        ds = gm.delta_sigma_ratio(eps)
        assert np.isclose(ds[0, 0], -0.1 * 0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

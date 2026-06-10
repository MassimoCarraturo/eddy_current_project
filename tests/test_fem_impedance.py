"""Tests for the FEM-based coil impedance computation.

These tests require NGSolve. They are skipped automatically when NGSolve is
not installed (``pip install ngsolve``).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.utils.common import ECTCoilParams, MaterialParams

ngsolve = pytest.importorskip("ngsolve")

from src.fem_impedance import FEMCoilImpedance


@pytest.fixture(scope="module")
def fem_result():
    coil = ECTCoilParams(
        r_inner=0.00404, r_outer=0.01184, length=0.00802,
        n_turns=1858, liftoff=0.001, frequency=240e3,
    )
    material = MaterialParams(sigma=16.2e6)
    fem = FEMCoilImpedance(coil=coil, material=material)
    return fem.setup_and_solve()


class TestFEMCoilImpedance:
    def test_z0_purely_inductive(self, fem_result):
        """Coil in free space: reactance dominates, negligible resistance."""
        z0 = fem_result["z0"]
        assert z0.imag > 0
        assert abs(z0.real) < 1e-6 * abs(z0.imag)

    def test_conductor_adds_resistance(self, fem_result):
        """Eddy currents in the conductor dissipate power -> Re(Z) > 0."""
        z = fem_result["impedance"]
        assert z.real > 0

    def test_conductor_reduces_reactance(self, fem_result):
        """Non-magnetic eddy currents reduce the net inductance."""
        z = fem_result["impedance"]
        z0 = fem_result["z0"]
        assert z.imag < z0.imag

    def test_resistance_consistency(self, fem_result):
        """Flux-linkage resistance must equal the direct ohmic-loss integral."""
        z = fem_result["impedance"]
        r_check = fem_result["resistance_check"]
        rel = abs(r_check - z.real) / abs(z.real)
        assert rel < 1e-3, f"R mismatch {rel:.2e}"

    def test_normalized_reactance_matches_analytical(self, fem_result):
        """|Im(Z_norm)| should match the analytical model within ~15%.

        (Sign differs from the current analytical model, whose reflection
        coefficient has the opposite convention; magnitude is the physical
        check here.)
        """
        from src.inversion.dodd_deeds import DoddDeedsModel
        coil = ECTCoilParams(
            r_inner=0.00404, r_outer=0.01184, length=0.00802,
            n_turns=1858, liftoff=0.001, frequency=240e3,
        )
        z_norm_fem = fem_result["impedance_normalized"]
        z_norm_ana = DoddDeedsModel(coil).z_normalized(MaterialParams(sigma=16.2e6))
        rel = abs(abs(z_norm_fem.imag) - abs(z_norm_ana.imag)) / abs(z_norm_ana.imag)
        assert rel < 0.15, f"reactance mismatch {rel:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

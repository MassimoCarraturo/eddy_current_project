"""
Dodd-Deeds analytical model for a multi-turn coil above a conductive half-space.

Implements the impedance formulas from Bowler (2019), "Eddy-Current
Nondestructive Evaluation", Chapter 6 — specifically Equations 6.88 (self-
impedance Z0) and 6.89 (total impedance Z with specimen).

This is the forward model used by the optimisation-based inversion in
optimizer.py.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import jv

from ..utils.common import ECTCoilParams, MaterialParams, mu0


class DoddDeedsModel:
    """Analytical impedance model for a cylindrical coil above a conductor.

    The model evaluates semi-infinite integrals over the Bessel-kernel
    weighted impedance formula.  Integration limits and tolerances are tuned
    for typical ECT frequencies (100 kHz – 1 MHz) and coil sizes (mm-scale).
    """

    def __init__(self, coil: ECTCoilParams, k_max: float = 6000.0):
        self.coil = coil
        self.k_max = k_max

    # -- Bessel kernel -------------------------------------------------------

    @staticmethod
    def _j_integral(k: float, ri: float, ro: float) -> float:
        """Evaluate ∫_{k·ri}^{k·ro} x·J1(x) dx numerically."""
        a, b = k * ri, k * ro
        val, _ = quad(lambda x: x * jv(1, x), a, b,
                      epsabs=1e-9, epsrel=1e-7, limit=500)
        return val

    @staticmethod
    def _j_squared(k: float, ri: float, ro: float) -> float:
        i_val = DoddDeedsModel._j_integral(k, ri, ro)
        return i_val * i_val

    # -- Self-impedance (coil in free space) ---------------------------------

    def z0(self) -> complex:
        """Eq. 6.88 — self-impedance of the coil in free space."""
        c = self.coil
        omega = c.omega
        const = (2j * np.pi * omega * mu0 * c.n_turns ** 2
                 / (c.length ** 2 * (c.r_outer - c.r_inner) ** 2))

        def integrand(k):
            j2 = self._j_squared(k, c.r_inner, c.r_outer)
            return (j2 / k ** 5) * (c.length + (np.exp(-k * c.length) - 1.0) / k)

        val, _ = quad(integrand, 1e-8, self.k_max,
                      epsabs=1e-8, epsrel=1e-6, limit=300)
        return const * val

    # -- Total impedance (coil + conductor) ----------------------------------

    def z_total(self, material: MaterialParams) -> complex:
        """Eq. 6.89 — impedance of the coil over a conductive half-space."""
        c = self.coil
        omega = c.omega
        s = c.liftoff
        l = c.length
        sigma = material.sigma
        mu = material.mu

        const = (1j * omega * mu0 * np.pi * c.n_turns ** 2
                 / (l ** 2 * (c.r_outer - c.r_inner) ** 2))

        def _integrand_part(k):
            j2 = self._j_squared(k, c.r_inner, c.r_outer) / k ** 5
            gamma = np.sqrt(k * k + 1j * omega * mu * sigma / mu0)
            # Reflection coefficient for a non-magnetic half-space.
            # Physics requires (k - gamma)/(k + gamma): eddy currents reduce the
            # net inductance and add resistance. (Verified against an independent
            # axisymmetric FEM solve; see src/fem_impedance/coil_impedance.py.)
            reflection = (k - gamma) / (k + gamma)
            exp_part = (2 * l + (1.0 / k) * (
                2 * np.exp(-k * l) - 2
                + (np.exp(-2 * k * (l + s)) + np.exp(-2 * k * s)
                   - 2 * np.exp(-k * (l + 2 * s))) * reflection
            ))
            return j2 * exp_part

        re_val, _ = quad(lambda k: _integrand_part(k).real, 1e-8, self.k_max,
                         epsabs=1e-8, epsrel=1e-6, limit=300)
        im_val, _ = quad(lambda k: _integrand_part(k).imag, 1e-8, self.k_max,
                         epsabs=1e-8, epsrel=1e-6, limit=300)
        return const * (re_val + 1j * im_val)

    # -- Normalised impedance ------------------------------------------------

    def z_normalized(self, material: MaterialParams) -> complex:
        """Normalised impedance: (Z0 - Z_total) / Im(Z0)."""
        z0_val = self.z0()
        zt_val = self.z_total(material)
        return (z0_val - zt_val) / z0_val.imag

    # -- Convenience: sweep over conductivities -----------------------------

    def impedance_vs_sigma(
        self, sigma_array: np.ndarray
    ) -> np.ndarray:
        """Compute normalised impedance for an array of conductivity values."""
        result = np.empty(len(sigma_array), dtype=complex)
        for i, sigma in enumerate(sigma_array):
            mat = MaterialParams(sigma=sigma)
            result[i] = self.z_normalized(mat)
        return result

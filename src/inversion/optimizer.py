"""
Optimisation-based impedance inversion.

Given a measured complex impedance Z_meas, find the electrical conductivity σ
that minimises |Z_model(σ) − Z_meas| using the Dodd-Deeds forward model.

This replaces the RBF-interpolation approach in the thesis's whole_pipeline.py
with a physics-based inversion that works for any coil geometry without
requiring a pre-computed lookup table.
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize

from .dodd_deeds import DoddDeedsModel
from ..utils.common import ECTCoilParams, MaterialParams


class ImpedanceInverter:
    """Invert measured impedance to conductivity using Dodd-Deeds + optimisation."""

    def __init__(
        self,
        coil: ECTCoilParams,
        sigma_bounds: tuple[float, float] = (1e4, 1e8),
    ):
        self.model = DoddDeedsModel(coil)
        self.sigma_bounds = sigma_bounds
        self._z0_cache = None

    def _residual(self, sigma: float, z_meas: complex) -> float:
        """Squared distance between model prediction and measurement."""
        mat = MaterialParams(sigma=sigma)
        z_pred = self.model.z_normalized(mat)
        return abs(z_pred - z_meas) ** 2

    def invert_single(self, z_meas: complex, sigma_init: float = None) -> float:
        """Find the conductivity that best matches a single impedance measurement.

        Uses bounded scalar minimisation (Brent's method) for robustness.

        Args:
            z_meas: Measured normalised impedance (complex).
            sigma_init: Optional initial guess (not used by Brent but kept for API).

        Returns:
            Estimated conductivity [S/m].
        """
        result = minimize_scalar(
            self._residual,
            bounds=self.sigma_bounds,
            args=(z_meas,),
            method="bounded",
            options={"xatol": 1e-2, "maxiter": 200},
        )
        return result.x

    def invert_array(
        self,
        z_array: np.ndarray,
        sigma_init: float = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Invert an array of impedance measurements to conductivity values.

        Args:
            z_array: 1D array of complex normalised impedance values.
            sigma_init: Initial guess for the first point; subsequent points
                        use the previous result as warm-start.
            verbose: Print progress every 50 points.

        Returns:
            Array of estimated conductivity values [S/m].
        """
        n = len(z_array)
        sigma_out = np.empty(n)
        guess = sigma_init or np.mean(self.sigma_bounds)

        for i in range(n):
            sigma_out[i] = self.invert_single(z_array[i], sigma_init=guess)
            guess = sigma_out[i]

            if verbose and (i + 1) % 50 == 0:
                print(f"  Inverted {i + 1}/{n} points, σ = {sigma_out[i]:.4e} S/m")

        return sigma_out

    def invert_to_strain(
        self,
        z_array: np.ndarray,
        sigma_ref: float,
        kappa: float,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full inversion pipeline: impedance → conductivity → strain.

        Implements:  ε = (Δσ / σ₀) / κ   where  Δσ = σ_inverted − σ_ref

        Args:
            z_array: Array of measured normalised impedance values.
            sigma_ref: Reference (unstressed) conductivity [S/m].
            kappa: Material parameter linking strain and conductivity change.
            verbose: Print progress.

        Returns:
            (sigma_array, strain_array) — inverted conductivities and strains.
        """
        sigma_array = self.invert_array(z_array, sigma_init=sigma_ref, verbose=verbose)
        delta_sigma_rel = (sigma_array - sigma_ref) / sigma_ref
        strain_array = delta_sigma_rel / kappa
        return sigma_array, strain_array

"""
Regularised tensor-strain inversion with a simulation prior -- Step 4.

Recovers the full strain tensor from a set of directional ECT measurements by
combining the (often rank-deficient) measurements with a prior strain field --
e.g. the thermomechanical process simulation.  This is a maximum-a-posteriori
(MAP) / data-assimilation estimate: data where the probes resolve a component,
prior where they do not.

Model
-----
Measurements y relate to the strain Voigt vector eps through the strain map
A = sigma0 * M * K (Step 2/3):

    y = A eps + noise,     noise ~ N(0, sigma_d^2 I)
    eps ~ N(eps_prior, sigma_p^2 I)        (Gaussian prior, mean = simulation)

The MAP estimate is

    eps* = eps_prior + (A^T A + lam I)^{-1} A^T (y - A eps_prior),   lam = sigma_d^2 / sigma_p^2

with resolution matrix  R = (A^T A + lam I)^{-1} A^T A.  diag(R) is the
data-resolved fraction of each component (1 = determined by measurement,
0 = taken from the prior).  The posterior covariance is
sigma_d^2 (A^T A + lam I)^{-1}.

Scalar fallback
---------------
Where only a single normal-coil scalar is available, the measurement senses
(1/2)(sigma_xx + sigma_yy) and can only constrain the volumetric strain via the
invariant coupling kappa_v = (kappa_long + 2 kappa_trans)/3.  ``volumetric_strain``
implements this rigorous scalar-to-scalar reduction.
"""

import numpy as np

from .observability import ObservabilityAnalysis
from .elastoresistivity import from_voigt, to_voigt


class TensorStrainInverter:
    """MAP inversion of the strain tensor from directional ECT measurements."""

    def __init__(self, model, directions, sigma0: float = 1.0,
                 prior_weight: float = 1e-3):
        """
        Parameters
        ----------
        model : ElastoResistivityModel
        directions : list of unit sensing directions (the probe configuration)
        sigma0 : reference conductivity (scales the forward map)
        prior_weight : lam = sigma_d^2 / sigma_p^2; small -> trust data,
            large -> trust prior. Also regularises rank-deficient configs.
        """
        self.model = model
        self.sigma0 = sigma0
        self.prior_weight = float(prior_weight)
        self.A = ObservabilityAnalysis.from_directions(directions).strain_map(model, sigma0)
        self._AtA = self.A.T @ self.A
        self._reg = self._AtA + self.prior_weight * np.eye(6)
        self._reg_inv = np.linalg.inv(self._reg)

    # -- Forward ------------------------------------------------------------

    def forward(self, strain, noise_std: float = 0.0, rng=None) -> np.ndarray:
        """Synthetic measurements y = A eps (+ optional Gaussian noise)."""
        eps_v = to_voigt(strain) if np.ndim(strain) == 2 else np.asarray(strain, float)
        y = self.A @ eps_v
        if noise_std > 0:
            rng = np.random.default_rng() if rng is None else rng
            y = y + rng.normal(scale=noise_std, size=y.shape)
        return y

    # -- Inversion ----------------------------------------------------------

    def invert(self, y, prior_strain=None, as_tensor: bool = False):
        """MAP estimate of the strain Voigt vector (or 3x3 tensor)."""
        y = np.asarray(y, dtype=float)
        if prior_strain is None:
            prior_v = np.zeros(6)
        elif np.ndim(prior_strain) == 2:
            prior_v = to_voigt(prior_strain)
        else:
            prior_v = np.asarray(prior_strain, dtype=float)
        eps_v = prior_v + self._reg_inv @ (self.A.T @ (y - self.A @ prior_v))
        return from_voigt(eps_v) if as_tensor else eps_v

    def invert_field(self, Y, prior_field=None) -> np.ndarray:
        """Vectorised inversion for many points.

        Y : (N, n_meas) measurements; prior_field : (N, 6) or None.
        Returns (N, 6) strain Voigt vectors.
        """
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        n = Y.shape[0]
        prior = np.zeros((n, 6)) if prior_field is None else np.atleast_2d(prior_field)
        # eps = prior + (reg_inv A^T)(y - A prior)
        G = self._reg_inv @ self.A.T               # 6 x n_meas
        residual = Y - prior @ self.A.T            # N x n_meas
        return prior + residual @ G.T

    # -- Diagnostics --------------------------------------------------------

    def resolution_matrix(self) -> np.ndarray:
        """R = (A^T A + lam I)^{-1} A^T A; diag = data-resolved fraction."""
        return self._reg_inv @ self._AtA

    def data_resolved_fraction(self) -> np.ndarray:
        """Per-component fraction determined by data (rest comes from prior)."""
        return np.clip(np.diag(self.resolution_matrix()), 0.0, 1.0)

    def posterior_std(self, noise_std: float) -> np.ndarray:
        """Posterior standard deviation per component, sqrt(diag(cov))."""
        cov = (noise_std ** 2) * self._reg_inv
        return np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))


def volumetric_strain(scalar_measurement: float, model, sigma0: float = 1.0) -> float:
    """Recover volumetric strain from a single normal-coil scalar measurement.

    A normal coil senses (1/2)(sigma_xx + sigma_yy). For an isotropic
    (volumetric) strain state this equals sigma0 * kappa_v * eps_vol with
    kappa_v = (kappa_long + 2 kappa_trans)/3, so eps_vol = y / (sigma0 kappa_v).
    """
    kappa_v = (model.kappa_long + 2.0 * model.kappa_trans) / 3.0
    if kappa_v == 0:
        return np.nan
    return float(scalar_measurement / (sigma0 * kappa_v))

"""
Equilibrium-constrained strain inversion: recovering the out-of-plane strain.

Eddy-current probes sense only the in-plane conductivity (skin effect), so the
measurement map A leaves the out-of-plane components epsilon_zz, epsilon_xz,
epsilon_yz unobserved. Mechanics supplies them. On a traction-free surface with
normal z, equilibrium forces a plane-stress state (sigma_zz = sigma_xz =
sigma_yz = 0); isotropic Hooke's law then gives, exactly,

    epsilon_zz = - nu/(1-nu) (epsilon_xx + epsilon_yy),   epsilon_xz = epsilon_yz = 0.

For a build whose residual field varies mainly through the height (laterally
homogeneous), interior equilibrium propagates this plane-stress state to every
depth, so the same closure holds through the column. Lateral gradients make
sigma_zz nonzero and require the full elastic boundary-value problem (the role of
the process simulation / Digital Twin) -- a higher-order correction not modelled
here.

These relations are three linear constraints on the strain Voigt vector. Stacking
them with the EM measurement map turns the rank-deficient inverse problem into a
full-rank one: an in-plane rosette then resolves all six strain components ---
three from the data, three from equilibrium --- and the resolution matrix makes
the gain explicit (the epsilon_zz diagonal rises from ~0 to ~1).
"""

import numpy as np

from .observability import ObservabilityAnalysis
from .elastoresistivity import to_voigt, from_voigt


def free_surface_operator(poisson: float) -> np.ndarray:
    """Free-surface (plane-stress) constraints as a 3x6 Voigt operator G.

    The rows encode, in Voigt order [xx, yy, zz, yz, xz, xy],
        eps_zz + c (eps_xx + eps_yy) = 0  with c = nu/(1-nu),
        eps_yz = 0,
        eps_xz = 0,
    i.e. G @ eps_voigt = 0 on a traction-free surface with normal z.
    """
    c = poisson / (1.0 - poisson)
    G = np.zeros((3, 6))
    G[0, 0] = c
    G[0, 1] = c
    G[0, 2] = 1.0
    G[1, 3] = 1.0
    G[2, 4] = 1.0
    return G


def close_out_of_plane(strain, poisson: float) -> np.ndarray:
    """Fill the out-of-plane components from the in-plane ones (free surface).

    Accepts a 3x3 tensor or a length-6 Voigt vector; returns a 3x3 tensor whose
    epsilon_zz, epsilon_xz, epsilon_yz are set by the plane-stress closure and
    whose in-plane components (xx, yy, xy) are unchanged.
    """
    eps = from_voigt(strain) if np.ndim(strain) == 1 else np.array(strain, float)
    out = eps.copy()
    out[2, 2] = -poisson / (1.0 - poisson) * (eps[0, 0] + eps[1, 1])
    out[0, 2] = out[2, 0] = 0.0
    out[1, 2] = out[2, 1] = 0.0
    return out


class EquilibriumConstrainedInverter:
    """MAP strain inversion augmented with free-surface equilibrium constraints.

    Solves
        eps* = argmin || A eps - y ||^2 + w || G eps ||^2 + lambda || eps - prior ||^2,
    where A is the directional-probe strain map, G the free-surface operator, w a
    large constraint weight (hard limit), and lambda the prior weight. With w = 0
    this reduces to the data-only inverter of ``TensorStrainInverter``.
    """

    def __init__(self, model, directions, poisson: float, sigma0: float = 1.0,
                 prior_weight: float = 1e-3, constraint_weight: float = 1e6):
        self.model = model
        self.poisson = float(poisson)
        self.A = ObservabilityAnalysis.from_directions(directions).strain_map(model, sigma0)
        self.G = free_surface_operator(poisson)
        self.lam = float(prior_weight)
        self.wc = float(constraint_weight)

        self._AtA = self.A.T @ self.A
        self._GtG = self.G.T @ self.G
        self._info = self._AtA + self.wc * self._GtG     # data + equilibrium
        self._reg_inv = np.linalg.inv(self._info + self.lam * np.eye(6))

    # -- forward / inverse ---------------------------------------------------

    def forward(self, strain, noise_std: float = 0.0, rng=None) -> np.ndarray:
        eps_v = to_voigt(strain) if np.ndim(strain) == 2 else np.asarray(strain, float)
        y = self.A @ eps_v
        if noise_std > 0:
            rng = np.random.default_rng() if rng is None else rng
            y = y + rng.normal(scale=noise_std, size=y.shape)
        return y

    def invert(self, y, prior_strain=None, as_tensor: bool = False):
        y = np.asarray(y, dtype=float)
        if prior_strain is None:
            prior_v = np.zeros(6)
        elif np.ndim(prior_strain) == 2:
            prior_v = to_voigt(prior_strain)
        else:
            prior_v = np.asarray(prior_strain, dtype=float)
        # G eps = 0 is the equilibrium target; residual of A and of G about prior
        rhs = (self.A.T @ (y - self.A @ prior_v)
               - self.wc * (self.G.T @ (self.G @ prior_v)))
        eps_v = prior_v + self._reg_inv @ rhs
        return from_voigt(eps_v) if as_tensor else eps_v

    def invert_field(self, Y, prior_field=None) -> np.ndarray:
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        n = Y.shape[0]
        prior = np.zeros((n, 6)) if prior_field is None else np.atleast_2d(prior_field)
        out = np.empty((n, 6))
        for i in range(n):
            out[i] = self.invert(Y[i], prior[i])
        return out

    # -- diagnostics ---------------------------------------------------------

    def resolution_matrix(self) -> np.ndarray:
        """R = (A^T A + w G^T G + lambda I)^{-1} (A^T A + w G^T G).

        Diagonal: fraction of each component fixed by data plus equilibrium
        (versus the prior).
        """
        return self._reg_inv @ self._info

    def resolved_fraction(self) -> np.ndarray:
        return np.clip(np.diag(self.resolution_matrix()), 0.0, 1.0)

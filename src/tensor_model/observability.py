"""
Observability analysis for tensor-strain reconstruction from ECT.

The question this module answers: given a set of ECT probe configurations,
*which* components of the conductivity tensor (and hence, through the
elastoresistivity model, which components of the strain tensor) can actually be
resolved, and which are invisible?

Directional-probe model
------------------------
A directionally-selective eddy-current probe drives current predominantly along
an in-plane direction n_hat and therefore senses, to leading order, the
effective conductivity along that direction:

    y(n_hat) = n_hat^T sigma n_hat

This is the conductivity analogue of a mechanical strain-gauge rosette, where a
gauge at angle theta reads the normal strain along theta.  Stacking several
probes gives a linear map

    y = M . sigma_voigt           (M has one row n_hat^T(.)n_hat per probe)

Composing with the elastoresistivity matrix K (sigma_voigt = sigma0 * K *
eps_voigt) yields the map from strain to measurements,

    y = (sigma0 * M . K) . eps_voigt = A . eps_voigt

The singular value decomposition of A reveals the rank (number of resolvable
strain combinations), the condition number (how well-posed the inversion is),
and the null space (strain combinations that produce no signal -- the blind
directions).

The axisymmetric normal coil
----------------------------
A conventional coil with its axis normal to the surface is azimuthally
symmetric.  Its eddy currents are purely in-plane and, averaged around the
azimuth, it senses the in-plane mean conductivity (1/2)(sigma_xx + sigma_yy).
It is blind to sigma_zz, to the in-plane anisotropy (sigma_xx - sigma_yy) and to
all shears.  This is encoded by ``NORMAL_COIL_ROW``.
"""

import numpy as np

from .elastoresistivity import VOIGT_INDEX

# A normal axisymmetric coil senses (1/2)(sigma_xx + sigma_yy).
NORMAL_COIL_ROW = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])


def projection_row(n_hat) -> np.ndarray:
    """Row vector r such that  r . to_voigt(T) == n_hat^T T n_hat.

    With tensor-Voigt ordering [xx, yy, zz, yz, xz, xy], the off-diagonal
    entries carry a factor of two (because T_ij and T_ji both contribute).
    """
    n = np.asarray(n_hat, dtype=float)
    n = n / np.linalg.norm(n)
    r = np.empty(6)
    for idx, (i, j) in enumerate(VOIGT_INDEX):
        r[idx] = n[i] * n[j] * (1.0 if i == j else 2.0)
    return r


def direction(theta_deg: float, tilt_deg: float = 90.0) -> np.ndarray:
    """Unit sensing direction from azimuth and tilt angles (degrees).

    ``tilt_deg`` is measured from the +z (build) axis: tilt = 90 gives an
    in-plane direction at azimuth ``theta_deg``; tilt = 0 gives +z.
    """
    th = np.radians(theta_deg)
    ph = np.radians(tilt_deg)
    return np.array([
        np.sin(ph) * np.cos(th),
        np.sin(ph) * np.sin(th),
        np.cos(ph),
    ])


class ObservabilityAnalysis:
    """Assemble a probe measurement matrix and analyse what it can resolve."""

    def __init__(self, rows, labels=None):
        self.M = np.atleast_2d(np.asarray(rows, dtype=float))
        self.labels = list(labels) if labels is not None else None

    # -- Constructors --------------------------------------------------------

    @classmethod
    def from_directions(cls, directions, labels=None) -> "ObservabilityAnalysis":
        rows = [projection_row(d) for d in directions]
        return cls(rows, labels=labels)

    # -- Measurement maps ----------------------------------------------------

    def sigma_map(self) -> np.ndarray:
        """Linear map from conductivity Voigt vector to measurements."""
        return self.M

    def strain_map(self, model, sigma0: float = 1.0) -> np.ndarray:
        """Linear map from strain Voigt vector to measurements.

        Composes the probe matrix with the elastoresistivity matrix K.
        """
        return sigma0 * (self.M @ model.voigt_matrix)

    # -- Analysis ------------------------------------------------------------

    def analyse(self, matrix=None) -> dict:
        """SVD-based observability metrics for a measurement matrix.

        Defaults to the conductivity map M; pass ``strain_map(...)`` to analyse
        strain observability instead.
        """
        A = self.M if matrix is None else np.atleast_2d(matrix)
        U, s, Vt = np.linalg.svd(A, full_matrices=True)
        smax = s[0] if s.size else 0.0
        tol = max(A.shape) * np.finfo(float).eps * smax
        rank = int((s > tol).sum())
        if rank > 0:
            cond = float(s[0] / s[rank - 1])
        else:
            cond = np.inf
        null_space = Vt[rank:]  # rows of V^T beyond the rank span the null space
        return {
            "singular_values": s,
            "rank": rank,
            "condition_number": cond,
            "null_space": null_space,
            "row_space": Vt[:rank],
            "tolerance": tol,
        }

    def resolvable_dimension(self, matrix=None) -> int:
        return self.analyse(matrix)["rank"]

    def blind_components(self, matrix=None, threshold: float = 1e-9):
        """Human-readable description of the null space (blind directions).

        Returns a list of (vector, dominant_component_label) describing strain
        or conductivity combinations the configuration cannot see.
        """
        names = ["xx", "yy", "zz", "yz", "xz", "xy"]
        null = self.analyse(matrix)["null_space"]
        out = []
        for vec in null:
            v = np.where(np.abs(vec) < threshold, 0.0, vec)
            dominant = names[int(np.argmax(np.abs(v)))]
            out.append((v, dominant))
        return out

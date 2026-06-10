"""
Elastoresistivity (piezoconductivity) model.

Generalises the scalar strain-conductivity law used in the thesis,

    delta_sigma / sigma0 = kappa * epsilon          (scalar, 1-D)

to the full tensorial relation between the strain tensor and the induced
change in the (now anisotropic) conductivity tensor,

    (delta_sigma / sigma0)_ij = K_ijkl  epsilon_kl   (tensorial, 3-D)

Why a tensor?
-------------
Strain breaks the isotropy of the material, so under load the conductivity
itself becomes a second-order tensor sigma_ij.  To first order it couples to
the strain tensor through the fourth-order *elastoresistivity* tensor K_ijkl.

For an isotropic base material K_ijkl has the isotropic form

    K_ijkl = a * d_ij d_kl + b * (d_ik d_jl + d_il d_jk)

i.e. only TWO independent constants.  We parameterise them physically as

    kappa_long  (kappa_L): relative change of conductivity measured ALONG a
                           direction, caused by a normal strain in that same
                           direction;
    kappa_trans (kappa_T): relative change of conductivity measured along a
                           direction, caused by a normal strain PERPENDICULAR
                           to it,

so that

    (dsig)_ii = kappa_L * eps_ii + kappa_T * sum_{k != i} eps_kk
    (dsig)_ij = (kappa_L - kappa_T) * eps_ij        (i != j, shear)

with a = kappa_T and 2b = kappa_L - kappa_T.

Connection to the thesis scalar kappa
--------------------------------------
The thesis value kappa = -0.326 (316L) is a uniaxial *projection* of this
tensor: it is the relative conductivity change sensed along one direction in a
uniaxial-stress test.  ``effective_scalar_kappa`` reproduces it for a given
(load, sensing) direction pair.  Because one scalar test gives one equation
for the two constants (kappa_L, kappa_T), separating them requires loading the
specimen along several orientations -- the experimental recommendation that
accompanies this model.

Anisotropic base material
-------------------------
LPBF 316L is itself microstructurally anisotropic (columnar grains along the
build axis), so the unstrained sigma0 may already be anisotropic and K_ijkl
may have more than two constants.  ``ElastoResistivityModel.from_voigt_matrix``
accepts an arbitrary 6x6 elastoresistivity matrix for that general case.
"""

import numpy as np
from dataclasses import dataclass

# Voigt ordering for a symmetric 3x3 tensor, using *tensor* (not engineering)
# components: v = [T_xx, T_yy, T_zz, T_yz, T_xz, T_xy].
VOIGT_INDEX = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def to_voigt(tensor: np.ndarray) -> np.ndarray:
    """Convert a symmetric 3x3 tensor to its length-6 Voigt vector.

    Uses tensor components (no factor of two on the shear entries).
    """
    t = np.asarray(tensor, dtype=float)
    return np.array([t[i, j] for (i, j) in VOIGT_INDEX])


def from_voigt(vec: np.ndarray) -> np.ndarray:
    """Convert a length-6 Voigt vector (tensor components) to a symmetric 3x3."""
    v = np.asarray(vec, dtype=float)
    t = np.zeros((3, 3))
    for idx, (i, j) in enumerate(VOIGT_INDEX):
        t[i, j] = v[idx]
        t[j, i] = v[idx]
    return t


@dataclass
class ElastoResistivityModel:
    """Fourth-order strain-to-conductivity coupling for an isotropic base.

    Parameters
    ----------
    kappa_long, kappa_trans : float
        Longitudinal and transverse elastoresistivity constants (dimensionless,
        relative conductivity change per unit strain).
    """

    kappa_long: float
    kappa_trans: float

    # -- Constructors --------------------------------------------------------

    @classmethod
    def isotropic(cls, kappa_long: float, kappa_trans: float) -> "ElastoResistivityModel":
        return cls(kappa_long=kappa_long, kappa_trans=kappa_trans)

    @classmethod
    def from_scalar_kappa(cls, kappa: float) -> "ElastoResistivityModel":
        """Degenerate model reproducing the scalar volumetric law.

        Setting kappa_long = kappa_trans = kappa makes the conductivity respond
        only to the volumetric strain: (dsig/sig0)_ij = kappa * tr(eps) * d_ij.
        This is the rigorous scalar limit of the tensor model.
        """
        return cls(kappa_long=kappa, kappa_trans=kappa)

    @classmethod
    def from_voigt_matrix(cls, matrix: np.ndarray) -> "GeneralElastoResistivityModel":
        """Build a general (anisotropic) model from an arbitrary 6x6 matrix."""
        return GeneralElastoResistivityModel(np.asarray(matrix, dtype=float))

    # -- Core tensor relations ----------------------------------------------

    @property
    def voigt_matrix(self) -> np.ndarray:
        """6x6 matrix K mapping a strain Voigt vector to (dsigma/sigma0) Voigt."""
        a = self.kappa_trans
        two_b = self.kappa_long - self.kappa_trans
        K = np.zeros((6, 6))
        # normal-normal 3x3 block
        for i in range(3):
            for j in range(3):
                K[i, j] = a + (two_b if i == j else 0.0)
        # shear diagonal block
        for s in range(3, 6):
            K[s, s] = two_b
        return K

    def delta_sigma_ratio(self, strain: np.ndarray) -> np.ndarray:
        """Relative conductivity change (delta_sigma / sigma0) as a 3x3 tensor."""
        eps = np.asarray(strain, dtype=float)
        a = self.kappa_trans
        b = 0.5 * (self.kappa_long - self.kappa_trans)
        return a * np.trace(eps) * np.eye(3) + 2.0 * b * eps

    def conductivity_tensor(self, sigma0: float, strain: np.ndarray) -> np.ndarray:
        """Full anisotropic conductivity tensor sigma_ij = sigma0 (I + dsig/sig0)."""
        return sigma0 * (np.eye(3) + self.delta_sigma_ratio(strain))

    def principal_conductivities(self, sigma0: float, strain: np.ndarray):
        """Eigenvalues (principal conductivities) of the strained tensor."""
        sig = self.conductivity_tensor(sigma0, strain)
        vals, vecs = np.linalg.eigh(sig)
        return vals, vecs

    # -- Reduction to the scalar (uniaxial) regime --------------------------

    def effective_scalar_kappa(self, load_dir, sense_dir, poisson: float) -> float:
        """Effective scalar kappa for a uniaxial-*stress* test.

        Applies unit axial strain along ``load_dir`` with transverse strain
        ``-poisson`` (uniaxial stress state) and returns the relative
        conductivity change sensed along ``sense_dir``:

            kappa_eff = sense_dir^T (delta_sigma / sigma0) sense_dir.

        For load == sense (along the load axis):  kappa_L - 2*nu*kappa_T.
        """
        l = np.asarray(load_dir, dtype=float)
        l = l / np.linalg.norm(l)
        n = np.asarray(sense_dir, dtype=float)
        n = n / np.linalg.norm(n)
        eps = (1.0 + poisson) * np.outer(l, l) - poisson * np.eye(3)
        dsig = self.delta_sigma_ratio(eps)
        return float(n @ dsig @ n)


@dataclass
class GeneralElastoResistivityModel:
    """Elastoresistivity model defined by an arbitrary 6x6 Voigt matrix.

    Use for anisotropic base materials (e.g. transversely isotropic LPBF 316L)
    where the two-constant isotropic form is insufficient.
    """

    matrix: np.ndarray

    @property
    def voigt_matrix(self) -> np.ndarray:
        return self.matrix

    def delta_sigma_ratio(self, strain: np.ndarray) -> np.ndarray:
        return from_voigt(self.matrix @ to_voigt(np.asarray(strain, dtype=float)))

    def conductivity_tensor(self, sigma0: float, strain: np.ndarray) -> np.ndarray:
        return sigma0 * (np.eye(3) + self.delta_sigma_ratio(strain))

    def principal_conductivities(self, sigma0: float, strain: np.ndarray):
        sig = self.conductivity_tensor(sigma0, strain)
        vals, vecs = np.linalg.eigh(sig)
        return vals, vecs

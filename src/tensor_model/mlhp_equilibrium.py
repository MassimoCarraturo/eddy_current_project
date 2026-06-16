"""Field-level out-of-plane strain recovery by FE elastic equilibrium (mlhp).

The per-column free-surface closure (``equilibrium.py``) is exact only at the
traction-free surface, where plane stress forces
``eps_zz = -nu/(1-nu) (eps_xx + eps_yy)``. Below the surface, lateral gradients
make ``sigma_zz`` non-zero and the algebraic closure degrades -- near a clamped
baseplate it can even change sign. Resolving the out-of-plane strain there needs
the genuine balance law ``div(sigma) = 0``, not a pointwise relation.

This module solves that balance law with the **mlhp** hp-FEM kernel. It is the
transparent stand-in for the thermomechanical Digital Twin whose role in the
workflow is exactly to supply an equilibrated stress field: an in-plane ECT
rosette measures ``eps_xx`` (and ``eps_yy``) at the surface, and the equilibrium
solve carries that information into the depth to fix the unobserved ``eps_zz``.

Model
-----
A vertical ``(x, z)`` slice through the build (plane strain in the out-of-slice
direction ``y``): ``x`` is lateral and ECT-measurable, ``z`` is the build height
whose top face ``z = H`` is the traction-free ECT surface and whose base
``z = 0`` is clamped to the baseplate. Residual stress is driven by an in-plane,
isotropic inelastic (thermal-like) eigenstrain

    eps*(x, z) = beta(x, z) (e_xx + e_zz),
    beta(x, z) = eps0 * x/W (1 - x/W) * (1 - z/H),

which vanishes on the three free faces. The eigenstress sigma* = kappa*beta*I
(``kappa = E / ((1+nu)(1-2nu))``) therefore has no normal traction on the free
boundary, so the Mura problem reduces to a single equivalent body force
``b = -div(sigma*) = -kappa*grad(beta)`` with a clamped base -- no surface
tractions to assemble. The recovered *elastic* strain (what diffraction / the
elastoresistive ECT senses) is ``eps_el = eps(u) - eps*`` and ``sigma = C:eps_el``.

Requires the ``mlhp`` package (an external FE kernel, not a hard dependency of
this project); :data:`HAVE_MLHP` reports availability.
"""

import numpy as np

try:
    import mlhp
    HAVE_MLHP = True
except ImportError:  # pragma: no cover - mlhp is an optional backend
    mlhp = None
    HAVE_MLHP = False


def plane_strain_stiffness(E: float, nu: float) -> np.ndarray:
    """In-plane plane-strain stiffness (3x3) for Voigt ``[xx, zz, xz]``."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return np.array([[lam + 2 * mu, lam, 0.0],
                     [lam, lam + 2 * mu, 0.0],
                     [0.0, 0.0, 2 * mu]])


class EigenstrainEquilibrium:
    """Plane-strain residual-stress solve on a clamped-base ``(x, z)`` slice.

    Parameters
    ----------
    E, nu : float
        Young's modulus and Poisson's ratio (only ``nu`` affects the strain
        ratios; ``E`` scales out of ``div(sigma) = 0``).
    eps0 : float
        Peak eigenstrain magnitude (negative for thermal contraction).
    width, height : float
        Slice extent ``W`` (lateral ``x``) and ``H`` (build height ``z``).
    ncells : (int, int)
        Background grid cell counts.
    degree : int
        Polynomial degree of the hp trunk space.
    """

    def __init__(self, E=1.0, nu=0.3, eps0=-2.0e-3,
                 width=1.0, height=1.0, ncells=(10, 10), degree=4):
        if not HAVE_MLHP:
            raise ImportError("EigenstrainEquilibrium requires the 'mlhp' package.")
        self.E, self.nu, self.eps0 = float(E), float(nu), float(eps0)
        self.W, self.H = float(width), float(height)
        self.ncells, self.degree = tuple(ncells), int(degree)
        self.C = plane_strain_stiffness(E, nu)
        self.kappa = E / ((1 + nu) * (1 - 2 * nu))     # sigma*_xx = sigma*_zz = kappa*beta
        self._dofs = None
        self._grad = None

    # -- eigenstrain ---------------------------------------------------------

    def beta(self, x, z):
        """Scalar eigenstrain field (in-plane isotropic magnitude)."""
        xn, zn = np.asarray(x) / self.W, np.asarray(z) / self.H
        return self.eps0 * xn * (1 - xn) * (1 - zn)

    # -- solve ---------------------------------------------------------------

    def solve(self):
        """Assemble and solve the equilibrium system; caches the solution."""
        D = 2
        grid = mlhp.makeRefinedGrid(list(self.ncells), [self.W, self.H])
        basis = mlhp.makeHpTrunkSpace(grid, degree=self.degree, nfields=D)

        # Clamp the baseplate (face 2, z = 0).
        dirichlet = mlhp.combineDirichletDofs([
            mlhp.integrateDirichletDofs(mlhp.scalarField(D, 0.0), basis, [2], ifield=0),
            mlhp.integrateDirichletDofs(mlhp.scalarField(D, 0.0), basis, [2], ifield=1),
        ])

        kin = mlhp.smallStrainKinematics(D)
        con = mlhp.planeStrainMaterial(mlhp.scalarField(D, self.E),
                                       mlhp.scalarField(D, self.nu))

        # b = -kappa * grad(beta); beta = eps0 (x/W)(1-x/W)(1-z/H)
        #   dbeta/dx = (eps0/W)(1-2x/W)(1-z/H),  dbeta/dz = -(eps0/H)(x/W)(1-x/W)
        cx = -self.kappa * self.eps0 / self.W
        cz = self.kappa * self.eps0 / self.H
        bx = f"{cx}*(1-2*x/{self.W})*(1-y/{self.H})"
        bz = f"{cz}*((x/{self.W})*(1-x/{self.W}))"
        source = mlhp.vectorField(D, f"[{bx}, {bz}]")

        matrix = mlhp.allocateSparseMatrix(basis, dirichlet[0])
        vector = mlhp.allocateRhsVector(matrix)
        integ = mlhp.staticDomainIntegrand(kin, con, source)
        mlhp.integrateOnDomain(basis, integ, [matrix, vector], dirichletDofs=dirichlet)

        interior = mlhp.makeCGSolver(rtol=1e-12, maxiter=40000)(matrix, vector)
        dofs = mlhp.inflateDofs(interior, dirichlet)

        self._basis = basis
        self._dofs = dofs
        self._grad = mlhp.vectorEvaluator(basis, dofs, difforder=1)
        return self

    # -- sampling ------------------------------------------------------------

    def _require_solved(self):
        if self._grad is None:
            raise RuntimeError("call solve() before sampling.")

    def sample(self, x, z):
        """Elastic strain ``[xx, zz, xz]`` and stress ``[xx, zz, xz]`` at ``(x, z)``."""
        self._require_solved()
        g = list(self._grad([float(x), float(z)]))     # [du0/dx, du0/dz, du1/dx, du1/dz]
        total = np.array([g[0], g[3], 0.5 * (g[1] + g[2])])
        b = float(self.beta(x, z))
        eps_el = total - np.array([b, b, 0.0])          # eps*_xz = 0
        return eps_el, self.C @ eps_el

    def sample_grid(self, nx=41, nz=41):
        """Sample on a regular grid.

        Returns ``X, Z`` (each ``nx x nz``) and ``EPS, SIG`` (each
        ``nx x nz x 3`` for Voigt ``[xx, zz, xz]``).
        """
        self._require_solved()
        xs = np.linspace(0, self.W, nx)
        zs = np.linspace(0, self.H, nz)
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        EPS = np.zeros((nx, nz, 3))
        SIG = np.zeros((nx, nz, 3))
        for i in range(nx):
            for j in range(nz):
                EPS[i, j], SIG[i, j] = self.sample(xs[i], zs[j])
        return X, Z, EPS, SIG

    def closure(self, eps_xx):
        """Per-column free-surface closure ``eps_zz = -nu/(1-nu) eps_xx``.

        On this plane-strain slice ``eps_yy = 0``, so the surface plane-stress
        relation reduces to a function of ``eps_xx`` alone.
        """
        return -self.nu / (1 - self.nu) * np.asarray(eps_xx)

"""
FEM-based coil impedance computation using NGSolve.

Solves the time-harmonic eddy-current problem for an axisymmetric multi-turn
coil above a conductive half-space and extracts the coil impedance from the
magnetic vector potential.  This enables impedance computation for geometries
where the analytical Dodd-Deeds model breaks down (finite-thickness plates,
layered media, defects).

Formulation
-----------
Unknown: the azimuthal magnetic vector potential A_phi(rho, z) (complex).
The time-harmonic curl-curl equation in cylindrical (axisymmetric) coordinates,
written in the standard symmetric weak form, is

    (1/mu) integral [ grad(A).grad(v) + A v / rho^2 ] rho drho dz
    + j w integral sigma A v rho drho dz
    = integral J_s v rho drho dz                          for all test v,

with Dirichlet condition A = 0 on the symmetry axis (rho = 0, where the
azimuthal component must vanish) and on the truncated outer boundary.

The driving coil is modelled as a uniform azimuthal current density
J_s = n I / S_coil over the winding cross-section S_coil = (r_o - r_i) * length.

Impedance
---------
The flux linkage of the whole winding is

    Lambda = (n / S_coil) integral_coil (2 pi rho A) drho dz,

and the coil impedance follows from Z = j w Lambda / I.  The self-impedance
Z0 is obtained from the same model with sigma = 0 everywhere (coil in free
space), giving a self-consistent normalisation

    Z_norm = (Z0 - Z) / Im(Z0).

This mirrors the definition used by the analytical model in dodd_deeds.py.
"""

import numpy as np
from dataclasses import dataclass

from ..utils.common import ECTCoilParams, MaterialParams, mu0


@dataclass
class FEMCoilImpedance:
    """Compute coil impedance from a 2D axisymmetric FEM solution.

    Requires NGSolve (optional dependency). Install via ``pip install ngsolve``.
    """

    coil: ECTCoilParams
    material: MaterialParams
    domain_r_max: float = 0.05      # Radial extent of the domain [m]
    domain_z_top: float = 0.03      # Top of the air region [m]
    domain_z_bot: float = -0.02     # Bottom of the conductor [m]
    max_mesh_size: float = 0.004    # Background mesh size [m]
    order: int = 3                  # FE polynomial order
    skin_mesh_div: float = 3.0      # Skin-layer mesh size = skin_depth / div
    coil_mesh_size: float = 8e-4    # Mesh size inside the coil winding [m]

    def _check_ngsolve(self):
        try:
            import ngsolve  # noqa: F401
            return True
        except ImportError:
            raise ImportError(
                "NGSolve is required for FEM impedance computation. "
                "Install it via: pip install ngsolve"
            )

    # -- Geometry & mesh -----------------------------------------------------

    def _build_mesh(self):
        from netgen.geom2d import CSG2d, Rectangle
        from ngsolve import Mesh

        c = self.coil
        delta = self.material.skin_depth(c.omega)
        # Conductor "skin" block: a few skin depths thick (at least 1 mm) so the
        # exponential eddy-current decay is captured by the refined mesh.
        skin = float(max(5.0 * delta, 1e-3))
        z0c = float(c.liftoff)
        z1c = float(c.liftoff + c.length)
        R = float(self.domain_r_max)

        geo = CSG2d()

        cond_skin = Rectangle(
            pmin=(0, -skin), pmax=(R, 0), mat="conductor",
            left="axis", right="outer",
        ).Maxh(float(delta / self.skin_mesh_div))
        cond_bulk = Rectangle(
            pmin=(0, float(self.domain_z_bot)), pmax=(R, -skin), mat="conductor",
            left="axis", right="outer", bottom="outer",
        ).Maxh(float(self.max_mesh_size))
        air = Rectangle(
            pmin=(0, 0), pmax=(R, float(self.domain_z_top)), mat="air",
            left="axis", right="outer", top="outer",
        ).Maxh(float(self.max_mesh_size))
        coil_rect = Rectangle(
            pmin=(float(c.r_inner), z0c), pmax=(float(c.r_outer), z1c),
            mat="coil",
        ).Maxh(float(self.coil_mesh_size))

        geo.Add(cond_bulk + cond_skin)
        geo.Add(air - coil_rect)
        geo.Add(coil_rect)

        return Mesh(geo.GenerateMesh(maxh=float(self.max_mesh_size)))

    # -- Solve ---------------------------------------------------------------

    def setup_and_solve(self) -> dict:
        """Build the FEM model, solve, and return the solution and impedance.

        Returns a dictionary with keys:
            'mesh', 'V', 'solution'            : NGSolve mesh, space, A over conductor
            'impedance'                        : Z over the conductor (complex)
            'z0'                               : self-impedance Z0 in free space
            'impedance_normalized'             : (Z0 - Z) / Im(Z0)
            'resistance_check'                 : R from direct ohmic-loss integral
        """
        self._check_ngsolve()

        from ngsolve import (
            H1, BilinearForm, LinearForm, GridFunction, Integrate,
            grad, dx, Conj, x as rho,
        )

        c = self.coil
        m = self.material
        omega = c.omega

        mesh = self._build_mesh()
        V = H1(mesh, order=self.order, complex=True, dirichlet="axis|outer")

        S_coil = (c.r_outer - c.r_inner) * c.length
        I_source = 1.0
        J0 = c.n_turns * I_source / S_coil
        coil_cf = mesh.MaterialCF({"coil": 1.0}, default=0.0)

        def solve_for_sigma(sigma_value):
            sigma_cf = mesh.MaterialCF({"conductor": sigma_value}, default=0.0)
            A = V.TrialFunction()
            v = V.TestFunction()

            a = BilinearForm(V, symmetric=True, check_unused=False)
            a += (1.0 / mu0) * (grad(A) * grad(v) + A * v / rho ** 2) * rho * dx
            a += 1j * omega * sigma_cf * A * v * rho * dx
            a.Assemble()

            f = LinearForm(V)
            f += J0 * coil_cf * v * rho * dx
            f.Assemble()

            gfu = GridFunction(V)
            gfu.vec.data = a.mat.Inverse(V.FreeDofs(), inverse="umfpack") * f.vec

            # Flux linkage Lambda = (n/S) * 2pi * integral_coil rho * A
            lam = (c.n_turns / S_coil) * 2 * np.pi * Integrate(coil_cf * rho * gfu, mesh)
            impedance = 1j * omega * lam / I_source
            return gfu, impedance, sigma_cf

        gfu_air, z0, _ = solve_for_sigma(0.0)
        gfu_cond, z_cond, sigma_cf = solve_for_sigma(m.sigma)

        # Independent cross-check: resistance from ohmic loss P = 1/2 integral sigma |E|^2 dV
        abs_e2 = omega ** 2 * (gfu_cond * Conj(gfu_cond)).real
        p_ohmic = Integrate(0.5 * sigma_cf * abs_e2 * 2 * np.pi * rho, mesh)
        r_check = p_ohmic / (0.5 * I_source ** 2)

        z_norm = (z0 - z_cond) / z0.imag if z0.imag != 0 else 0.0

        return {
            "mesh": mesh,
            "V": V,
            "solution": gfu_cond,
            "solution_air": gfu_air,
            "impedance": z_cond,
            "z0": z0,
            "impedance_normalized": z_norm,
            "resistance_check": r_check,
        }

    # -- Post-processing helpers --------------------------------------------

    def impedance_vs_sigma(self, sigma_array) -> np.ndarray:
        """Compute the normalised impedance for several conductivities.

        Reuses one mesh and the free-space (sigma=0) reference solve, then
        re-solves the conductor problem for each conductivity.
        """
        self._check_ngsolve()
        from ngsolve import (
            H1, BilinearForm, LinearForm, GridFunction, Integrate,
            grad, dx, x as rho,
        )

        c = self.coil
        omega = c.omega
        mesh = self._build_mesh()
        V = H1(mesh, order=self.order, complex=True, dirichlet="axis|outer")

        S_coil = (c.r_outer - c.r_inner) * c.length
        I_source = 1.0
        J0 = c.n_turns * I_source / S_coil
        coil_cf = mesh.MaterialCF({"coil": 1.0}, default=0.0)

        def solve_for_sigma(sigma_value):
            sigma_cf = mesh.MaterialCF({"conductor": sigma_value}, default=0.0)
            A = V.TrialFunction()
            v = V.TestFunction()
            a = BilinearForm(V, symmetric=True, check_unused=False)
            a += (1.0 / mu0) * (grad(A) * grad(v) + A * v / rho ** 2) * rho * dx
            a += 1j * omega * sigma_cf * A * v * rho * dx
            a.Assemble()
            f = LinearForm(V)
            f += J0 * coil_cf * v * rho * dx
            f.Assemble()
            gfu = GridFunction(V)
            gfu.vec.data = a.mat.Inverse(V.FreeDofs(), inverse="umfpack") * f.vec
            lam = (c.n_turns / S_coil) * 2 * np.pi * Integrate(coil_cf * rho * gfu, mesh)
            return 1j * omega * lam / I_source

        z0 = solve_for_sigma(0.0)
        out = np.empty(len(sigma_array), dtype=complex)
        for i, sig in enumerate(sigma_array):
            z_cond = solve_for_sigma(float(sig))
            out[i] = (z0 - z_cond) / z0.imag
        return out

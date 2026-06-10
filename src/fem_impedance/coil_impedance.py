"""
FEM-based coil impedance computation using NGSolve.

Extends the thesis's current_loop2D.py (single current loop) to compute
the impedance of a multi-turn cylindrical coil by integrating the electric
field E_phi over the coil cross-section, following Eq. 4.53 from the thesis:

    V = -(2π n) / [l (r_o - r_i)] ∫∫ ρ E_phi(ρ, z) dρ dz
    Z = V / I

This enables handling geometries where the analytical Dodd-Deeds model
breaks down (finite-thickness plates, layered media, defects).
"""

import numpy as np
from dataclasses import dataclass

from ..utils.common import ECTCoilParams, MaterialParams, mu0


@dataclass
class FEMCoilImpedance:
    """Compute coil impedance from a 2D axisymmetric FEM solution.

    This class sets up the NGSolve problem, solves for E_phi, and integrates
    over the coil cross-section to obtain impedance.  It requires NGSolve to
    be installed (optional dependency).
    """

    coil: ECTCoilParams
    material: MaterialParams
    domain_r_max: float = 0.05
    domain_z_top: float = 0.03
    domain_z_bot: float = -0.02
    gauss_sigma: float = 1e-4
    max_mesh_size: float = 0.005

    def _check_ngsolve(self):
        try:
            import ngsolve  # noqa: F401
            return True
        except ImportError:
            raise ImportError(
                "NGSolve is required for FEM impedance computation. "
                "Install it via: pip install ngsolve"
            )

    def setup_and_solve(self) -> dict:
        """Build the FEM model, solve, and return the solution + mesh.

        Returns:
            Dictionary with keys: 'mesh', 'solution', 'V' (FE space),
            'impedance', 'impedance_normalized'.
        """
        self._check_ngsolve()

        from netgen.geom2d import SplineGeometry
        from ngsolve import (
            Mesh, H1, BilinearForm, LinearForm, GridFunction,
            CoefficientFunction, Integrate, dx,
            grad, exp, sqrt,
        )

        c = self.coil
        m = self.material
        omega = c.omega
        sig = self.gauss_sigma

        delta = m.skin_depth(omega)
        mesh_size_skin = delta / 3.0
        skin_layer = 4 * delta

        # -- Geometry: three stacked regions in the rho-z plane --
        geo = SplineGeometry()

        # Conductor bulk (below skin layer)
        geo.AddRectangle(
            p1=(0, self.domain_z_bot),
            p2=(self.domain_r_max, -skin_layer),
            leftdomain=1, rightdomain=0,
            bcs=("outer", "outer", "default", "axis"),
            maxh=self.max_mesh_size,
        )

        # Conductor skin layer (refined)
        geo.AddRectangle(
            p1=(0, -skin_layer),
            p2=(self.domain_r_max, 0),
            leftdomain=1, rightdomain=0,
            bcs=("default", "outer", "interface_cond", "axis"),
            maxh=mesh_size_skin,
        )

        # Air gap (between conductor and coil)
        geo.AddRectangle(
            p1=(0, 0),
            p2=(self.domain_r_max, c.liftoff),
            leftdomain=2, rightdomain=0,
            bcs=("interface_cond", "outer", "interface_coil", "axis"),
            maxh=0.001,
        )

        # Air above coil
        geo.AddRectangle(
            p1=(0, c.liftoff),
            p2=(self.domain_r_max, self.domain_z_top),
            leftdomain=3, rightdomain=0,
            bcs=("interface_coil", "outer", "outer", "axis"),
            maxh=self.max_mesh_size,
        )

        # Refine around the coil source location
        r_coil = c.r_mean
        z_coil = c.liftoff + c.length / 2
        refine_radius = 3 * sig
        geo.AddCircle(
            c=(r_coil, z_coil),
            r=refine_radius,
            leftdomain=0, rightdomain=0,
            maxh=sig,
        )

        geo.SetMaterial(1, "conductor")
        geo.SetMaterial(2, "air_gap")
        geo.SetMaterial(3, "air_above")

        mesh = Mesh(geo.GenerateMesh(maxh=self.max_mesh_size))

        # -- FE space and bilinear form --
        V = H1(mesh, order=2, complex=True)
        E_trial = V.TrialFunction()
        v = V.TestFunction()

        # NGSolve uses x,y for the 2D mesh — here x=rho, y=z
        from ngsolve import x as rho, y as z

        sigma_cf = mesh.MaterialCF({
            "conductor": m.sigma,
            "air_gap": 0,
            "air_above": 0,
        })
        k2 = 1j * omega * m.mu * sigma_cf

        # Axisymmetric weak form: -∫(∇E·∇v + (1/ρ²)Ev + k²Ev) ρ dρdz = ∫ f·v ρ dρdz
        a = BilinearForm(V, symmetric=True)
        a += (-grad(E_trial) * grad(v) - (1 / rho ** 2) * E_trial * v
              - k2 * E_trial * v) * rho * dx
        a.Assemble()

        # Source: Gaussian-regularised multi-turn coil
        # The coil extends from r_inner to r_outer and from liftoff to liftoff+length.
        # We model it as a superposition spread over the cross-section.
        r_c = (c.r_inner + c.r_outer) / 2
        z_c = c.liftoff + c.length / 2
        g = exp(-((rho - r_c) ** 2 + (z - z_c) ** 2) / (2 * sig ** 2))
        int_g = Integrate(g * rho, mesh)
        g_norm = g / int_g

        I_source = 1e-3  # 1 mA reference current

        f = LinearForm(V)
        f += (1j * omega * m.mu * I_source * r_c) * g_norm * v * rho * dx
        f.Assemble()

        # -- Solve --
        gfu = GridFunction(V)
        gfu.vec.data = a.mat.Inverse(V.FreeDofs()) * f.vec

        # -- Compute coil impedance by integration over coil cross-section --
        # V = -(2π n) / [l(r_o - r_i)] ∫∫ ρ E_phi dρ dz  over coil region
        # We approximate by sampling E_phi on a grid over the coil cross-section.
        impedance = self._compute_impedance_from_field(
            gfu, mesh, c, I_source
        )

        # Normalised impedance
        z0 = self._compute_self_impedance(c, I_source)
        z_norm = (z0 - impedance) / z0.imag if z0.imag != 0 else 0

        return {
            "mesh": mesh,
            "solution": gfu,
            "V": V,
            "impedance": impedance,
            "impedance_normalized": z_norm,
            "z0": z0,
        }

    def _compute_impedance_from_field(
        self, gfu, mesh, coil: ECTCoilParams, I_source: float,
        n_rho: int = 30, n_z: int = 30,
    ) -> complex:
        """Integrate E_phi over the coil cross-section to get impedance.

        Uses numerical quadrature over a regular grid in the coil region.
        Z = V/I = -(2π n) / [l(r_o - r_i) I] ∫∫ ρ E_phi(ρ,z) dρ dz
        """
        rho_vals = np.linspace(coil.r_inner, coil.r_outer, n_rho)
        z_vals = np.linspace(coil.liftoff, coil.liftoff + coil.length, n_z)

        d_rho = (coil.r_outer - coil.r_inner) / (n_rho - 1)
        d_z = coil.length / (n_z - 1)

        integral = 0.0 + 0.0j
        for rr in rho_vals:
            for zz in z_vals:
                try:
                    e_val = complex(gfu(mesh(rr, zz)))
                    integral += rr * e_val * d_rho * d_z
                except Exception:
                    pass

        voltage = -2 * np.pi * coil.n_turns / (
            coil.length * (coil.r_outer - coil.r_inner)
        ) * integral

        impedance = voltage / I_source
        return impedance

    def _compute_self_impedance(
        self, coil: ECTCoilParams, I_source: float
    ) -> complex:
        """Compute self-impedance analytically (coil in free space).

        Uses the Dodd-Deeds Z0 formula as the FEM self-impedance reference.
        """
        from ..inversion.dodd_deeds import DoddDeedsModel
        model = DoddDeedsModel(coil)
        return model.z0()

    def evaluate_field_on_line(
        self, gfu, mesh, rho_val: float,
        z_min: float = None, z_max: float = None, n_points: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample E_phi along a vertical line for plotting/comparison."""
        z_min = z_min or self.domain_z_bot
        z_max = z_max or self.domain_z_top
        z_vals = np.linspace(z_min, z_max, n_points)
        e_vals = np.array([complex(gfu(mesh(rho_val, z))) for z in z_vals])
        return z_vals, e_vals

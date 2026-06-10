"""
Sensitivity kernel of the axisymmetric ECT coil.

By driving-point reciprocity, a small conductivity perturbation delta_sigma(x)
changes the coil impedance by

    delta_Z = -(1 / I^2) integral  delta_sigma(x)  E(x) . E(x)  dV

where E is the electric field driven by the coil current I.  For the
axisymmetric coil solved in coil_impedance.py the vector potential has only an
azimuthal component, A = A_phi phi_hat, so the electric field is purely
azimuthal as well:

    E = -j w A_phi phi_hat        =>   E . E = (E_phi)^2 = -w^2 A_phi^2

Two consequences, central to the tensor-strain question:

1. The sensitivity density is  J(rho, z) = E_phi^2  -- the measurement is
   weighted towards the region where the induced field is strongest (just under
   the coil, within a skin depth of the surface).

2. Only the *azimuthal* (local in-plane, circumferential) component of a
   conductivity-tensor perturbation is sensed.  E_rho = E_z = 0 identically, so
   the kernel J_ij = E_i E_j has only a phi-phi entry.  A single normal coil is
   therefore blind to sigma_zz and to currents in the rho/z directions -- which
   is exactly why directional probes are needed to resolve the full tensor.

This module computes the kernel from a solved field and validates the
reciprocity relation against finite-difference re-solves on the same mesh.
"""

import numpy as np

from ..utils.common import ECTCoilParams, MaterialParams
from .coil_impedance import FEMCoilImpedance


def validate_reciprocity(coil: ECTCoilParams, material: MaterialParams,
                         dsigma_frac: float = 0.01) -> dict:
    """Check the reciprocity sensitivity kernel against finite differences.

    Perturbs the conductor conductivity by ``dsigma_frac`` and compares the
    resulting impedance change with the reciprocity prediction
    delta_Z = -(1/I^2) integral delta_sigma E.E dV.

    Returns a dict with the finite-difference and reciprocity impedance changes
    and their ratio (which should be ~1).
    """
    from ngsolve import Integrate, x as rho

    fem = FEMCoilImpedance(coil=coil, material=material)
    sig0 = material.sigma
    dsig = dsigma_frac * sig0

    out = fem.solve_fields([sig0, sig0 + dsig])
    mesh = out["mesh"]
    I_source = out["I_source"]
    base, pert = out["results"][0], out["results"][1]

    dZ_fd = pert["Z"] - base["Z"]

    omega = coil.omega
    A = base["gfu"]
    # E.E = (E_phi)^2 = -w^2 A^2 (azimuthal component only)
    EE = -(omega ** 2) * A * A
    conductor = mesh.MaterialCF({"conductor": 1.0}, default=0.0)
    # delta_Z = -(1/I^2) integral delta_sigma E.E dV,  dV = 2 pi rho drho dz
    dZ_recip = -(1.0 / I_source ** 2) * dsig * Integrate(conductor * EE * 2 * np.pi * rho, mesh)

    ratio = dZ_fd / dZ_recip if dZ_recip != 0 else np.nan
    return {
        "delta_Z_fd": complex(dZ_fd),
        "delta_Z_reciprocity": complex(dZ_recip),
        "ratio": complex(ratio),
        "dsigma": dsig,
    }


def sensitivity_grid(coil: ECTCoilParams, material: MaterialParams,
                     n_rho: int = 140, n_z: int = 140,
                     rho_max: float = None, z_min: float = None,
                     z_max: float = None) -> tuple:
    """Sample the sensitivity density |E_phi|^2 over the (rho, z) plane.

    The sampling window defaults to the full meshed domain; pass ``rho_max``,
    ``z_min``, ``z_max`` to zoom (e.g. onto the near-surface specimen region
    where the eddy-current sensitivity actually lives).

    Returns (rho_axis, z_axis, kernel) where ``kernel`` is normalised to a
    maximum of 1 within the window and NaN outside the meshed domain.
    """
    fem = FEMCoilImpedance(coil=coil, material=material)
    out = fem.solve_fields([material.sigma])
    mesh = out["mesh"]
    A = out["results"][0]["gfu"]
    omega = coil.omega

    rho_max = fem.domain_r_max if rho_max is None else rho_max
    z_min = fem.domain_z_bot if z_min is None else z_min
    z_max = fem.domain_z_top if z_max is None else z_max
    rho_axis = np.linspace(0.0, rho_max, n_rho)
    z_axis = np.linspace(z_min, z_max, n_z)
    kernel = np.full((n_z, n_rho), np.nan)
    for iz, zz in enumerate(z_axis):
        for ir, rr in enumerate(rho_axis):
            try:
                a_val = complex(A(mesh(rr, zz)))
                kernel[iz, ir] = (omega ** 2) * abs(a_val) ** 2
            except Exception:
                pass

    finite = np.isfinite(kernel)
    if finite.any():
        kmax = np.nanmax(kernel)
        if kmax > 0:
            kernel = kernel / kmax
    return rho_axis, z_axis, kernel

"""Shared data structures and constants for the ECT simulation framework."""

import numpy as np
from dataclasses import dataclass, field

mu0 = 4 * np.pi * 1e-7  # Vacuum permeability [H/m]


@dataclass
class ECTCoilParams:
    """Parameters defining a cylindrical multi-turn ECT coil."""
    r_inner: float       # Inner radius [m]
    r_outer: float       # Outer radius [m]
    length: float        # Axial length [m]
    n_turns: int         # Number of turns
    liftoff: float       # Distance from conductor surface [m]
    frequency: float     # Excitation frequency [Hz]

    @property
    def omega(self) -> float:
        return 2 * np.pi * self.frequency

    @property
    def r_mean(self) -> float:
        return 0.5 * (self.r_inner + self.r_outer)


@dataclass
class MaterialParams:
    """Electrical and magnetic properties of the test specimen."""
    sigma: float         # Electrical conductivity [S/m]
    mu_r: float = 1.0    # Relative permeability (1.0 for non-ferromagnetic)

    @property
    def mu(self) -> float:
        return self.mu_r * mu0

    def skin_depth(self, omega: float) -> float:
        return np.sqrt(2.0 / (omega * self.mu * self.sigma))

    def with_strain(self, strain: float, kappa: float) -> "MaterialParams":
        """Return new MaterialParams with conductivity modified by strain."""
        sigma_new = self.sigma * (1.0 + kappa * strain)
        return MaterialParams(sigma=sigma_new, mu_r=self.mu_r)

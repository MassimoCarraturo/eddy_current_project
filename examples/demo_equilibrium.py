"""
Demo: recovering the out-of-plane strain epsilon_zz by equilibrium.

Eddy-current probes sense only in-plane conductivity, so an in-plane rosette
cannot observe epsilon_zz. On a traction-free surface, mechanical equilibrium
forces plane stress, giving the exact closure
    epsilon_zz = -nu/(1-nu) (epsilon_xx + epsilon_yy).
Stacking this with the measurement map lets the in-plane rosette resolve all six
strain components. This demo contrasts the rosette alone (epsilon_zz follows a
biased prior) with the equilibrium-constrained inversion (epsilon_zz recovered).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.tensor_model import (ElastoResistivityModel, direction,
                              EquilibriumConstrainedInverter)

COMP = ["xx", "yy", "zz", "yz", "xz", "xy"]
NU = 0.30


def main():
    print("=" * 60)
    print("Demo: out-of-plane strain recovery by equilibrium")
    print("=" * 60)

    model = ElastoResistivityModel(kappa_long=-0.40, kappa_trans=-0.08)
    rng = np.random.default_rng(7)

    # Laterally homogeneous build column -> plane stress at every depth.
    z = np.linspace(0, 1, 40)
    exx, eyy = -0.003 + 0.004 * z, -0.002 + 0.003 * z
    exy = 0.0015 * np.sin(np.pi * z)
    ezz = -NU / (1 - NU) * (exx + eyy)
    eps = np.zeros((40, 6))
    eps[:, 0], eps[:, 1], eps[:, 2], eps[:, 5] = exx, eyy, ezz, exy
    prior = 0.85 * eps + np.array([0.001, 0.001, 0.0015, 0, 0, 0])

    dirs = [direction(a) for a in (0, 60, 120)]
    inv_eq = EquilibriumConstrainedInverter(model, dirs, poisson=NU,
                                            prior_weight=1e-4, constraint_weight=1e6)
    inv_do = EquilibriumConstrainedInverter(model, dirs, poisson=NU,
                                            prior_weight=1e-4, constraint_weight=0.0)
    Y = np.array([inv_eq.forward(eps[i], noise_std=1e-4, rng=rng) for i in range(40)])
    rec_eq = inv_eq.invert_field(Y, prior)
    rec_do = inv_do.invert_field(Y, prior)

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    print(f"\n  epsilon_zz resolution  rosette only = {inv_do.resolved_fraction()[2]:.2f}"
          f"   + equilibrium = {inv_eq.resolved_fraction()[2]:.2f}")
    print(f"  epsilon_zz RMSE        rosette only = {rmse(rec_do[:,2], eps[:,2]):.2e}"
          f"   + equilibrium = {rmse(rec_eq[:,2], eps[:,2]):.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(z, eps[:, 2] * 100, "k-", lw=2, label="true")
    ax.plot(z, prior[:, 2] * 100, color="0.6", ls="--", label="prior")
    ax.plot(z, rec_do[:, 2] * 100, "x", color="#d55e00", ms=5, label="rosette only")
    ax.plot(z, rec_eq[:, 2] * 100, ".", color="#0072b2", ms=7, label="rosette + equil.")
    ax.set_xlabel("build height (norm.)"); ax.set_ylabel(r"$\varepsilon_{zz}$ (%)")
    ax.set_title("Out-of-plane strain recovery"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    x = np.arange(6)
    ax.bar(x - 0.2, inv_do.resolved_fraction(), 0.4, color="#d55e00", label="rosette only")
    ax.bar(x + 0.2, inv_eq.resolved_fraction(), 0.4, color="#0072b2", label="+ equilibrium")
    ax.set_xticks(x); ax.set_xticklabels(COMP); ax.set_ylim(0, 1.1)
    ax.set_ylabel("data-resolved fraction")
    ax.set_title("Resolution gain"); ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("equilibrium_demo.png", dpi=150)
    plt.show()
    print("\nSaved: equilibrium_demo.png")


if __name__ == "__main__":
    main()

"""
Demo: Tensorial strain-conductivity (elastoresistivity) model -- Step 1.

Shows how the scalar law  delta_sigma/sigma0 = kappa * eps  generalises to the
tensor law  (delta_sigma/sigma0)_ij = K_ijkl eps_kl, and what that implies:

  * a general 3-D strain produces an anisotropic conductivity tensor;
  * the conductivity change sensed by a probe depends on its direction
    (the "conductivity rosette"), so a single scalar reading is ambiguous;
  * the thesis scalar kappa = -0.326 is one directional projection of the
    tensor model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.tensor_model import ElastoResistivityModel


def main():
    print("=" * 60)
    print("Demo: Tensorial Elastoresistivity Model (Step 1)")
    print("=" * 60)

    # Two-constant isotropic model. (Illustrative values; calibrate with
    # multi-orientation tensile tests -- see effective_scalar_kappa below.)
    kappa_long, kappa_trans = -0.40, -0.08
    poisson = 0.30
    model = ElastoResistivityModel(kappa_long=kappa_long, kappa_trans=kappa_trans)
    sigma0 = 17.7e6  # S/m, 316L

    print(f"\nElastoresistivity constants: kappa_L={kappa_long}, kappa_T={kappa_trans}")

    # --- 1. A general 3-D strain -> anisotropic conductivity tensor ---
    eps = np.array([
        [0.010, 0.002, 0.000],
        [0.002, -0.003, 0.001],
        [0.000, 0.001, 0.004],
    ])
    print("\n--- General strain tensor (strain) ---")
    print(eps)

    dsig = model.delta_sigma_ratio(eps)
    print("\n--- Induced (delta_sigma / sigma0) tensor ---")
    print(dsig)

    sigma = model.conductivity_tensor(sigma0, eps)
    vals, vecs = model.principal_conductivities(sigma0, eps)
    print("\nPrincipal conductivities (MS/m):", np.round(vals / 1e6, 4))
    anisotropy = (vals.max() - vals.min()) / sigma0 * 100
    print(f"Strain-induced conductivity anisotropy: {anisotropy:.3f}%")

    # --- 2. Reduction to the scalar regime ---
    print("\n--- Reduction to scalar kappa (uniaxial-stress test) ---")
    k_par = model.effective_scalar_kappa([1, 0, 0], [1, 0, 0], poisson)
    k_perp = model.effective_scalar_kappa([1, 0, 0], [0, 1, 0], poisson)
    print(f"  kappa_eff sensed ALONG load  = {k_par:.4f}")
    print(f"  kappa_eff sensed TRANSVERSE  = {k_perp:.4f}")
    print("  -> the same uniaxial strain yields different kappa per direction;")
    print("     the thesis scalar (-0.326) is one such projection.")

    # --- 3. The conductivity rosette: kappa_eff vs sensing angle ---
    angles = np.linspace(0, 180, 181)
    load = np.array([1.0, 0.0, 0.0])
    eps_uni = (1 + poisson) * np.outer(load, load) - poisson * np.eye(3)
    dsig_uni = model.delta_sigma_ratio(eps_uni)
    kappa_theta = np.array([
        (np.array([np.cos(np.radians(a)), np.sin(np.radians(a)), 0.0])
         @ dsig_uni
         @ np.array([np.cos(np.radians(a)), np.sin(np.radians(a)), 0.0]))
        for a in angles
    ])

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Tensorial Elastoresistivity Model", fontsize=14)

    ax = axes[0]
    ax.plot(angles, kappa_theta, "b-", lw=2)
    ax.axhline(k_par, color="g", ls="--", lw=1, label=f"along load = {k_par:.3f}")
    ax.axhline(k_perp, color="r", ls="--", lw=1, label=f"transverse = {k_perp:.3f}")
    ax.set_xlabel("Sensing direction angle from load axis (deg)")
    ax.set_ylabel(r"Effective $\kappa$ (sensed $\Delta\sigma/\sigma_0$)")
    ax.set_title("Conductivity rosette (uniaxial load)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    im = ax.imshow(dsig, cmap="RdBu_r",
                   vmin=-np.abs(dsig).max(), vmax=np.abs(dsig).max())
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(["x", "y", "z"]); ax.set_yticklabels(["x", "y", "z"])
    ax.set_title(r"$(\Delta\sigma/\sigma_0)_{ij}$ for the general strain")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{dsig[i, j]:.4f}", ha="center", va="center",
                    color="k", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig("tensor_model_demo.png", dpi=150)
    plt.show()
    print("\nSaved: tensor_model_demo.png")


if __name__ == "__main__":
    main()

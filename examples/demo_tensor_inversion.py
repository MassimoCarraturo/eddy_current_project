"""
Demo: Regularised tensor-strain inversion with a simulation prior -- Step 4.

Reconstructs a full strain-tensor profile along the build height from
directional ECT measurements, fusing them with a (biased) process-simulation
prior. Demonstrates the data-assimilation principle:

  * a full probe set lets the ECT data override the biased prior for every
    component;
  * an in-plane rosette resolves the in-plane components from data and falls
    back to the prior for the out-of-plane ones (sigma_zz, sigma_xz, sigma_yz);
  * the resolution matrix quantifies, per component, how much comes from data
    versus the prior.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.tensor_model import (ElastoResistivityModel, direction,
                              optimize_probe_set, TensorStrainInverter, to_voigt)

COMP = ["xx", "yy", "zz", "yz", "xz", "xy"]


def build_strain_profile(n=40):
    """Synthetic residual-strain tensor field along normalised build height."""
    z = np.linspace(0.0, 1.0, n)
    eps = np.zeros((n, 3, 3))
    eps[:, 0, 0] = -0.004 + 0.002 * z          # xx
    eps[:, 1, 1] = -0.003 + 0.0015 * z         # yy
    eps[:, 2, 2] = 0.001 + 0.006 * z           # zz (grows with height)
    eps[:, 0, 1] = eps[:, 1, 0] = 0.002 * np.sin(np.pi * z)   # xy
    eps[:, 0, 2] = eps[:, 2, 0] = 0.0015 * z   # xz
    eps[:, 1, 2] = eps[:, 2, 1] = -0.001 * z   # yz
    return z, eps


def main():
    print("=" * 60)
    print("Demo: Tensor-Strain Inversion with Simulation Prior (Step 4)")
    print("=" * 60)

    model = ElastoResistivityModel(kappa_long=-0.40, kappa_trans=-0.08)
    rng = np.random.default_rng(1)

    z, eps_true = build_strain_profile(40)
    eps_true_v = np.array([to_voigt(e) for e in eps_true])

    # Biased "simulation" prior: right trend, wrong magnitude + offset.
    eps_prior_v = 0.85 * eps_true_v + np.array([0.001, 0.001, 0.0015, 0, 0.0008, 0])

    # Two probe configurations.
    full = optimize_probe_set(model, n_probes=6)
    dirs_full = full["directions"]
    dirs_rosette = [direction(a) for a in (0, 60, 120)]

    noise = 1e-4
    inv_full = TensorStrainInverter(model, dirs_full, prior_weight=1e-5)
    inv_ros = TensorStrainInverter(model, dirs_rosette, prior_weight=2e-3)

    # Forward-model measurements (+noise) and invert per point.
    Y_full = np.array([inv_full.forward(e, noise_std=noise, rng=rng) for e in eps_true])
    Y_ros = np.array([inv_ros.forward(e, noise_std=noise, rng=rng) for e in eps_true])
    rec_full = inv_full.invert_field(Y_full, eps_prior_v)
    rec_ros = inv_ros.invert_field(Y_ros, eps_prior_v)

    res_full = inv_full.data_resolved_fraction()
    res_ros = inv_ros.data_resolved_fraction()

    def rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    print("\n--- Data-resolved fraction (1 = data, 0 = prior) ---")
    print(f"  {'comp':4s} {'full':>8s} {'rosette':>8s}")
    for i, c in enumerate(COMP):
        print(f"  {c:4s} {res_full[i]:8.3f} {res_ros[i]:8.3f}")

    print("\n--- RMSE vs ground truth ---")
    print(f"  prior only      : {rmse(eps_prior_v, eps_true_v):.2e}")
    print(f"  full config     : {rmse(rec_full, eps_true_v):.2e}")
    print(f"  in-plane rosette: {rmse(rec_ros, eps_true_v):.2e}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Tensor-Strain Inversion with Simulation Prior (data assimilation)",
                 fontsize=14)

    show = [("xx", 0), ("zz", 2), ("xy", 5), ("xz", 4)]
    for ax, (name, idx) in zip(axes.ravel()[:4], show):
        ax.plot(eps_true_v[:, idx] * 100, z, "k-", lw=2, label="true")
        ax.plot(eps_prior_v[:, idx] * 100, z, "0.6", ls="--", lw=2, label="prior (sim)")
        ax.plot(rec_full[:, idx] * 100, z, "b.", ms=7, label="recovered: full")
        ax.plot(rec_ros[:, idx] * 100, z, "rx", ms=6, label="recovered: rosette")
        ax.set_xlabel(f"strain_{name} (%)"); ax.set_ylabel("build height (norm.)")
        ax.set_title(f"epsilon_{name}")
        ax.grid(True, alpha=0.3)
        if name == "xx":
            ax.legend(fontsize=8)

    # resolution diagonal
    ax = axes[1, 1]
    x = np.arange(6); w = 0.38
    ax.bar(x - w / 2, res_full, w, label="full", color="#46c")
    ax.bar(x + w / 2, res_ros, w, label="rosette", color="#c44")
    ax.set_xticks(x); ax.set_xticklabels(COMP)
    ax.set_ylabel("data-resolved fraction"); ax.set_ylim(0, 1.1)
    ax.set_title("Resolution: data vs prior")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    # RMSE summary
    ax = axes[1, 2]
    labels = ["prior\nonly", "full\nconfig", "in-plane\nrosette"]
    vals = [rmse(eps_prior_v, eps_true_v), rmse(rec_full, eps_true_v),
            rmse(rec_ros, eps_true_v)]
    ax.bar(labels, np.array(vals) * 100, color=["0.6", "#46c", "#c44"])
    ax.set_ylabel("strain RMSE (%)")
    ax.set_title("Reconstruction error")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("tensor_inversion_demo.png", dpi=150)
    plt.show()
    print("\nSaved: tensor_inversion_demo.png")


if __name__ == "__main__":
    main()

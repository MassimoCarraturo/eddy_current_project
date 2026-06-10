"""
Demo: tensor strain field from the UnifiedECTPipeline.

Simulates a layer-wise LPBF build probed by a directional ECT rosette, feeds the
directional measurements into the pipeline, and reconstructs the full strain
TENSOR field (fused with a process-simulation prior). Shows the derived
invariant fields (volumetric and von-Mises-equivalent strain) and the per-layer
component profiles -- the tensor generalisation of the scalar strain profile.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.pipeline import UnifiedECTPipeline
from src.utils.common import ECTCoilParams, MaterialParams
from src.tensor_model import (ElastoResistivityModel, optimize_probe_set,
                              TensorStrainInverter, to_voigt)


def synthetic_build(n_layers, n_x):
    """Ground-truth strain tensor field eps(layer, x) for the build."""
    z = np.linspace(0, 1, n_layers)
    xg = np.linspace(0, 1, n_x)
    eps = np.zeros((n_layers, n_x, 3, 3))
    for i, zz in enumerate(z):
        for j, xx in enumerate(xg):
            eps[i, j, 0, 0] = -0.004 + 0.003 * zz                  # xx
            eps[i, j, 1, 1] = -0.003 + 0.002 * zz                  # yy
            eps[i, j, 2, 2] = 0.001 + 0.006 * zz                   # zz
            eps[i, j, 0, 1] = eps[i, j, 1, 0] = 0.0025 * np.sin(np.pi * xx) * zz
            eps[i, j, 0, 2] = eps[i, j, 2, 0] = 0.0015 * zz
            eps[i, j, 1, 2] = eps[i, j, 2, 1] = -0.001 * zz
    return z, eps


def main():
    print("=" * 60)
    print("Demo: Tensor Strain Field from UnifiedECTPipeline")
    print("=" * 60)

    coil = ECTCoilParams(r_inner=0.535e-3, r_outer=1.31e-3, length=2.93e-3,
                         n_turns=235, liftoff=0.56e-3, frequency=240e3)
    material = MaterialParams(sigma=17.7e6)
    model = ElastoResistivityModel(kappa_long=-0.40, kappa_trans=-0.08)
    rng = np.random.default_rng(3)

    n_layers, n_x = 60, 40
    z, eps_true = synthetic_build(n_layers, n_x)

    # Full directional probe set (Step 3) and the measurement model (Step 2).
    dirs = optimize_probe_set(model, n_probes=6, n_restarts=10)["directions"]
    fwd = TensorStrainInverter(model, dirs, prior_weight=1e-6)
    noise = 1e-4

    # Biased "simulation" prior (per-layer, x-averaged): right trend, wrong scale.
    prior_layer = np.zeros((n_layers, 6))
    for i in range(n_layers):
        prior_layer[i] = 0.8 * to_voigt(eps_true[i].mean(axis=0)) + \
            np.array([0.001, 0.001, 0.0015, 0, 0.0008, 0])

    # Build the pipeline and feed directional measurements per layer.
    pipe = UnifiedECTPipeline(coil, material)
    for i in range(n_layers):
        meas = np.column_stack([
            fwd.forward(eps_true[i, j], noise_std=noise, rng=rng)
            for j in range(n_x)
        ])                                                 # (n_probes, n_x)
        pipe.add_directional_layer(i, meas)

    result = pipe.run_tensor_strain_prediction(
        model, dirs, prior_field=prior_layer, prior_weight=1e-5)

    print(f"\n  Reconstructed tensor field shape: {result.strain_voigt.shape}")
    print(f"  Probe-set condition number: {result.condition_number:.2f}")
    print(f"  Data-resolved fraction: {np.round(result.resolution, 2)}")

    eqv = result.equivalent_strain()        # (n_layers, n_x)
    vol = result.volumetric_strain()        # (n_layers, n_x)

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Tensor Strain Field from the Unified ECT Pipeline", fontsize=14)

    ax = axes[0, 0]
    im = ax.imshow(eqv * 100, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, 1, 0, 1])
    ax.set_xlabel("x (norm.)"); ax.set_ylabel("build height (norm.)")
    ax.set_title("Equivalent (von Mises) strain (%)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    im = ax.imshow(vol * 100, aspect="auto", origin="lower", cmap="RdBu_r",
                   extent=[0, 1, 0, 1])
    ax.set_xlabel("x (norm.)"); ax.set_ylabel("build height (norm.)")
    ax.set_title("Volumetric strain tr(eps) (%)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    for name in ("xx", "zz", "xz"):
        true_prof = np.array([to_voigt(eps_true[i].mean(axis=0))
                              [("xx", "yy", "zz", "yz", "xz", "xy").index(name)]
                              for i in range(n_layers)])
        ax.plot(true_prof * 100, z, lw=2, label=f"{name} true")
        ax.plot(result.component_profile(name) * 100, z, "--",
                label=f"{name} recovered")
    ax.set_xlabel("strain (%)"); ax.set_ylabel("build height (norm.)")
    ax.set_title("Component profiles (x-averaged)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    comps = result.component_names
    ax.bar(range(6), result.resolution, color="#46c")
    ax.set_xticks(range(6)); ax.set_xticklabels(comps)
    ax.set_ylim(0, 1.1); ax.set_ylabel("data-resolved fraction")
    ax.set_title(f"Resolution (cond={result.condition_number:.1f})")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("pipeline_tensor_demo.png", dpi=150)
    plt.show()
    print("\nSaved: pipeline_tensor_demo.png")


if __name__ == "__main__":
    main()

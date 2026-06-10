"""
Demo: Observability of the strain tensor from ECT -- Step 2.

Part A (FEM): computes the coil's sensitivity kernel from the validated
axisymmetric solver and confirms, against finite differences, the reciprocity
relation delta_Z = -(1/I^2) integral delta_sigma E.E dV. Because the field is
purely azimuthal, the kernel has only an in-plane component -> a normal coil is
blind to sigma_zz.

Part B (design tool): treats each directional probe as sensing n^T sigma n and
builds the linear map from the strain tensor to the measurement set. Its SVD
shows how many -- and which -- strain components each probe configuration can
resolve. This is the conductivity analogue of a strain-gauge rosette.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.tensor_model import (ElastoResistivityModel, ObservabilityAnalysis,
                              direction, NORMAL_COIL_ROW)
from src.utils.common import ECTCoilParams, MaterialParams

COMP = ["xx", "yy", "zz", "yz", "xz", "xy"]


def run_fem_sensitivity():
    """Return (rho, z, kernel) sensitivity map, or None if NGSolve absent."""
    try:
        from src.fem_impedance.sensitivity import validate_reciprocity, sensitivity_grid
    except ImportError:
        print("  [SKIP] NGSolve not installed -- FEM sensitivity skipped.")
        return None

    coil = ECTCoilParams(r_inner=0.00404, r_outer=0.01184, length=0.00802,
                         n_turns=1858, liftoff=0.001, frequency=240e3)
    material = MaterialParams(sigma=16.2e6)

    print("\n--- FEM sensitivity kernel (reciprocity) ---")
    val = validate_reciprocity(coil, material, dsigma_frac=0.005)
    print(f"  dZ (finite difference) = {val['delta_Z_fd']:.4e}")
    print(f"  dZ (reciprocity kernel)= {val['delta_Z_reciprocity']:.4e}")
    print(f"  ratio = {val['ratio']:.4f}  (-> 1 confirms the kernel)")
    print("  Field is purely azimuthal (E_rho = E_z = 0): the coil senses only")
    print("  the in-plane conductivity and is blind to sigma_zz.")
    # Sample only the specimen (z < 0); normalisation then reflects the
    # in-specimen sensitivity, which decays from the surface over a skin depth.
    return sensitivity_grid(coil, material, n_rho=160, n_z=160,
                            rho_max=0.025, z_min=-0.0015, z_max=-2e-5)


def main():
    print("=" * 60)
    print("Demo: Strain-Tensor Observability from ECT (Step 2)")
    print("=" * 60)

    kernel = run_fem_sensitivity()

    # --- Probe configurations ---
    configs = {
        "Single normal coil": ObservabilityAnalysis([NORMAL_COIL_ROW]),
        "In-plane rosette (0/45/90)": ObservabilityAnalysis.from_directions(
            [direction(0), direction(45), direction(90)]),
        "Rosette + tilted probes": ObservabilityAnalysis.from_directions(
            [direction(0), direction(45), direction(90),
             direction(0, 45), direction(90, 45), direction(45, 45)]),
    }

    model = ElastoResistivityModel(kappa_long=-0.40, kappa_trans=-0.08)

    print("\n--- Strain-tensor observability per configuration ---")
    spectra = {}
    for name, oa in configs.items():
        A = oa.strain_map(model, sigma0=1.0)
        res = oa.analyse(A)
        spectra[name] = res["singular_values"]
        blind = [dom for _, dom in oa.blind_components(A)]
        cond = res["condition_number"]
        cond_str = f"{cond:.2f}" if np.isfinite(cond) else "inf"
        print(f"\n  {name}")
        print(f"    resolvable strain DOF : {res['rank']} / 6")
        print(f"    condition number      : {cond_str}")
        print(f"    blind (null) dominated by: {blind}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Strain-Tensor Observability from ECT", fontsize=14)

    # (a) FEM sensitivity kernel
    ax = axes[0, 0]
    if kernel is not None:
        rho, z, K = kernel
        im = ax.pcolormesh(rho * 1e3, z * 1e3, K, cmap="inferno", shading="auto")
        ax.text(rho.max() * 1e3 * 0.97, z.max() * 1e3, " surface (z=0)",
                color="cyan", va="top", ha="right", fontsize=8)
        ax.set_xlabel("rho (mm)"); ax.set_ylabel("depth z (mm)")
        ax.set_title("FEM sensitivity in specimen |E_phi|^2\n"
                     "(decays from surface over a skin depth)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.text(0.5, 0.5, "NGSolve not installed\n(FEM kernel skipped)",
                ha="center", va="center")
        ax.set_axis_off()

    # (b) Singular value spectra
    ax = axes[0, 1]
    idx = np.arange(1, 7)
    for name, s in spectra.items():
        s_full = np.zeros(6)
        s_full[:len(s)] = s
        ax.plot(idx, s_full, "o-", label=name)
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value")
    ax.set_title("Observability spectra (strain map)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Strain-map matrix for the full config
    ax = axes[1, 0]
    full = configs["Rosette + tilted probes"]
    A_full = full.strain_map(model, sigma0=1.0)
    im = ax.imshow(np.abs(A_full), cmap="viridis", aspect="auto")
    ax.set_xticks(range(6)); ax.set_xticklabels(COMP)
    ax.set_yticks(range(A_full.shape[0]))
    ax.set_yticklabels([f"probe {i+1}" for i in range(A_full.shape[0])])
    ax.set_xlabel("strain component")
    ax.set_title("|strain -> measurement| map (full config)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (d) Resolvable DOF per config
    ax = axes[1, 1]
    names = list(configs.keys())
    ranks = [configs[n].analyse(configs[n].strain_map(model)).get("rank", 0)
             for n in names]
    bars = ax.barh(range(len(names)), ranks, color=["#c44", "#4a4", "#46c"])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("resolvable strain DOF (of 6)")
    ax.set_xlim(0, 6)
    ax.set_title("Resolving power of each configuration")
    for b, r in zip(bars, ranks):
        ax.text(r + 0.1, b.get_y() + b.get_height() / 2, str(r), va="center")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("observability_demo.png", dpi=150)
    plt.show()
    print("\nSaved: observability_demo.png")


if __name__ == "__main__":
    main()

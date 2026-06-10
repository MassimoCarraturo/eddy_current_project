"""
Demo: Probe-configuration ("rosette") design -- Step 3.

Finds probe sets that make the strain inversion well-posed and compares them:

  * standard in-plane rosettes (0/45/90, 0/60/120) vs an optimised one;
  * an optimised full 3-D probe set that resolves all six components;
  * the fundamental limit: a purely volumetric (scalar) elastoresistive
    response can never be resolved beyond one DOF, no matter how many probes;
  * (FEM) how excitation frequency tunes the sensing DEPTH -- the lever for
    out-of-plane / through-thickness information.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.tensor_model import (ElastoResistivityModel, direction,
                              evaluate_configuration, optimize_inplane_rosette,
                              optimize_probe_set, max_achievable_rank)
from src.utils.common import ECTCoilParams, MaterialParams


def fem_depth_profiles():
    """Mean sensitivity vs depth at two frequencies (or None if no NGSolve)."""
    try:
        from src.fem_impedance.sensitivity import sensitivity_grid
    except ImportError:
        return None
    material = MaterialParams(sigma=16.2e6)
    profiles = {}
    for f in (120e3, 480e3):
        coil = ECTCoilParams(r_inner=0.00404, r_outer=0.01184, length=0.00802,
                             n_turns=1858, liftoff=0.001, frequency=f)
        rho, z, K = sensitivity_grid(coil, material, n_rho=80, n_z=160,
                                     rho_max=0.02, z_min=-0.003, z_max=-2e-5)
        prof = np.nanmean(K, axis=1)
        prof = prof / np.nanmax(prof)
        profiles[f] = (z * 1e3, prof)
    return profiles


def main():
    print("=" * 60)
    print("Demo: Probe-Configuration (Rosette) Design (Step 3)")
    print("=" * 60)

    model = ElastoResistivityModel(kappa_long=-0.40, kappa_trans=-0.08)

    print(f"\nMax resolvable DOF -- in-plane: "
          f"{max_achievable_rank(model, allow_tilt=False)}, "
          f"with tilt: {max_achievable_rank(model, allow_tilt=True)}")

    # In-plane rosette comparison
    configs = {
        "0/45/90": [direction(a) for a in (0, 45, 90)],
        "0/60/120": [direction(a) for a in (0, 60, 120)],
    }
    print("\n--- In-plane rosettes ---")
    cond_vals, names = [], []
    for name, dirs in configs.items():
        m = evaluate_configuration(dirs, model)
        print(f"  {name:10s}: rank={m['rank']}  cond={m['condition_number']:.3f}")
        names.append(name); cond_vals.append(m["condition_number"])

    opt_in = optimize_inplane_rosette(model, n_probes=3)
    ang = np.sort(opt_in["angles_deg"].ravel())
    print(f"  optimised : angles={np.round(ang,1)} deg  "
          f"cond={opt_in['condition_number']:.3f}")
    names.append("optimised"); cond_vals.append(opt_in["condition_number"])

    # Full 3-D set
    print("\n--- Full 3-D probe set (out-of-plane access) ---")
    opt_full = optimize_probe_set(model, n_probes=6)
    print(f"  rank={opt_full['rank']}/6  cond={opt_full['condition_number']:.3f}")

    # Fundamental limit
    scalar_model = ElastoResistivityModel.from_scalar_kappa(-0.326)
    print("\n--- Fundamental limit ---")
    print(f"  purely volumetric response -> max resolvable DOF = "
          f"{max_achievable_rank(scalar_model, allow_tilt=True)} "
          f"(directional diversity cannot help)")

    profiles = fem_depth_profiles()

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Probe-Configuration Design", fontsize=14)

    # (a) condition numbers
    ax = axes[0, 0]
    colors = ["#c44", "#4a4", "#46c"]
    ax.bar(names, cond_vals, color=colors)
    ax.set_ylabel("condition number (strain map)")
    ax.set_title("In-plane rosette conditioning (lower = better)")
    for i, c in enumerate(cond_vals):
        ax.text(i, c + 0.02, f"{c:.2f}", ha="center")
    ax.grid(True, axis="y", alpha=0.3)

    # (b) optimised full probe directions (azimuth vs tilt)
    ax = axes[0, 1]
    angs = opt_full["angles_deg"]
    az, tilt = angs[:, 0], angs[:, 1]
    sc = ax.scatter(az, tilt, c=range(len(az)), cmap="viridis", s=120,
                    edgecolor="k", zorder=3)
    ax.axhline(90, color="gray", ls="--", lw=1, label="in-plane (tilt=90)")
    ax.set_xlabel("azimuth (deg)"); ax.set_ylabel("tilt from build axis (deg)")
    ax.set_title(f"Optimised 6-probe set (cond={opt_full['condition_number']:.2f})")
    ax.set_ylim(0, 100); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (c) singular value spectra: in-plane vs full
    ax = axes[1, 0]
    s_in = evaluate_configuration(opt_in["directions"], model)["singular_values"]
    s_full = opt_full["singular_values"]
    ax.plot(range(1, len(s_in) + 1), s_in, "o-", label="optimised in-plane (3)")
    ax.plot(range(1, len(s_full) + 1), s_full, "s-", label="optimised full (6)")
    ax.set_yscale("log")
    ax.set_xlabel("singular value index"); ax.set_ylabel("singular value")
    ax.set_title("Strain-map spectra")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (d) FEM depth sensitivity vs frequency
    ax = axes[1, 1]
    if profiles is not None:
        for f, (zmm, prof) in profiles.items():
            ax.plot(prof, zmm, lw=2, label=f"{f/1e3:.0f} kHz")
        ax.set_xlabel("normalised sensitivity"); ax.set_ylabel("depth z (mm)")
        ax.set_ylim(-1.0, 0.0)  # zoom onto the near-surface decay region
        ax.set_title("Frequency tunes sensing depth (FEM)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "NGSolve not installed\n(depth illustration skipped)",
                ha="center", va="center")
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig("probe_design_demo.png", dpi=150)
    plt.show()
    print("\nSaved: probe_design_demo.png")


if __name__ == "__main__":
    main()

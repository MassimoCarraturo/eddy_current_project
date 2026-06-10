"""
Demo: FEM-Based Coil Impedance Computation (Item 3)

Computes the impedance of a multi-turn coil above a conductive half-space
using NGSolve FEM, and compares against the analytical Dodd-Deeds solution.

NOTE: Requires NGSolve to be installed (pip install ngsolve).
If NGSolve is not available, the demo falls back to showing only the
analytical solution with a conductivity sweep.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.inversion import DoddDeedsModel
from src.utils.common import ECTCoilParams, MaterialParams


def run_analytical_demo(coil, material):
    """Demonstrate the analytical model with conductivity and frequency sweeps."""
    print("\n--- Analytical Dodd-Deeds Model ---")
    model = DoddDeedsModel(coil)

    z0 = model.z0()
    print(f"  Self-impedance Z0 = {z0:.6e}")

    z_total = model.z_total(material)
    print(f"  Total impedance   = {z_total:.6e}")

    z_norm = model.z_normalized(material)
    print(f"  Normalised Z      = {z_norm:.6e}")

    # Conductivity sweep
    print("\n  Running conductivity sweep...")
    sigma_vals = np.logspace(4, 8, 50)
    z_sweep = model.impedance_vs_sigma(sigma_vals)

    # Frequency sweep at fixed conductivity
    print("  Running frequency sweep...")
    freqs = np.logspace(3, 6, 40)
    z_freq = np.empty(len(freqs), dtype=complex)
    for i, f in enumerate(freqs):
        coil_f = ECTCoilParams(
            r_inner=coil.r_inner, r_outer=coil.r_outer,
            length=coil.length, n_turns=coil.n_turns,
            liftoff=coil.liftoff, frequency=f,
        )
        model_f = DoddDeedsModel(coil_f)
        z_freq[i] = model_f.z_normalized(material)

    return sigma_vals, z_sweep, freqs, z_freq


def run_fem_demo(coil, material):
    """Run the FEM impedance computation and compare with analytical."""
    try:
        from src.fem_impedance import FEMCoilImpedance
    except ImportError:
        print("\n  [SKIP] NGSolve not installed — FEM demo skipped.")
        return None

    print("\n--- FEM Coil Impedance (NGSolve) ---")
    fem = FEMCoilImpedance(coil=coil, material=material)

    try:
        result = fem.setup_and_solve()
        z_fem = result["impedance"]
        z_norm_fem = result["impedance_normalized"]
        z0 = result["z0"]

        print(f"  FEM impedance          = {z_fem:.6e}")
        print(f"  FEM normalised Z       = {z_norm_fem:.6e}")

        # Analytical comparison
        model = DoddDeedsModel(coil)
        z_norm_ana = model.z_normalized(material)
        print(f"  Analytical normalised Z = {z_norm_ana:.6e}")

        rel_err_re = abs(z_norm_fem.real - z_norm_ana.real) / abs(z_norm_ana.real) * 100
        rel_err_im = abs(z_norm_fem.imag - z_norm_ana.imag) / abs(z_norm_ana.imag) * 100
        print(f"  Relative error (Re): {rel_err_re:.2f}%")
        print(f"  Relative error (Im): {rel_err_im:.2f}%")

        return result
    except Exception as e:
        print(f"  [ERROR] FEM computation failed: {e}")
        return None


def main():
    print("=" * 60)
    print("Demo: FEM-Based Coil Impedance Computation")
    print("=" * 60)

    # Coil from thesis coil_analytical.py (Bowler reference coil)
    coil = ECTCoilParams(
        r_inner=0.00404,
        r_outer=0.01184,
        length=0.00802,
        n_turns=1858,
        liftoff=0.001,
        frequency=240e3,
    )

    material = MaterialParams(sigma=16.2e6)  # From coil_analytical.py

    # Run analytical demo
    sigma_vals, z_sweep, freqs, z_freq = run_analytical_demo(coil, material)

    # Attempt FEM demo
    fem_result = run_fem_demo(coil, material)

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Coil Impedance Analysis", fontsize=14)

    # Impedance trajectory (conductivity sweep)
    ax = axes[0, 0]
    ax.plot(z_sweep.real, z_sweep.imag, "b-", lw=1.5)
    n_labels = 8
    step = max(1, len(sigma_vals) // n_labels)
    for i in range(0, len(sigma_vals), step):
        ax.annotate(
            f"{sigma_vals[i]:.1e}",
            (z_sweep[i].real, z_sweep[i].imag),
            fontsize=7, textcoords="offset points", xytext=(5, 5),
        )
    ax.set_xlabel("Re(Z_norm)")
    ax.set_ylabel("Im(Z_norm)")
    ax.set_title("Impedance Trajectory (Conductivity Sweep)")
    ax.grid(True, alpha=0.3)

    # Re and Im vs conductivity
    ax = axes[0, 1]
    ax.semilogx(sigma_vals, z_sweep.real, "b-", lw=1.5, label="Re(Z)")
    ax.semilogx(sigma_vals, z_sweep.imag, "r-", lw=1.5, label="Im(Z)")
    ax.set_xlabel("Conductivity (S/m)")
    ax.set_ylabel("Normalised Impedance")
    ax.set_title("Impedance vs Conductivity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Frequency sweep
    ax = axes[1, 0]
    ax.plot(z_freq.real, z_freq.imag, "g-", lw=1.5)
    step = max(1, len(freqs) // 8)
    for i in range(0, len(freqs), step):
        ax.annotate(
            f"{freqs[i]/1e3:.0f}kHz",
            (z_freq[i].real, z_freq[i].imag),
            fontsize=7, textcoords="offset points", xytext=(5, 5),
        )
    ax.set_xlabel("Re(Z_norm)")
    ax.set_ylabel("Im(Z_norm)")
    ax.set_title("Impedance Trajectory (Frequency Sweep)")
    ax.grid(True, alpha=0.3)

    # FEM comparison or frequency components
    ax = axes[1, 1]
    if fem_result is not None:
        model = DoddDeedsModel(coil)
        z_ana = model.z_normalized(material)
        z_fem = fem_result["impedance_normalized"]
        labels = ["Re(Z)", "Im(Z)"]
        x = np.arange(2)
        width = 0.35
        ax.bar(x - width / 2, [z_ana.real, z_ana.imag], width, label="Analytical")
        ax.bar(x + width / 2, [z_fem.real, z_fem.imag], width, label="FEM")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("FEM vs Analytical Impedance")
        ax.legend()
    else:
        ax.semilogx(freqs, z_freq.real, "b-", lw=1.5, label="Re(Z)")
        ax.semilogx(freqs, z_freq.imag, "r-", lw=1.5, label="Im(Z)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalised Impedance")
        ax.set_title("Impedance vs Frequency")
        ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fem_impedance_demo.png", dpi=150)
    plt.show()
    print("\nSaved: fem_impedance_demo.png")


if __name__ == "__main__":
    main()

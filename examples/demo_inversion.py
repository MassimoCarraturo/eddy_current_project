"""
Demo: Optimisation-Based Impedance Inversion (Item 2)

Demonstrates the full inversion pipeline:
1. Use the Dodd-Deeds model to generate synthetic impedance data for a
   range of strains (forward problem).
2. Invert the impedance data back to conductivity using optimisation.
3. Estimate strain from the recovered conductivity.
4. Compare with the ground-truth strain values.

Uses the coil parameters from the thesis (Section 4.2, Table in whole_pipeline.py).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import time

from src.inversion import DoddDeedsModel, ImpedanceInverter
from src.utils.common import ECTCoilParams, MaterialParams


def main():
    print("=" * 60)
    print("Demo: Optimisation-Based Impedance Inversion")
    print("=" * 60)

    # Coil parameters from the thesis (whole_pipeline.py)
    coil = ECTCoilParams(
        r_inner=0.00107 / 2,
        r_outer=0.00262 / 2,
        length=0.00293,
        n_turns=235,
        liftoff=0.00056,
        frequency=240e3,
    )

    # Material: 316L stainless steel
    sigma_0 = 17.7e6  # S/m (reference conductivity from thesis)
    kappa = -0.326     # Material parameter from thesis (Section 5.2)
    material = MaterialParams(sigma=sigma_0)

    model = DoddDeedsModel(coil)

    # --- Step 1: Generate synthetic impedance data (forward problem) ---
    print("\n--- Forward model: generating synthetic impedance ---")
    strain_true = np.linspace(0, 0.02, 30)  # 0 to 2% strain
    sigma_true = sigma_0 * (1 + kappa * strain_true)

    z_data = np.empty(len(strain_true), dtype=complex)
    t0 = time.time()
    for i, sig in enumerate(sigma_true):
        mat = MaterialParams(sigma=sig)
        z_data[i] = model.z_normalized(mat)
    t_forward = time.time() - t0
    print(f"  Forward model: {len(strain_true)} points in {t_forward:.1f}s")

    # --- Step 2: Inversion (impedance → conductivity → strain) ---
    print("\n--- Inversion: impedance -> conductivity -> strain ---")
    inverter = ImpedanceInverter(
        coil,
        sigma_bounds=(0.8 * sigma_0, 1.2 * sigma_0),
    )

    t0 = time.time()
    sigma_inv, strain_inv = inverter.invert_to_strain(
        z_data, sigma_0, kappa, verbose=False
    )
    t_inv = time.time() - t0
    print(f"  Inversion: {len(z_data)} points in {t_inv:.1f}s")

    # --- Step 3: Error analysis ---
    sigma_err = np.abs(sigma_inv - sigma_true)
    sigma_rel_err = sigma_err / sigma_true * 100

    strain_err = np.abs(strain_inv - strain_true)
    # Avoid division by zero at strain=0
    strain_rel_err = np.where(
        strain_true > 1e-6,
        strain_err / strain_true * 100,
        0.0,
    )

    print(f"\n  Conductivity: max rel. error = {np.max(sigma_rel_err):.4f}%")
    print(f"  Conductivity: mean rel. error = {np.mean(sigma_rel_err):.4f}%")
    print(f"  Strain: max abs. error = {np.max(strain_err):.6f}")
    print(f"  Strain: RMSE = {np.sqrt(np.mean(strain_err**2)):.6f}")

    # --- Step 4: Visualisation ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Optimisation-Based Impedance Inversion", fontsize=14)

    # Impedance trajectory
    ax = axes[0, 0]
    ax.plot(z_data.real, z_data.imag, "b-o", markersize=3, label="Impedance locus")
    ax.set_xlabel("Re(Z_norm)")
    ax.set_ylabel("Im(Z_norm)")
    ax.set_title("Normalised Impedance Trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Conductivity recovery
    ax = axes[0, 1]
    ax.plot(strain_true * 100, sigma_true / 1e6, "b-", lw=2, label="True")
    ax.plot(strain_true * 100, sigma_inv / 1e6, "r--", lw=2, label="Inverted")
    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("Conductivity (MS/m)")
    ax.set_title("Conductivity Recovery")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Strain prediction
    ax = axes[1, 0]
    ax.plot(strain_true * 100, strain_true * 100, "b-", lw=2, label="True")
    ax.plot(strain_true * 100, strain_inv * 100, "r--", lw=2, label="Predicted")
    ax.set_xlabel("True Strain (%)")
    ax.set_ylabel("Predicted Strain (%)")
    ax.set_title("Strain Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Relative error
    ax = axes[1, 1]
    ax.plot(strain_true[1:] * 100, sigma_rel_err[1:], "g-", lw=2)
    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("Conductivity Inversion Error")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("inversion_demo.png", dpi=150)
    plt.show()
    print("\nSaved: inversion_demo.png")


if __name__ == "__main__":
    main()

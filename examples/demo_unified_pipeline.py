"""
Demo: Unified Geometry + Strain Pipeline (Item 4)

Simulates a simplified LPBF build process:
1. Generates synthetic layer-wise ECT data (impedance with embedded defects
   and strain-induced conductivity variation).
2. Runs the unified pipeline: geometry reconstruction + strain prediction.
3. Visualises the reconstructed geometry and strain profile.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.pipeline import UnifiedECTPipeline
from src.inversion import DoddDeedsModel
from src.utils.common import ECTCoilParams, MaterialParams


def generate_synthetic_build(
    n_layers=100,
    n_x=300,
    n_channels=6,
    sigma_0=17.7e6,
    kappa=-0.326,
):
    """Simulate ECT data from a multi-layer build with defects and strain.

    Returns layer-wise impedance data (real + imag) and ground truth.
    """
    coil = ECTCoilParams(
        r_inner=0.00107 / 2,
        r_outer=0.00262 / 2,
        length=0.00293,
        n_turns=235,
        liftoff=0.00056,
        frequency=240e3,
    )
    model = DoddDeedsModel(coil)

    # Ground truth strain profile: increases linearly with build height
    # (simulating residual stress accumulation)
    strain_per_layer = np.linspace(0, 0.005, n_layers)

    # Defect mask: rectangular void in layers 30-50, x positions 100-150
    defect_mask = np.zeros((n_layers, n_channels, n_x), dtype=bool)
    defect_mask[30:50, 2:4, 100:150] = True

    layers_data = {}
    z_ref = model.z_normalized(MaterialParams(sigma=sigma_0))

    for layer in range(n_layers):
        real_arr = np.zeros((n_channels, n_x))
        imag_arr = np.zeros((n_channels, n_x))

        # Base impedance varies with strain
        sigma_strained = sigma_0 * (1 + kappa * strain_per_layer[layer])
        z_layer = model.z_normalized(MaterialParams(sigma=sigma_strained))

        for ch in range(n_channels):
            # Solid region: impedance from strained material
            real_arr[ch, :] = z_layer.real
            imag_arr[ch, :] = z_layer.imag

            # Defect region: much larger impedance change (air conductivity)
            defect_x = defect_mask[layer, ch, :]
            if np.any(defect_x):
                # In void regions, the real part drops significantly
                real_arr[ch, defect_x] = z_ref.real * 0.3
                imag_arr[ch, defect_x] = z_ref.imag * 0.3

        # Add measurement noise
        noise_re = np.random.normal(0, abs(z_ref.real) * 0.001, real_arr.shape)
        noise_im = np.random.normal(0, abs(z_ref.imag) * 0.001, imag_arr.shape)
        real_arr += noise_re
        imag_arr += noise_im

        layers_data[layer] = {"real": real_arr, "imag": imag_arr}

    return coil, layers_data, strain_per_layer, defect_mask


def main():
    print("=" * 60)
    print("Demo: Unified Geometry + Strain Pipeline")
    print("=" * 60)

    np.random.seed(42)
    sigma_0 = 17.7e6
    kappa = -0.326

    # Generate synthetic data
    print("\n--- Generating synthetic LPBF build data ---")
    coil, layers_data, strain_true, defect_mask = generate_synthetic_build(
        n_layers=100, n_x=300, n_channels=6,
        sigma_0=sigma_0, kappa=kappa,
    )
    print(f"  Generated {len(layers_data)} layers, "
          f"{layers_data[0]['real'].shape[0]} channels, "
          f"{layers_data[0]['real'].shape[1]} x-points each")

    # Set up pipeline
    material = MaterialParams(sigma=sigma_0)
    pipeline = UnifiedECTPipeline(
        coil=coil,
        material=material,
        kappa=kappa,
        n_x_target=300,
        physical_dims={"x_mm": 150.0, "y_mm": 45.0, "z_mm": 40.0},
    )

    # Add all layers
    for idx, data in layers_data.items():
        pipeline.add_layer(idx, data["real"], data["imag"])

    # Run geometry reconstruction only (strain inversion is slow for 100 layers)
    print("\n--- Running geometry reconstruction ---")
    recon = pipeline.run_geometry_reconstruction()
    print(f"  Volume shape: {recon.binary_volume.shape}")
    print(f"  Spacing: x={recon.spacing['x_mm']:.4f}mm, "
          f"y={recon.spacing['y_mm']:.2f}mm, "
          f"z={recon.spacing['z_mm']:.4f}mm")
    print(f"  Threshold info: {recon.threshold_info}")

    # DICOM metadata
    full_result = pipeline.run(run_strain=False)
    dicom_meta = pipeline.export_dicom_metadata(full_result)
    print(f"\n  DICOM metadata:")
    for k, v in dicom_meta.items():
        print(f"    {k}: {v}")

    # --- Visualisation ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Unified ECT Pipeline: Geometry Reconstruction", fontsize=14)

    # Show cross-sections at different y-channels
    for ch in range(min(6, recon.binary_volume.shape[1])):
        row, col = divmod(ch, 3)
        ax = axes[row, col]
        slice_img = recon.binary_volume[:, ch, :]
        ax.imshow(slice_img, cmap="gray_r", aspect="auto",
                  extent=[0, 150, 40, 0])
        ax.set_title(f"Channel {ch + 1} (y = {ch * recon.spacing['y_mm']:.1f} mm)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")

    plt.tight_layout()
    plt.savefig("unified_pipeline_geometry.png", dpi=150)
    plt.show()
    print("\nSaved: unified_pipeline_geometry.png")

    # Show the ground-truth defect vs reconstructed for one channel
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.imshow(defect_mask[:, 3, :], cmap="gray_r", aspect="auto",
              extent=[0, 150, 40, 0])
    ax.set_title("Ground Truth Defect (Channel 4)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")

    ax = axes[1]
    ax.imshow(recon.binary_volume[:, 3, :], cmap="gray_r", aspect="auto",
              extent=[0, 150, 40, 0])
    ax.set_title("Reconstructed (Channel 4)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")

    plt.tight_layout()
    plt.savefig("unified_pipeline_defect_comparison.png", dpi=150)
    plt.show()
    print("Saved: unified_pipeline_defect_comparison.png")

    # Strain profile comparison (using true strain vs. build height)
    fig, ax = plt.subplots(figsize=(8, 5))
    z_positions = np.linspace(0, 40, len(strain_true))
    ax.plot(z_positions, strain_true * 100, "b-", lw=2, label="True strain")
    ax.set_xlabel("Build height z (mm)")
    ax.set_ylabel("Strain (%)")
    ax.set_title("Residual Strain Profile vs Build Height")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("unified_pipeline_strain_profile.png", dpi=150)
    plt.show()
    print("Saved: unified_pipeline_strain_profile.png")


if __name__ == "__main__":
    main()

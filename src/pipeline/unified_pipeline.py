"""
Unified ECT pipeline: geometry reconstruction + strain prediction.

Connects the two pillars of the thesis into a single workflow:

  Raw ECT data  ──►  Segmentation  ──►  3D DICOM  ──►  FCM-ready geometry
       │
       └──►  Impedance inversion  ──►  Δσ/σ₀  ──►  ε = Δσ/(σ₀·κ)

This pipeline takes layer-wise ECT impedance data and produces both
the reconstructed geometry and the estimated strain field.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from ..segmentation.auto_threshold import otsu_threshold, multi_method_segment
from ..segmentation.morphology import clean_binary_image
from ..inversion.dodd_deeds import DoddDeedsModel
from ..inversion.optimizer import ImpedanceInverter
from ..utils.common import ECTCoilParams, MaterialParams


@dataclass
class LayerData:
    """Data for a single layer of the build."""
    layer_index: int
    impedance_real: np.ndarray      # Raw real part of impedance (1D, n_x points)
    impedance_imag: np.ndarray      # Raw imaginary part of impedance
    binary_image: Optional[np.ndarray] = None
    conductivity_map: Optional[np.ndarray] = None
    strain_map: Optional[np.ndarray] = None


@dataclass
class ReconstructionResult:
    """Output of the geometry reconstruction stage."""
    binary_volume: np.ndarray       # 3D binary array (n_z, n_channels, n_x)
    spacing: dict                   # Voxel spacing in mm for each axis
    dimensions: dict                # Physical dimensions in mm
    threshold_info: dict            # Thresholding method and parameters used


@dataclass
class StrainResult:
    """Output of the strain prediction stage."""
    sigma_ref: float                # Reference conductivity
    kappa: float                    # Material parameter
    mean_conductivity: np.ndarray   # Per-layer mean conductivity
    mean_strain: np.ndarray         # Per-layer mean strain
    conductivity_maps: list         # Per-layer 1D conductivity arrays
    strain_maps: list               # Per-layer 1D strain arrays


@dataclass
class PipelineResult:
    """Combined output of the unified pipeline."""
    reconstruction: ReconstructionResult
    strain: Optional[StrainResult]
    n_layers: int
    n_channels: int
    n_x_points: int


class UnifiedECTPipeline:
    """Orchestrates the full ECT analysis pipeline.

    Usage:
        pipeline = UnifiedECTPipeline(coil_params, material_params, kappa=-0.326)
        pipeline.add_layer(layer_index, impedance_real, impedance_imag)
        ...  # add all layers
        result = pipeline.run()
    """

    def __init__(
        self,
        coil: ECTCoilParams,
        material: MaterialParams,
        kappa: float = -0.326,
        n_x_target: int = 300,
        physical_dims: dict = None,
    ):
        self.coil = coil
        self.material = material
        self.kappa = kappa
        self.n_x_target = n_x_target
        self.physical_dims = physical_dims or {
            "x_mm": 150.0, "y_mm": 45.0, "z_mm": 40.0
        }

        self.layers: dict[int, dict] = {}
        self.n_channels = 0

    def add_layer(
        self,
        layer_index: int,
        impedance_real: np.ndarray,
        impedance_imag: np.ndarray,
    ):
        """Add a layer's impedance data.

        Args:
            layer_index: Layer number (0-based).
            impedance_real: Real part, shape (n_x,) or (n_channels, n_x).
            impedance_imag: Imaginary part, same shape.
        """
        if impedance_real.ndim == 1:
            impedance_real = impedance_real[np.newaxis, :]
            impedance_imag = impedance_imag[np.newaxis, :]

        self.n_channels = max(self.n_channels, impedance_real.shape[0])
        self.layers[layer_index] = {
            "real": impedance_real,
            "imag": impedance_imag,
        }

    def _resample_to_grid(self, data: np.ndarray, n_target: int) -> np.ndarray:
        """Linearly interpolate data to a uniform grid of n_target points."""
        n_orig = data.shape[-1]
        x_orig = np.linspace(0, 1, n_orig)
        x_target = np.linspace(0, 1, n_target)
        if data.ndim == 1:
            return np.interp(x_target, x_orig, data)
        result = np.zeros((data.shape[0], n_target))
        for ch in range(data.shape[0]):
            result[ch] = np.interp(x_target, x_orig, data[ch])
        return result

    def _normalise_to_grayscale(self, data: np.ndarray) -> np.ndarray:
        """Normalise data to [0, 255] range for segmentation."""
        d_min, d_max = data.min(), data.max()
        if d_max - d_min < 1e-12:
            return np.zeros_like(data, dtype=np.float64)
        return (data - d_min) / (d_max - d_min) * 255.0

    def run_geometry_reconstruction(self) -> ReconstructionResult:
        """Stage 1: reconstruct 3D binary geometry from ECT impedance data."""
        sorted_indices = sorted(self.layers.keys())
        n_layers = len(sorted_indices)

        # Build per-channel 2D images (z × x) from the real part of impedance
        channel_images = []
        for ch in range(self.n_channels):
            image = np.zeros((n_layers, self.n_x_target))
            for row, idx in enumerate(sorted_indices):
                layer = self.layers[idx]
                real_data = layer["real"]
                if ch < real_data.shape[0]:
                    image[row] = self._resample_to_grid(real_data[ch], self.n_x_target)
            channel_images.append(image)

        # Segment each channel image
        binary_channels = []
        threshold_info = {}
        for ch, image in enumerate(channel_images):
            grayscale = self._normalise_to_grayscale(image)
            info, consensus = multi_method_segment(grayscale)
            cleaned = clean_binary_image(consensus, min_component_size=30)
            binary_channels.append(cleaned)
            threshold_info[f"channel_{ch}"] = {
                "otsu_threshold": info["otsu_threshold"],
                "median_threshold": info["median_threshold"],
            }

        # Stack channels into 3D volume (z, y_channels, x)
        binary_volume = np.stack(binary_channels, axis=1)

        # Compute spacing
        spacing = {
            "x_mm": self.physical_dims["x_mm"] / self.n_x_target,
            "y_mm": self.physical_dims["y_mm"] / max(self.n_channels, 1),
            "z_mm": self.physical_dims["z_mm"] / max(n_layers, 1),
        }

        return ReconstructionResult(
            binary_volume=binary_volume,
            spacing=spacing,
            dimensions=self.physical_dims.copy(),
            threshold_info=threshold_info,
        )

    def run_strain_prediction(self) -> StrainResult:
        """Stage 2: invert impedance to conductivity, then estimate strain."""
        inverter = ImpedanceInverter(
            self.coil,
            sigma_bounds=(0.5 * self.material.sigma, 2.0 * self.material.sigma),
        )

        sorted_indices = sorted(self.layers.keys())
        sigma_ref = self.material.sigma

        mean_sigmas = []
        mean_strains = []
        sigma_maps = []
        strain_maps = []

        for idx in sorted_indices:
            layer = self.layers[idx]
            real_data = layer["real"]
            imag_data = layer["imag"]

            # Average across channels
            real_avg = real_data.mean(axis=0) if real_data.ndim > 1 else real_data
            imag_avg = imag_data.mean(axis=0) if imag_data.ndim > 1 else imag_data

            z_meas = real_avg + 1j * imag_avg

            sigma_arr, strain_arr = inverter.invert_to_strain(
                z_meas, sigma_ref, self.kappa
            )

            mean_sigmas.append(np.mean(sigma_arr))
            mean_strains.append(np.mean(strain_arr))
            sigma_maps.append(sigma_arr)
            strain_maps.append(strain_arr)

        return StrainResult(
            sigma_ref=sigma_ref,
            kappa=self.kappa,
            mean_conductivity=np.array(mean_sigmas),
            mean_strain=np.array(mean_strains),
            conductivity_maps=sigma_maps,
            strain_maps=strain_maps,
        )

    def run(self, run_strain: bool = True) -> PipelineResult:
        """Execute the full pipeline.

        Args:
            run_strain: If True, also run the strain prediction stage.
                       Set to False for geometry-only reconstruction.
        """
        if not self.layers:
            raise ValueError("No layer data added. Call add_layer() first.")

        reconstruction = self.run_geometry_reconstruction()

        strain = None
        if run_strain:
            strain = self.run_strain_prediction()

        return PipelineResult(
            reconstruction=reconstruction,
            strain=strain,
            n_layers=len(self.layers),
            n_channels=self.n_channels,
            n_x_points=self.n_x_target,
        )

    def export_dicom_metadata(self, result: PipelineResult) -> dict:
        """Generate DICOM-compatible metadata for the reconstructed volume.

        Returns a dictionary of DICOM tags matching Table 3.2 of the thesis.
        """
        import uuid
        recon = result.reconstruction

        study_uid = f"1.2.826.0.1.{uuid.uuid4().int % 10**12}"
        series_uid = f"1.2.826.0.1.{uuid.uuid4().int % 10**12}"
        frame_uid = f"1.2.826.0.1.{uuid.uuid4().int % 10**12}"

        return {
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": series_uid,
            "FrameOfReferenceUID": frame_uid,
            "PixelSpacing": [recon.spacing["x_mm"], recon.spacing["z_mm"]],
            "SliceThickness": recon.spacing["y_mm"],
            "Modality": "CT",
            "Rows": recon.binary_volume.shape[0],
            "Columns": recon.binary_volume.shape[2],
            "NumberOfSlices": recon.binary_volume.shape[1],
            "PhysicalDimensions": recon.dimensions,
        }

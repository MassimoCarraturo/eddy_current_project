"""
Unified ECT pipeline: geometry reconstruction + strain prediction.

Connects the pillars of the project into a single workflow:

  Raw ECT data  ──►  Segmentation  ──►  3D DICOM  ──►  FCM-ready geometry
       │
       ├──►  Impedance inversion  ──►  Δσ/σ₀  ──►  ε = Δσ/(σ₀·κ)      (scalar)
       │
       └──►  Directional probes  ──►  MAP inversion + prior  ──►  ε_ij  (tensor)

The scalar branch reproduces the thesis workflow; the tensor branch fuses a set
of directional measurements with a process-simulation prior to reconstruct the
full strain tensor field (see src/tensor_model/).
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
class TensorStrainResult:
    """Output of the tensor strain-prediction stage.

    strain_voigt has shape (n_layers, n_points, 6) in tensor-Voigt order
    [xx, yy, zz, yz, xz, xy].  ``resolution`` is the per-component data-resolved
    fraction (1 = determined by measurement, 0 = taken from the prior).
    """
    layer_indices: np.ndarray
    strain_voigt: np.ndarray
    directions: list
    resolution: np.ndarray
    condition_number: float
    component_names: tuple = ("xx", "yy", "zz", "yz", "xz", "xy")

    def mean_strain_voigt(self) -> np.ndarray:
        """Per-layer strain Voigt vectors, averaged over x (n_layers, 6)."""
        return self.strain_voigt.mean(axis=1)

    def component_profile(self, name: str) -> np.ndarray:
        """Per-layer profile of one component, averaged over x (n_layers,)."""
        i = self.component_names.index(name)
        return self.strain_voigt[..., i].mean(axis=1)

    def strain_tensor(self, layer_pos: int, x_pos: int = 0) -> np.ndarray:
        """Full 3x3 strain tensor at a (layer, x) location."""
        from ..tensor_model.elastoresistivity import from_voigt
        return from_voigt(self.strain_voigt[layer_pos, x_pos])

    def volumetric_strain(self) -> np.ndarray:
        """Volumetric strain tr(eps) field (n_layers, n_points)."""
        return self.strain_voigt[..., :3].sum(axis=-1)

    def equivalent_strain(self) -> np.ndarray:
        """von-Mises-equivalent strain field (n_layers, n_points)."""
        v = self.strain_voigt
        exx, eyy, ezz = v[..., 0], v[..., 1], v[..., 2]
        eyz, exz, exy = v[..., 3], v[..., 4], v[..., 5]
        dev = ((exx - eyy) ** 2 + (eyy - ezz) ** 2 + (ezz - exx) ** 2
               + 6.0 * (eyz ** 2 + exz ** 2 + exy ** 2))
        return np.sqrt(dev / 2.0) * (2.0 / 3.0)


@dataclass
class PipelineResult:
    """Combined output of the unified pipeline."""
    reconstruction: ReconstructionResult
    strain: Optional[StrainResult]
    n_layers: int
    n_channels: int
    n_x_points: int
    tensor_strain: Optional["TensorStrainResult"] = None


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
        self.directional_layers: dict[int, np.ndarray] = {}
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

    # -- Tensor strain (directional probes + simulation prior) --------------

    def add_directional_layer(self, layer_index: int, measurements: np.ndarray):
        """Add directional ECT measurements for one layer.

        Args:
            layer_index: Layer number (0-based).
            measurements: Relative directional readings n_hat^T (delta_sigma/sigma0)
                n_hat, shape (n_probes,) for a single per-layer estimate, or
                (n_probes, n_x) for a spatially resolved one. Probe ordering must
                match the ``directions`` passed to run_tensor_strain_prediction.
        """
        m = np.asarray(measurements, dtype=float)
        if m.ndim == 1:
            m = m[:, np.newaxis]  # (n_probes,) -> (n_probes, 1 point)
        self.directional_layers[layer_index] = m

    def run_tensor_strain_prediction(
        self,
        model,
        directions,
        prior_field: np.ndarray = None,
        prior_weight: float = 1e-3,
    ) -> TensorStrainResult:
        """Stage 2b: reconstruct the full strain tensor from directional probes.

        Fuses the directional measurements with an optional process-simulation
        prior via the MAP estimator in tensor_model.TensorStrainInverter.

        Args:
            model: ElastoResistivityModel relating strain to conductivity.
            directions: list of unit sensing directions (the probe set).
            prior_field: optional prior strain, shape (n_layers, 6) or
                (n_layers, n_points, 6); broadcast per layer when 2-D.
            prior_weight: MAP regularisation lambda (small -> trust data).
        """
        from ..tensor_model import TensorStrainInverter
        from ..tensor_model.probe_design import design_metrics

        if not self.directional_layers:
            raise ValueError(
                "No directional layer data. Call add_directional_layer() first."
            )
        n_probes = len(directions)
        idxs = sorted(self.directional_layers.keys())

        inverter = TensorStrainInverter(
            model, directions, sigma0=1.0, prior_weight=prior_weight
        )

        strain_layers = []
        for li, idx in enumerate(idxs):
            meas = self.directional_layers[idx]            # (n_probes, n_points)
            if meas.shape[0] != n_probes:
                raise ValueError(
                    f"Layer {idx}: {meas.shape[0]} measurements but "
                    f"{n_probes} directions were given."
                )
            Y = meas.T                                     # (n_points, n_probes)
            prior = None
            if prior_field is not None:
                pf = np.asarray(prior_field, dtype=float)
                prior = np.tile(pf[li], (Y.shape[0], 1)) if pf.ndim == 2 else pf[li]
            strain_layers.append(inverter.invert_field(Y, prior))

        strain_voigt = np.stack(strain_layers, axis=0)     # (n_layers, n_points, 6)
        metrics = design_metrics(inverter.A)

        return TensorStrainResult(
            layer_indices=np.array(idxs),
            strain_voigt=strain_voigt,
            directions=list(directions),
            resolution=inverter.data_resolved_fraction(),
            condition_number=metrics["condition_number"],
        )

    # -- Orchestration -------------------------------------------------------

    def run(self, run_strain: bool = True) -> PipelineResult:
        """Execute the full pipeline.

        Args:
            run_strain: If True, also run the scalar strain prediction stage.
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

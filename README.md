# Eddy Current Testing Framework for Additive Manufacturing

A Python framework for in-situ Eddy Current Testing (ECT) of Laser Powder Bed Fusion (LPBF) processes. It provides automated geometry reconstruction from ECT impedance data and physics-based strain prediction via conductivity inversion.

This work extends the codebase from the MSc thesis *"Towards Eddy Current-Based Strain Prediction in Laser Powder Bed Fusion of Metals"* (B. I. Ates, TUM, December 2025).

## Modules

### Segmentation (`src/segmentation/`)

Converts raw ECT impedance images into binary void/solid maps. Replaces the manual threshold (fixed at 110) with automatic methods:

- **Otsu's method** &mdash; maximises inter-class variance to find the optimal global threshold
- **Adaptive thresholding** &mdash; local mean via integral images, handles non-uniform sensor sensitivity
- **Ensemble voting** &mdash; 2-of-3 consensus (Otsu + adaptive + median) with morphological cleanup

### Impedance Inversion (`src/inversion/`)

Recovers electrical conductivity from measured impedance using the analytical Dodd-Deeds model (Bowler 2019, Eq. 6.88/6.89) and bounded Brent optimisation. Replaces the previous RBF-interpolation lookup table.

The full pipeline maps impedance to strain:

```
Z_measured  -->  sigma (Brent optimisation)  -->  epsilon = (sigma - sigma_0) / (sigma_0 * kappa)
```

where kappa = -0.326 for 316L stainless steel.

### FEM Coil Impedance (`src/fem_impedance/`)

Computes coil impedance from FEM electric-field solutions via cross-section integration over the coil winding area. Uses NGSolve for 2D axisymmetric models with skin-depth-aware mesh refinement. Falls back to the analytical model when NGSolve is not installed.

### Unified Pipeline (`src/pipeline/`)

Integrates geometry reconstruction and strain prediction into a single `UnifiedECTPipeline` class:

```
Raw ECT data  -->  Segmentation  -->  3D volume  -->  DICOM metadata
     |
     +--->  Impedance inversion  -->  conductivity  -->  scalar strain (kappa)
     |
     +--->  Directional probes  -->  MAP inversion + prior  -->  strain TENSOR field
```

Processes layer-wise multi-channel impedance data and exports DICOM-compatible metadata for downstream Finite Cell Method (FCM) simulations. The tensor branch (`add_directional_layer` + `run_tensor_strain_prediction`) consumes directional-probe measurements and a process-simulation prior to reconstruct the full strain-tensor field, returning a `TensorStrainResult` with per-component resolution and derived volumetric / von-Mises-equivalent strain fields. See `examples/demo_pipeline_tensor.py`.

### Tensor Strain Model (`src/tensor_model/`)

Generalises the scalar strain-conductivity law (`Δσ/σ₀ = κ·ε`) to the full tensorial relation `(Δσ/σ₀)_ij = K_ijkl·ε_kl`, and provides the tools to recover a 3-D strain tensor from scalar ECT measurements:

- **`elastoresistivity`** &mdash; `ElastoResistivityModel`: two-constant isotropic (or general anisotropic) strain-to-conductivity coupling. The thesis scalar κ is one directional projection of this tensor.
- **`observability`** &mdash; treats each directional probe as sensing `n̂ᵀσn̂` (a conductivity "rosette") and uses an SVD to report which strain components a probe configuration can resolve. Backed by the FEM reciprocity kernel in `fem_impedance/sensitivity.py`.
- **`probe_design`** &mdash; optimises probe orientations/tilts to make the inversion full-rank and well-conditioned; reports the fundamental rank limit set by the material's elastoresistivity.
- **`strain_inversion`** &mdash; `TensorStrainInverter`: regularised MAP inversion that fuses ECT measurements with a process-simulation prior (data assimilation). The resolution matrix shows, per component, how much comes from data vs prior.

A single scalar impedance cannot determine a 6-component tensor, but a small set of directional/tilted probes (plus a simulation prior for the hard-to-observe components) can. See `examples/demo_observability.py`, `demo_probe_design.py`, and `demo_tensor_inversion.py`.

## Quick Start

```python
from src.utils.common import ECTCoilParams, MaterialParams
from src.inversion import DoddDeedsModel, ImpedanceInverter
from src.segmentation import otsu_threshold, multi_method_segment
from src.pipeline import UnifiedECTPipeline

# Define coil and material
coil = ECTCoilParams(
    r_inner=0.535e-3, r_outer=1.31e-3, length=2.93e-3,
    n_turns=235, liftoff=0.56e-3, frequency=240e3,
)
material = MaterialParams(sigma=17.7e6)  # 316L steel

# Segment an ECT image
threshold, binary = otsu_threshold(grayscale_image)

# Invert impedance to strain
inverter = ImpedanceInverter(coil, sigma_bounds=(0.8*17.7e6, 1.2*17.7e6))
sigma_recovered, strain = inverter.invert_to_strain(z_array, sigma_ref=17.7e6, kappa=-0.326)
```

## Examples

Run the demo scripts to see each module in action:

```bash
python examples/demo_segmentation.py       # Segmentation method comparison
python examples/demo_inversion.py          # Forward model + round-trip inversion
python examples/demo_fem_impedance.py      # Conductivity/frequency sweeps
python examples/demo_unified_pipeline.py   # Full pipeline on synthetic LPBF build
python examples/demo_tensor_model.py       # Tensor elastoresistivity + rosette
python examples/demo_observability.py      # Which strain components are resolvable
python examples/demo_probe_design.py       # Optimal probe-set design
python examples/demo_tensor_inversion.py   # Tensor inversion with simulation prior
python examples/demo_pipeline_tensor.py    # Tensor strain field via UnifiedECTPipeline
```

## Tests

```bash
pytest tests/ -v
```

63 unit tests covering segmentation, forward-model structure, round-trip inversion, pipeline integration (scalar and tensor), FEM impedance (NGSolve), the tensor elastoresistivity model, observability, probe design, and tensor-strain inversion. FEM tests skip automatically when NGSolve is absent.

## Dependencies

- Python 3.10+
- NumPy, SciPy, Matplotlib
- [NGSolve](https://ngsolve.org/) (optional, for FEM impedance and sensitivity kernels)

## References

- Dodd, C. V. & Deeds, W. E. (1968). Analytical solutions to eddy-current probe-coil problems. *Journal of Applied Physics*, 39(6).
- Bowler, J. R. (2019). *Eddy-Current Nondestructive Evaluation*. Springer.
- Ates, B. I. (2025). *Towards Eddy Current-Based Strain Prediction in Laser Powder Bed Fusion of Metals*. MSc Thesis, Technical University of Munich.

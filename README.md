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
     +-------->  Impedance inversion  -->  conductivity  -->  strain
```

Processes layer-wise multi-channel impedance data and exports DICOM-compatible metadata for downstream Finite Cell Method (FCM) simulations.

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
```

## Tests

```bash
pytest tests/ -v
```

16 unit tests covering segmentation accuracy, forward-model structure, round-trip inversion, and pipeline integration. All pass in under 1 second.

## Dependencies

- Python 3.10+
- NumPy, SciPy, Matplotlib
- [NGSolve](https://ngsolve.org/) (optional, for FEM impedance computation)

## References

- Dodd, C. V. & Deeds, W. E. (1968). Analytical solutions to eddy-current probe-coil problems. *Journal of Applied Physics*, 39(6).
- Bowler, J. R. (2019). *Eddy-Current Nondestructive Evaluation*. Springer.
- Ates, B. I. (2025). *Towards Eddy Current-Based Strain Prediction in Laser Powder Bed Fusion of Metals*. MSc Thesis, Technical University of Munich.

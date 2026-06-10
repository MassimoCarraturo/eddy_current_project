"""Tensorial strain-conductivity (elastoresistivity) modelling and the
observability analysis that determines which strain-tensor components an ECT
probe configuration can resolve.
"""

from .elastoresistivity import (
    ElastoResistivityModel,
    to_voigt,
    from_voigt,
    VOIGT_INDEX,
)
from .observability import (
    ObservabilityAnalysis,
    projection_row,
    direction,
    NORMAL_COIL_ROW,
)
from .probe_design import (
    design_metrics,
    evaluate_configuration,
    max_achievable_rank,
    optimize_inplane_rosette,
    optimize_probe_set,
)
from .strain_inversion import (
    TensorStrainInverter,
    volumetric_strain,
)

__all__ = [
    "ElastoResistivityModel",
    "to_voigt",
    "from_voigt",
    "VOIGT_INDEX",
    "ObservabilityAnalysis",
    "projection_row",
    "direction",
    "NORMAL_COIL_ROW",
    "design_metrics",
    "evaluate_configuration",
    "max_achievable_rank",
    "optimize_inplane_rosette",
    "optimize_probe_set",
    "TensorStrainInverter",
    "volumetric_strain",
]

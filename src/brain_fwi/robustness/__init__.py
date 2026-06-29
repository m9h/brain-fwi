"""FWI-US robustness test & evaluation (MAITE-conformant, lightweight).

Domain-specific perturbers + a sweep harness that produces robustness curves
(reconstruction error vs perturbation strength), framed in the MAITE/NRTK
AI-robustness-T&E paradigm. See ``docs/design/maite_robustness_te.md``.
"""
from .perturbers import (
    AbsorptionErrorPerturber,
    MeasurementNoisePerturber,
    SkullPosePerturber,
)
from .harness import robustness_sweep, plot_robustness_curve
from .metrics import brain_roi_rmse, lesion_rmse

__all__ = [
    "AbsorptionErrorPerturber",
    "MeasurementNoisePerturber",
    "SkullPosePerturber",
    "robustness_sweep",
    "plot_robustness_curve",
    "brain_roi_rmse",
    "lesion_rmse",
]

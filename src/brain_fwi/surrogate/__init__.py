"""Neural-operator surrogates for the j-Wave forward simulator.

Phase 4 home. See ``docs/design/phase4_fno_surrogate.md``.
"""

from .fno2d import CToTraceFNO
from .fno3d import CToTraceFNO3D
from .train import surrogate_loss, train_fno_surrogate
from .validation import (
    format_gate_report,
    gradient_accuracy,
    trace_fidelity,
)

__all__ = [
    "CToTraceFNO",
    "CToTraceFNO3D",
    "surrogate_loss",
    "train_fno_surrogate",
    "trace_fidelity",
    "gradient_accuracy",
    "format_gate_report",
]

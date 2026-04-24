"""Neural-operator surrogates for the j-Wave forward simulator.

Phase 4 home. See ``docs/design/phase4_fno_surrogate.md``.
"""

from .fno2d import CToTraceFNO, FNO2D, SpectralConv2D

__all__ = ["CToTraceFNO", "FNO2D", "SpectralConv2D"]

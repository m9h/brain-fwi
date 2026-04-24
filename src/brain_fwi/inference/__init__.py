"""Learning-based inference utilities.

Phase 2+ home: NPE (conditional normalizing flow), flow-matching /
diffusion priors, SBC calibration.

Export split:

- :mod:`dataprep` (eager) — SIREN weight reconstruction + theta
  extraction. Pure NumPy + Equinox; safe on any platform.
- :mod:`flow` (lazy) — conditional normalizing flow + NPE training.
  Imports flowjax only on first access so platforms without flowjax /
  jaxlib wheels (macOS 26 x86_64 is a live case) can still use
  everything else.

See ``docs/design/data_pipeline.md`` and
``docs/design/phase3_diffusion_prior.md``.
"""

from .dataprep import siren_from_sample, theta_from_sample

__all__ = ["siren_from_sample", "theta_from_sample", "ConditionalFlow", "train_npe"]


def __getattr__(name):
    if name in ("ConditionalFlow", "train_npe"):
        from .flow import ConditionalFlow, train_npe
        return {"ConditionalFlow": ConditionalFlow, "train_npe": train_npe}[name]
    raise AttributeError(name)

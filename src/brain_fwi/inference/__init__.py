"""Learning-based inference utilities.

Prepares Phase-0 samples for amortised posterior estimation / score priors /
neural-operator training. See ``docs/design/data_pipeline.md``.
"""

from .dataprep import siren_from_sample, theta_from_sample

__all__ = ["siren_from_sample", "theta_from_sample"]

"""MAITE-``Augmentation``-conformant perturbers for FWI-US robustness T&E.

Each perturber is a *functional* transform on a "sample" dict (the inversion
input + ground truth + acquisition metadata). They duck-type MAITE's
``Augmentation`` protocol (callable returning a perturbed sample), so they
compose with ``maite.evaluate`` if the maite package is installed — but require
no new dependency here.

These are the *domain-specific* acoustic / acquisition / model perturbers
(see ``docs/design/maite_robustness_te.md``). NRTK's optical-sensor perturbers
(lens blur, aperture, photometric noise) are the wrong physics for acoustic FWI.
"""
from __future__ import annotations
import numpy as np


def _copy(sample):
    return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in sample.items()}


class AbsorptionErrorPerturber:
    """Scale the ASSUMED skull absorption (alpha) by ``scale`` — tests
    robustness to error in the CT-derived alpha prior. Strength 1.0 = truth."""

    key = "alpha_assumed"

    def __init__(self, scale: float):
        self.scale = float(scale)

    def __call__(self, sample: dict) -> dict:
        s = _copy(sample)
        s[self.key] = np.asarray(s[self.key]) * self.scale
        return s


class MeasurementNoisePerturber:
    """Add white Gaussian noise to the measurements at a target SNR (dB)."""

    key = "observed"

    def __init__(self, snr_db: float, seed: int = 0):
        self.snr_db = float(snr_db)
        self.seed = int(seed)

    def __call__(self, sample: dict) -> dict:
        s = _copy(sample)
        x = np.asarray(s[self.key], dtype=np.float64)
        p_noise = np.var(x) / (10 ** (self.snr_db / 10))
        rng = np.random.default_rng(self.seed)
        noisy = x + rng.standard_normal(x.shape) * np.sqrt(p_noise)
        s[self.key] = noisy.astype(np.asarray(sample[self.key]).dtype)
        return s


class SkullPosePerturber:
    """Rigidly shift the ASSUMED model fields by an integer-voxel translation —
    a crude skull-pose / registration error. This is the *targeting-relevant*
    perturber: the same misregistration that displaces a neuromodulation focus
    also degrades the reconstruction."""

    keys = ("c_assumed", "density_assumed", "alpha_assumed", "c_init")

    def __init__(self, shift_vox=(0, 0, 0)):
        self.shift = tuple(int(x) for x in shift_vox)

    def __call__(self, sample: dict) -> dict:
        s = _copy(sample)
        axes = tuple(range(len(self.shift)))
        for k in self.keys:
            v = s.get(k)
            if v is not None and getattr(v, "ndim", -1) == len(self.shift):
                s[k] = np.roll(v, self.shift, axis=axes)
        return s

"""MAITE-style robustness harness for FWI-US: domain-specific perturbers +
a sweep that produces a robustness curve.

Fast tests use a toy model (no FWI) to verify the harness mechanics + perturber
transforms; the real alpha-error sweep through j-Wave FWI is exercised by
scripts/robustness_alpha_sweep.py.
"""
from __future__ import annotations
import numpy as np


def test_absorption_perturber_scales_assumed_alpha_functionally():
    from brain_fwi.robustness import AbsorptionErrorPerturber
    s = {"alpha_assumed": np.full((3, 3), 8.0)}
    out = AbsorptionErrorPerturber(0.5)(s)
    assert np.allclose(out["alpha_assumed"], 4.0)          # scaled
    assert np.allclose(s["alpha_assumed"], 8.0)            # original untouched (functional)


def test_noise_perturber_hits_target_snr():
    from brain_fwi.robustness import MeasurementNoisePerturber
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((2000,)).astype(np.float32)
    out = MeasurementNoisePerturber(20.0, seed=1)({"observed": sig})
    noise = out["observed"] - sig
    snr = 10 * np.log10(np.var(sig) / np.var(noise))
    assert abs(snr - 20.0) < 1.5, f"got {snr:.1f} dB, expected ~20"


def test_pose_perturber_shifts_assumed_model():
    from brain_fwi.robustness import SkullPosePerturber
    c = np.arange(27.0).reshape(3, 3, 3)
    out = SkullPosePerturber(shift_vox=(1, 0, 0))({"c_assumed": c})
    assert np.allclose(out["c_assumed"], np.roll(c, 1, axis=0))


def test_robustness_sweep_curve_minimised_at_true_parameter():
    from brain_fwi.robustness import AbsorptionErrorPerturber, robustness_sweep
    # toy model: recon error grows with |assumed-alpha mean - true(=1)|
    sample = {"alpha_assumed": np.ones((4, 4)), "c_true": np.zeros((4, 4))}
    toy_model = lambda s: np.full((4, 4), abs(float(s["alpha_assumed"].mean()) - 1.0))
    metric = lambda recon, s: float(recon.mean())
    strengths = [0.5, 0.75, 1.0, 1.25, 1.5]
    curve = robustness_sweep(toy_model, AbsorptionErrorPerturber, strengths, sample, metric)
    scores = [c[1] for c in curve]
    assert np.argmin(scores) == 2, "robustness curve should bottom out at the true alpha (scale 1.0)"
    assert scores[0] > scores[2] and scores[-1] > scores[2], "error should grow away from truth"

"""TDD spec for ICL syn-vs-exp calibration diagnostic helpers.

Each test was written failing first, then minimum code to GREEN.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_best_lag_correlation_recovers_known_shift():
    """If exp = roll(syn, k_true), best_lag_correlation returns k_true
    and a normalised correlation > 0.99.
    """
    from brain_fwi.diagnostics.calibration import best_lag_correlation

    rng = np.random.default_rng(0)
    n = 1024
    syn = rng.standard_normal(n)
    k_true = 7
    exp = np.zeros_like(syn)
    exp[k_true:] = syn[:-k_true]  # exp lags syn by k_true samples

    k_est, corr = best_lag_correlation(syn, exp, max_lag=20)
    assert k_est == k_true, f"got lag {k_est}, expected {k_true}"
    assert corr > 0.99, f"got corr {corr:.3f}, expected > 0.99"


def test_best_lag_correlation_returns_zero_for_aligned_signals():
    """Identical inputs ⇒ best lag is 0 with corr ≈ 1."""
    from brain_fwi.diagnostics.calibration import best_lag_correlation

    rng = np.random.default_rng(1)
    x = rng.standard_normal(512)
    k, c = best_lag_correlation(x, x.copy(), max_lag=10)
    assert k == 0
    assert c == pytest.approx(1.0, abs=1e-9)


def test_best_lag_correlation_negative_lag_when_a_lags_b():
    """If a is the delayed version of b, the recovered lag is negative."""
    from brain_fwi.diagnostics.calibration import best_lag_correlation

    rng = np.random.default_rng(2)
    n = 512
    b = rng.standard_normal(n)
    a = np.zeros_like(b)
    a[5:] = b[:-5]                  # a lags b by 5 samples
    k, c = best_lag_correlation(a, b, max_lag=20)
    assert k == -5, f"got lag {k}, expected -5"
    assert c > 0.99


def test_best_lag_correlation_handles_zero_signal():
    """A zero-energy input does not crash; correlation returns 0."""
    from brain_fwi.diagnostics.calibration import best_lag_correlation

    a = np.zeros(128)
    b = np.random.default_rng(0).standard_normal(128)
    k, c = best_lag_correlation(a, b, max_lag=5)
    assert c == 0.0


def test_perp_distance_from_centre_known_geometry():
    """Perpendicular distance from origin to a unit-length horizontal
    line at z=3 is exactly 3.0 in the xz plane."""
    from brain_fwi.diagnostics.calibration import perp_distance_xz

    src = np.array([-1.0, 0.0, 3.0])
    rcv = np.array([+1.0, 0.0, 3.0])
    centre = np.array([0.0, 0.0, 0.0])
    d = perp_distance_xz(src, rcv, centre)
    assert d == pytest.approx(3.0, abs=1e-9)


def test_perp_distance_zero_when_line_passes_through_centre():
    """Line through phantom centre ⇒ perp distance = 0."""
    from brain_fwi.diagnostics.calibration import perp_distance_xz

    src = np.array([-1.0, 0.0, 0.0])
    rcv = np.array([+1.0, 0.0, 0.0])
    centre = np.array([0.0, 0.0, 0.0])
    assert perp_distance_xz(src, rcv, centre) == pytest.approx(0.0, abs=1e-12)


def test_perp_distance_ignores_y_coordinate():
    """The 2.5D imaging plane is xz; y differences must not affect
    the perpendicular distance."""
    from brain_fwi.diagnostics.calibration import perp_distance_xz

    src = np.array([-1.0, 7.0, 3.0])     # arbitrary y
    rcv = np.array([+1.0, -2.0, 3.0])    # arbitrary y
    centre = np.array([0.0, 99.0, 0.0])
    assert perp_distance_xz(src, rcv, centre) == pytest.approx(3.0, abs=1e-9)

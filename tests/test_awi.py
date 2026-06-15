"""Adaptive Waveform Inversion (AWI) objective — cycle-skip robustness.

AWI (Warner & Guasch 2016) is the missing ingredient behind the 192^3 MIDA
reconstruction failure (job 984: plain multiscale L2 from a water start
recovered 0% of the skull contrast — it cycle-skipped on the high-contrast
skull). AWI replaces the raw L2 residual with a Wiener matching filter
penalised for energy away from zero lag, giving a far wider basin of
attraction.

This is the RED-first test: it characterises the property we need (AWI is
unimodal in a time shift where L2 cycle-skips) BEFORE awi_loss exists.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.inversion.losses import l2_loss, awi_loss
from brain_fwi.utils.wavelets import toneburst


@pytest.fixture
def centered_burst():
    """A 3-cycle tone burst centred in the window. Period = 40 samples."""
    dt = 1.0e-7          # 100 ns
    f0 = 250e3           # period T = 1/f0 = 4 us = 40 samples
    nt = 400
    d = toneburst(f0=f0, dt=dt, n_cycles=3, n_samples=nt)
    # centre the envelope peak at nt//2 so ± shifts stay in-window
    d = jnp.roll(d, nt // 2 - int(jnp.argmax(jnp.abs(d))))
    return d, 40  # (trace, period_samples)


def _shift(d, k):
    """Predicted = observed shifted by k samples, shaped (nt, 1)."""
    return jnp.roll(d, k)[:, None]


def test_matching_filter_peaks_at_shift(centered_burst):
    """Proper (limited-length) AWI: the Wiener matching filter that maps the
    predicted trace onto a time-shifted copy must PEAK AT THE SHIFT LAG.

    This is the property the full-length FFT variant lacked — and the reason
    its FWI velocity gradient was uninformative (it underperformed L2 on the
    skull-recovery experiment). A filter that localises the misfit as a lag
    is what makes the gradient drive the model toward alignment.
    """
    from brain_fwi.inversion.losses import _awi_matching_filter
    d, _T = centered_burst
    s = 12  # observed = predicted delayed by 12 samples
    w, lags = _awi_matching_filter(d, jnp.roll(d, s), filter_half_len=40, eps=1e-3)
    peak_lag = int(np.asarray(lags)[int(jnp.argmax(jnp.abs(w)))])
    assert abs(peak_lag - s) <= 1, f"filter peaked at lag {peak_lag}, expected ~{s}"


def test_awi_minimised_at_alignment(centered_burst):
    """AWI is non-negative and strictly minimised at perfect alignment.

    (It does NOT reach 0 for a narrowband signal — a narrow spectrum forces
    a time-spread matching filter, so there is a bandwidth-dependent floor.
    The meaningful property is that alignment is the strict minimum, clearly
    below both a half-cycle and a full-cycle misalignment.)
    """
    d, T = centered_burst
    col = d[:, None]
    a0 = float(awi_loss(col, col))
    assert a0 >= 0.0
    assert a0 < float(awi_loss(_shift(d, T // 2), col))   # vs half-cycle out of phase
    assert a0 < 0.5 * float(awi_loss(_shift(d, T), col))  # clearly below one-cycle skip


def test_awi_unimodal_where_l2_cycle_skips(centered_burst):
    """The core contrast: at a full-cycle shift (τ=T) L2 has a deceptive
    LOCAL minimum (the cycle-skip trap), while AWI does not — AWI keeps
    increasing with |shift|. This is why AWI escapes the skull cycle-skip."""
    d, T = centered_burst
    col = d[:, None]
    lags = list(range(-3 * T // 2, 3 * T // 2 + 1, 4))  # includes 0, ±T/2, ±T, ±3T/2

    l2 = {k: float(l2_loss(_shift(d, k), col)) for k in lags}
    awi = {k: float(awi_loss(_shift(d, k), col)) for k in lags}

    # AWI: global minimum at zero shift
    assert min(awi, key=awi.get) == 0, f"AWI min not at 0: {min(awi, key=awi.get)}"
    # AWI: NO local minimum at one full cycle — it is larger than at half a
    # cycle (monotone-ish in |shift|), unlike L2.
    assert awi[T] > awi[T // 2] > awi[0], "AWI should rise monotonically with shift"
    assert awi[T] < awi[3 * T // 2], "AWI should keep rising past one cycle"

    # L2: cycle-skips — a full-cycle shift is a LOCAL min (lower than the
    # half-cycle out-of-phase points on either side).
    assert l2[T] < l2[T // 2], "L2 should have a cycle-skip dip at +1 cycle"
    assert l2[T] < l2[3 * T // 2], "L2 should have a cycle-skip dip at +1 cycle"

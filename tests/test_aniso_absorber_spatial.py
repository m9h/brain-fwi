"""Spatially-varying anisotropy in the j-Wave absorber: the attenuation tensor
(fibre orientation) may differ per voxel. A domain split left (fibre||x) / right
(fibre||y): a wave propagating along x attenuates at a_par (LOW, along fibre) in
the left region and a_perp (HIGH, across fibre) in the right. The mean-tensor form
could not do this (both regions would see trace/n); the per-voxel fractional-
directional operator can, and stably.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.slow
def test_local_attenuation_follows_local_fibre_orientation():
    from jwave.geometry import Medium
    from brain_fwi.simulation.forward import (
        build_domain, build_time_axis, simulate_shot_sensors, _build_source_signal)
    S = (160, 24, 24); dx = 2.5e-4; pml = 8; Y = 1.1
    a_par, a_perp = 3.0, 9.0
    dom = build_domain(S, dx)
    c = jnp.full(S, 1500.0, jnp.float32); rho = jnp.full(S, 1000.0, jnp.float32)

    # left half (x<80): fibre||x  -> [Dxx=a_par, Dyy=a_perp, ...]
    # right half (x>=80): fibre||y -> [Dxx=a_perp, Dyy=a_par, ...]
    D = np.zeros((*S, 6), np.float32)
    D[:80, ..., 0] = a_par;  D[:80, ..., 1] = a_perp; D[:80, ..., 2] = a_perp
    D[80:, ..., 0] = a_perp; D[80:, ..., 1] = a_par;  D[80:, ..., 2] = a_perp

    def med(tens):
        return Medium(domain=dom, sound_speed=c, density=rho, attenuation=0.0,
                      pml_size=pml, alpha_power=Y, attenuation_tensor=tens)
    ta = build_time_axis(med(None), cfl=0.3, t_end=6e-5); dt, nt = float(ta.dt), int(ta.Nt)
    sig = _build_source_signal(500e3, dt, nt)
    src = (8, 12, 12)
    rx = np.arange(20, 150, 10)
    recv = (rx, np.full(len(rx), 12), np.full(len(rx), 12))

    def peak(tens):
        tr = np.asarray(simulate_shot_sensors(med(tens), ta, src, recv, sig, dt))
        return np.max(np.abs(tr), axis=0)
    free = peak(None)
    loss = peak(jnp.asarray(D))
    assert np.all(np.isfinite(loss)), "spatially-varying anisotropic absorber unstable"
    R = np.clip(loss / (free + 1e-30), 1e-6, 1.0)
    cum = -20 * np.log10(R)                                   # cumulative dB
    Lc = (rx - src[0]) * dx * 100.0

    # local slope (dB/cm) in the left (along-fibre) vs right (across-fibre) region
    left = (rx >= 30) & (rx < 70); right = (rx >= 90) & (rx < 140)
    a_left = np.polyfit(Lc[left], cum[left], 1)[0]
    a_right = np.polyfit(Lc[right], cum[right], 1)[0]
    ana_par = a_par * 0.5 ** Y; ana_perp = a_perp * 0.5 ** Y
    # left ~ a_par (low), right ~ a_perp (high); they must DIFFER (per-voxel works)
    assert a_left < a_right, f"local anisotropy not spatially resolved: left {a_left:.2f} >= right {a_right:.2f}"
    assert a_right / a_left > 1.8, f"weak spatial contrast: left {a_left:.2f} right {a_right:.2f}"
    assert abs(a_left / ana_par - 1.0) < 0.5, f"left {a_left:.2f} vs analytic a_par {ana_par:.2f}"

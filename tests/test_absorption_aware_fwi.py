"""Absorption-aware FWI: thread a KNOWN, fixed attenuation field through the
FWI forward model.

When the observed data carries power-law absorption (a real skull attenuates
~10-20% per pass), a *lossless* forward model has irreducible amplitude misfit:
it cannot reproduce the attenuated waveforms no matter the velocity, so the
velocity update tries (wrongly) to explain the missing amplitude with
structure. Baking the known alpha (e.g. from CT) into the forward model removes
that bias — this is the clinical setup: skull rho and alpha known, invert for c.

Two guards:
  - mechanism (fast, deterministic): at the *true* velocity, the data misfit
    against absorption-containing observations is far lower with the known
    alpha than lossless. This is *why* absorption-aware FWI helps.
  - plumbing: ``FWIConfig.attenuation`` actually reaches ``run_fwi``'s forward
    medium — a one-iteration FWI at the true velocity has near-zero initial
    loss with the known alpha and large loss without it.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import pytest


def _skull_slab_phantom():
    """A strongly-absorbing bone slab across a water tank. The slab both
    refracts (c, rho contrast) and attenuates (alpha); the absorption-aware
    model reproduces it exactly, the lossless model cannot."""
    grid = (40, 40, 40)
    dx = 4e-4
    y = 1.1
    c = jnp.full(grid, 1500.0, jnp.float32)
    rho = jnp.full(grid, 1000.0, jnp.float32)
    skull = jnp.zeros(grid, jnp.float32).at[16:24, :, :].set(1.0)
    c = jnp.where(skull > 0, 2800.0, c)
    rho = jnp.where(skull > 0, 1850.0, rho)
    alpha = jnp.where(skull > 0, 8.0, 0.0).astype(jnp.float32)
    return grid, dx, y, c, rho, alpha


def _misfit(pred, obs):
    m = min(pred.shape[0], obs.shape[0])
    return float(jnp.linalg.norm(pred[:m] - obs[:m]) / (jnp.linalg.norm(obs[:m]) + 1e-12))


@pytest.mark.slow
def test_known_alpha_forward_matches_lossy_data_lossless_does_not():
    """Mechanism: at the true velocity, the known-alpha forward reproduces the
    (lossy) observations; the lossless forward has large residual amplitude."""
    from brain_fwi.simulation.forward import (
        build_domain, build_medium, build_time_axis, simulate_shot_sensors,
        _build_source_signal,
    )

    grid, dx, y, c, rho, alpha = _skull_slab_phantom()
    pml = 8
    dom = build_domain(grid, dx)
    med_truth = build_medium(dom, c, rho, pml_size=pml, attenuation=alpha, alpha_power=y)
    ta = build_time_axis(med_truth, cfl=0.3, t_end=3e-5)
    dt = float(ta.dt); nt = int(ta.Nt)
    sig = _build_source_signal(400e3, dt, nt)

    src = (6, 20, 20)
    rx = np.array([34, 34, 34, 34, 34, 34, 34, 34, 34])
    ry = np.array([18, 18, 18, 20, 20, 20, 22, 22, 22])
    rz = np.array([18, 20, 22, 18, 20, 22, 18, 20, 22])
    recv = (rx, ry, rz)

    obs = simulate_shot_sensors(med_truth, ta, src, recv, sig, dt)

    med_lossless = build_medium(dom, c, rho, pml_size=pml, attenuation=None, alpha_power=y)
    med_aware = build_medium(dom, c, rho, pml_size=pml, attenuation=alpha, alpha_power=y)
    pred_lossless = simulate_shot_sensors(med_lossless, ta, src, recv, sig, dt)
    pred_aware = simulate_shot_sensors(med_aware, ta, src, recv, sig, dt)

    misfit_lossless = _misfit(pred_lossless, obs)
    misfit_aware = _misfit(pred_aware, obs)
    assert misfit_aware < 0.2 * misfit_lossless, (
        f"known-alpha forward misfit {misfit_aware:.4f} should be << lossless "
        f"{misfit_lossless:.4f}; absorption not reproduced in forward"
    )


@pytest.mark.slow
def test_fwi_config_threads_known_attenuation_into_forward():
    """Plumbing: ``FWIConfig.attenuation`` reaches the forward medium inside
    ``run_fwi``. At the true velocity, iter-0 loss is ~0 with the known alpha
    and large without it (RED before the config carries attenuation)."""
    from brain_fwi.simulation.forward import (
        build_domain, build_medium, build_time_axis, simulate_shot_sensors,
        _build_source_signal,
    )
    from brain_fwi.inversion.fwi import FWIConfig, run_fwi

    grid, dx, y, c, rho, alpha = _skull_slab_phantom()
    pml = 8; c_max = 3200.0; t_end = 3e-5

    # Reproduce run_fwi's internal time axis (ref medium at c_max) so the
    # observed data is generated on the exact grid the inversion steps on.
    dom = build_domain(grid, dx)
    ref_med = build_medium(dom, c_max, 1000.0, pml_size=pml)
    ta = build_time_axis(ref_med, cfl=0.3, t_end=t_end)
    dt = float(ta.dt); nt = int(ta.Nt)
    sig = _build_source_signal(400e3, dt, nt)

    src = (6, 20, 20)
    recv = (np.array([34, 34, 34]), np.array([18, 20, 22]), np.array([20, 20, 20]))

    med_truth = build_medium(dom, c, rho, pml_size=pml, attenuation=alpha, alpha_power=y)
    obs = simulate_shot_sensors(med_truth, ta, src, recv, sig, dt)[None]  # (1, nt, n_recv)

    common = dict(
        n_iters_per_band=1, shots_per_iter=1, skip_bandpass=True,
        freq_bands=[(1e3, 1e6)], pml_size=pml, cfl=0.3, c_max=c_max,
        gradient_smooth_sigma=0.0, verbose=False,
    )

    res_lossless = run_fwi(
        obs, c, rho, dx, [src], recv, sig, dt, t_end,
        config=FWIConfig(attenuation=None, **common), key=jr.PRNGKey(0),
    )
    res_aware = run_fwi(
        obs, c, rho, dx, [src], recv, sig, dt, t_end,
        config=FWIConfig(attenuation=alpha, alpha_power=y, **common), key=jr.PRNGKey(0),
    )

    loss0_lossless = res_lossless.loss_history[0]
    loss0_aware = res_aware.loss_history[0]
    assert loss0_aware < 0.1 * loss0_lossless, (
        f"FWI iter-0 loss with known alpha {loss0_aware:.4e} should be << lossless "
        f"{loss0_lossless:.4e}; attenuation not threaded into run_fwi forward"
    )

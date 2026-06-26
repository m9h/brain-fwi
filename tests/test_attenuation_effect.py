"""Regression guard: enabling attenuation in the j-Wave forward sim
produces measurably different traces from a lossless run.

This is the foundational invariant for the Phase-5 dispersion A/B
experiment: if α has no effect on the simulated traces, the whole
experiment is meaningless.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest


def test_attenuation_changes_traces_for_skull_block():
    """A 32^3 cube with a centered skull block produces ≥5% rel-L2
    different traces with vs without attenuation.

    Phase 5 invariant: if attenuation has no effect on simulated traces,
    the whole CANN α(ω) modelling effort is meaningless. This test
    closed the XFAIL once Treeby-Cox-flavoured power-law absorption
    landed in m9h/jwave (commit on `feature/time-domain-absorption`).
    """
    from brain_fwi.simulation.forward import (
        _build_source_signal,
        build_domain,
        build_medium,
        build_time_axis,
        simulate_shot_sensors,
    )

    grid = (32, 32, 32)
    dx = 5e-4

    c = jnp.full(grid, 1500.0, dtype=jnp.float32)
    rho = jnp.full(grid, 1000.0, dtype=jnp.float32)
    alpha = jnp.zeros(grid, dtype=jnp.float32)
    skull = jnp.zeros(grid, dtype=jnp.float32).at[12:20, 12:20, 12:20].set(1.0)
    c = jnp.where(skull > 0, 2800.0, c)
    rho = jnp.where(skull > 0, 1850.0, rho)
    # 8 dB/cm/MHz^1.1 over the skull block — diploic-bone-magnitude with
    # tissue-typical y=1.1 (Pinton et al. 2012). At the 500 kHz Ricker
    # peak, predicts ~13 % single-pass amplitude reduction across 4 mm.
    alpha_lossy = jnp.where(skull > 0, 8.0, alpha)
    y_skull = 1.1

    domain = build_domain(grid, dx)
    medium_lossless = build_medium(
        domain, c, rho, pml_size=4, attenuation=None, alpha_power=y_skull,
    )
    medium_lossy = build_medium(
        domain, c, rho, pml_size=4, attenuation=alpha_lossy, alpha_power=y_skull,
    )

    # Use the lossless medium's CFL for both so dt is identical.
    time_axis = build_time_axis(medium_lossless, cfl=0.3, t_end=2e-5)
    dt = float(time_axis.dt)
    n_t = int(float(time_axis.t_end) / dt)
    src_sig = _build_source_signal(500e3, dt, n_t)

    src = (4, 16, 16)
    recv = ([28], [16], [16])  # one receiver across the skull block

    d_lossless = simulate_shot_sensors(
        medium_lossless, time_axis, src, recv, src_sig, dt,
    )
    d_lossy = simulate_shot_sensors(
        medium_lossy, time_axis, src, recv, src_sig, dt,
    )

    diff = jnp.sqrt(jnp.sum((d_lossless - d_lossy) ** 2))
    norm = jnp.sqrt(jnp.sum(d_lossless ** 2)) + 1e-12
    rel_l2 = float(diff / norm)
    assert rel_l2 > 0.05, (
        f"attenuation produced only {rel_l2:.4f} rel-L2 diff "
        "(expected > 5%)"
    )


def test_attenuation_active_in_checkpointed_fwi_path():
    """The CHECKPOINTED forward (what FWI uses for gradients) must also apply
    absorption. It reimplements j-Wave's time loop and previously dropped the
    Treeby-Cox term, silently disabling absorption-aware inversion while the
    non-checkpointed path (above) worked — so the gating test passed but FWI
    saw no attenuation. Guard the parity.
    """
    from brain_fwi.simulation.forward import (
        _build_source_signal, build_domain, build_medium, build_time_axis,
        simulate_shot_sensors,
    )

    grid = (32, 32, 32); dx = 5e-4
    c = jnp.full(grid, 1500.0, dtype=jnp.float32)
    rho = jnp.full(grid, 1000.0, dtype=jnp.float32)
    skull = jnp.zeros(grid, dtype=jnp.float32).at[12:20, 12:20, 12:20].set(1.0)
    c = jnp.where(skull > 0, 2800.0, c); rho = jnp.where(skull > 0, 1850.0, rho)
    y = 1.1
    dom = build_domain(grid, dx)
    med_free = build_medium(dom, c, rho, pml_size=4, attenuation=None, alpha_power=y)
    med_loss = build_medium(dom, c, rho, pml_size=4,
                            attenuation=jnp.where(skull > 0, 8.0, 0.0), alpha_power=y)
    ta = build_time_axis(med_free, cfl=0.3, t_end=2e-5)
    dt = float(ta.dt); nt = int(2e-5 / dt)
    sig = _build_source_signal(500e3, dt, nt)
    src = (4, 16, 16); recv = ([28], [16], [16])

    d_free = simulate_shot_sensors(med_free, ta, src, recv, sig, dt, checkpointed=True)
    d_loss = simulate_shot_sensors(med_loss, ta, src, recv, sig, dt, checkpointed=True)
    rel = float(jnp.linalg.norm(d_free - d_loss) / (jnp.linalg.norm(d_free) + 1e-12))
    assert rel > 0.05, (
        f"checkpointed FWI forward applied only {rel:.4f} rel-L2 attenuation "
        "(expected > 5%) — absorption dropped in the checkpointed path"
    )

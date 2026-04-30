"""TDD spec for the FWI ↔ score-prior integration (§9 step 6).

Tests the FWIConfig fields and the compose-step plumbing inside
``_run_fwi_siren``. Defaults must be backwards-compatible (no
behaviour change when ``score_prior_weight == 0``).
"""

from __future__ import annotations

import jax.numpy as jnp


def test_fwi_config_has_score_prior_fields_with_no_op_defaults():
    """FWIConfig exposes the score-prior knobs with defaults that
    leave existing callers unaffected."""
    from brain_fwi.inversion.fwi import FWIConfig

    cfg = FWIConfig()
    assert hasattr(cfg, "score_prior_fn")
    assert cfg.score_prior_fn is None
    assert hasattr(cfg, "score_prior_weight")
    assert cfg.score_prior_weight == 0.0
    assert hasattr(cfg, "score_prior_t_eps")
    assert cfg.score_prior_t_eps == 0.01


def test_fwi_config_accepts_score_prior_fn_callable():
    """Setting ``score_prior_fn`` to a callable is allowed and
    preserves the configured weight + t_eps."""
    from brain_fwi.inversion.fwi import FWIConfig

    def fake_score(theta, t):
        return -theta

    cfg = FWIConfig(
        score_prior_fn=fake_score,
        score_prior_weight=0.3,
        score_prior_t_eps=0.05,
    )
    assert cfg.score_prior_fn is fake_score
    assert cfg.score_prior_weight == 0.3
    assert cfg.score_prior_t_eps == 0.05


def _tiny_siren_fwi_scenario(score_prior_fn=None, score_prior_weight=0.0):
    """Minimum SIREN-path FWI scenario for smoke tests.

    16² 2D grid, 1 source, 4 sensors, 1 band × 1 iter. Designed to
    exercise the gradient pipeline (including the score-prior wiring)
    in <30s on CPU.
    """
    import jax.numpy as jnp
    import jax.random as jr

    from brain_fwi.inversion.fwi import FWIConfig, run_fwi
    from brain_fwi.simulation.forward import (
        _build_source_signal,
        build_domain,
        build_medium,
        build_time_axis,
        simulate_shot_sensors,
    )

    grid = (16, 16)
    dx = 0.002
    c_min, c_max = 1400.0, 1800.0
    pml_size = 4

    c_true = jnp.where(
        jnp.linalg.norm(
            jnp.stack(jnp.meshgrid(
                jnp.arange(16) - 8.0, jnp.arange(16) - 8.0, indexing="ij",
            )), axis=0,
        ) < 4.0,
        1600.0, 1500.0,
    )
    rho = jnp.ones(grid) * 1000.0

    src_list = [(2, 8)]
    pos_grid = (jnp.array([14, 14, 8, 8]), jnp.array([4, 12, 2, 14]))

    domain = build_domain(grid, dx)
    ref_medium = build_medium(domain, c_max, 1000.0, pml_size=pml_size)
    time_axis = build_time_axis(ref_medium, cfl=0.3, t_end=40e-6)
    dt = float(time_axis.dt)
    n_samples = int(float(time_axis.t_end) / dt)
    source_signal = _build_source_signal(50e3, dt, n_samples)

    medium_true = build_medium(domain, c_true, rho, pml_size=pml_size)
    obs = simulate_shot_sensors(
        medium_true, time_axis, src_list[0], pos_grid, source_signal, dt,
    )
    observed = jnp.stack([obs], axis=0)

    cfg = FWIConfig(
        freq_bands=[(40e3, 80e3)],
        n_iters_per_band=1,
        shots_per_iter=1,
        learning_rate=50.0,
        c_min=c_min, c_max=c_max,
        pml_size=pml_size,
        skip_bandpass=True,
        verbose=False,
        parameterization="siren",
        siren_hidden=8, siren_layers=1,
        siren_pretrain_steps=10,
        score_prior_fn=score_prior_fn,
        score_prior_weight=score_prior_weight,
    )
    return run_fwi(
        observed_data=observed,
        initial_velocity=jnp.full(grid, 1500.0),
        density=rho, dx=dx,
        src_positions_grid=src_list,
        sensor_positions_grid=pos_grid,
        source_signal=source_signal,
        dt=dt, t_end=float(time_axis.t_end),
        config=cfg,
        key=jr.PRNGKey(0),
    )


def test_run_fwi_siren_calls_score_prior_fn_when_weight_positive():
    """Smoke test: with ``score_prior_weight > 0`` and a tracked score
    function, the SIREN FWI loop calls the score function during each
    gradient update. Confirms the wiring is hit end-to-end."""
    call_count = [0]

    def tracked_score(theta, t):
        call_count[0] += 1
        return jnp.zeros_like(theta)  # zero so FWI behaviour is unchanged

    _tiny_siren_fwi_scenario(
        score_prior_fn=tracked_score, score_prior_weight=0.1,
    )
    assert call_count[0] >= 1, "score_prior_fn was never called"


def test_run_fwi_siren_skips_score_prior_fn_when_weight_zero():
    """With ``score_prior_weight == 0``, the score function is not
    invoked even if it is provided."""
    call_count = [0]

    def tracked_score(theta, t):
        call_count[0] += 1
        return jnp.zeros_like(theta)

    _tiny_siren_fwi_scenario(
        score_prior_fn=tracked_score, score_prior_weight=0.0,
    )
    assert call_count[0] == 0, (
        f"score_prior_fn called {call_count[0]} times despite weight=0"
    )

"""TDD spec for CFL-safe / stability-controlled DPS guidance (issue #46).

The binding failure mode for diffusion-prior FWI (InverseBench, ICLR 2025)
is a guidance step that pushes the sound-speed field past the value used
to set the solver time step, violating the CFL condition ``c·dt/dx ≤ C``
and blowing up the forward solve. The fix is a *stability-aware* guidance
limiter: a per-step trust region on ``Δc`` plus a hard clamp of the
guided field to a CFL-admissible ``[c_min, c_max]`` band.

Each test is written failing first, then minimum code to GREEN.
Unit-tests the limiter directly — no full 3D diffusion sample.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest


# --------------------------------------------------------------------------
# CFL admissible c_max helper
# --------------------------------------------------------------------------
def test_cfl_admissible_cmax_matches_condition():
    """c_max = C·dx/dt exactly inverts the CFL condition c·dt/dx ≤ C."""
    from brain_fwi.inference.diffusion import cfl_admissible_cmax

    dx, dt, C = 5e-4, 6e-8, 0.3
    c_max = cfl_admissible_cmax(dx, dt, cfl_number=C)
    assert float(c_max) == pytest.approx(C * dx / dt)
    # A field at exactly c_max sits on the CFL boundary.
    assert float(c_max * dt / dx) == pytest.approx(C, rel=1e-6)


# --------------------------------------------------------------------------
# (a) Clamp: guidance that would exceed c_max is pulled back to c_max.
# --------------------------------------------------------------------------
def test_guidance_clamped_to_cmax_never_violates_cfl():
    from brain_fwi.inference.diffusion import GuidanceLimiter, limit_guidance_step

    c_min, c_max = 1400.0, 1600.0
    limiter = GuidanceLimiter(c_min=c_min, c_max=c_max, max_delta=jnp.inf)

    c_prev = jnp.array([1500.0, 1550.0, 1500.0])
    # Proposed guided field overshoots the band on both ends.
    c_proposed = jnp.array([1800.0, 1595.0, 1200.0])

    c_new, info = limit_guidance_step(c_prev, c_proposed, limiter)

    assert jnp.all(c_new <= c_max + 1e-6), "must never exceed c_max (CFL)"
    assert jnp.all(c_new >= c_min - 1e-6), "must never drop below c_min"
    assert float(c_new[0]) == pytest.approx(c_max)   # overshoot pulled to c_max
    assert float(c_new[1]) == pytest.approx(1595.0)  # inside band, untouched
    assert float(c_new[2]) == pytest.approx(c_min)   # undershoot pulled to c_min
    assert bool(info.bound_any), "info should report the limiter bound"


# --------------------------------------------------------------------------
# (b) Trust region: per-step |Δc| is capped at max_delta.
# --------------------------------------------------------------------------
def test_trust_region_caps_per_step_change():
    from brain_fwi.inference.diffusion import GuidanceLimiter, limit_guidance_step

    max_delta = 10.0
    limiter = GuidanceLimiter(c_min=0.0, c_max=1e9, max_delta=max_delta)

    c_prev = jnp.array([1500.0, 1500.0, 1500.0])
    c_proposed = jnp.array([1500.0 + 100.0, 1500.0 - 250.0, 1500.0 + 3.0])

    c_new, info = limit_guidance_step(c_prev, c_proposed, limiter)

    delta = c_new - c_prev
    assert jnp.max(jnp.abs(delta)) <= max_delta + 1e-6
    assert float(delta[0]) == pytest.approx(max_delta)    # +100 capped to +10
    assert float(delta[1]) == pytest.approx(-max_delta)   # -250 capped to -10
    assert float(delta[2]) == pytest.approx(3.0)          # within trust region
    assert int(info.n_bound) >= 2


# --------------------------------------------------------------------------
# (c) Regression: a permissive limiter is a no-op.
# --------------------------------------------------------------------------
def test_permissive_limiter_is_identity():
    from brain_fwi.inference.diffusion import GuidanceLimiter, limit_guidance_step

    limiter = GuidanceLimiter(c_min=-jnp.inf, c_max=jnp.inf, max_delta=jnp.inf)
    c_prev = jnp.array([1400.0, 1500.0, 1600.0])
    c_proposed = jnp.array([1234.0, 1789.0, 1590.0])

    c_new, info = limit_guidance_step(c_prev, c_proposed, limiter)

    assert jnp.allclose(c_new, c_proposed)
    assert not bool(info.bound_any)
    assert int(info.n_bound) == 0


# --------------------------------------------------------------------------
# (c') Sampler-level regression: dps_sample unchanged when limiter off /
#      permissive. Fast: tiny dim, trivial score + likelihood, no 3D solve.
# --------------------------------------------------------------------------
def _tiny_dps(guidance_limiter):
    from brain_fwi.inference.diffusion import VPSDE, dps_sample

    sde = VPSDE()
    score_fn = lambda x, t: jnp.zeros_like(x)
    log_lik = lambda x0: -0.5 * jnp.sum(x0 ** 2)
    return dps_sample(
        score_fn, sde, log_lik,
        n_samples=4, dim=3, n_steps=3, zeta=0.5,
        key=jr.PRNGKey(7),
        guidance_limiter=guidance_limiter,
    )


def test_dps_sample_limiter_off_matches_no_limiter_kwarg():
    from brain_fwi.inference.diffusion import VPSDE, dps_sample

    sde = VPSDE()
    score_fn = lambda x, t: jnp.zeros_like(x)
    log_lik = lambda x0: -0.5 * jnp.sum(x0 ** 2)
    baseline = dps_sample(
        score_fn, sde, log_lik,
        n_samples=4, dim=3, n_steps=3, zeta=0.5, key=jr.PRNGKey(7),
    )
    off = _tiny_dps(None)
    assert np.allclose(np.asarray(baseline), np.asarray(off))


def test_dps_sample_permissive_limiter_matches_off():
    from brain_fwi.inference.diffusion import GuidanceLimiter

    permissive = GuidanceLimiter(c_min=-jnp.inf, c_max=jnp.inf, max_delta=jnp.inf)
    off = _tiny_dps(None)
    on = _tiny_dps(permissive)
    assert np.allclose(np.asarray(off), np.asarray(on))


def test_dps_sample_tight_limiter_keeps_state_in_band():
    """A tight band clamps every reverse-step state into [c_min, c_max]."""
    from brain_fwi.inference.diffusion import GuidanceLimiter

    limiter = GuidanceLimiter(c_min=-0.5, c_max=0.5, max_delta=jnp.inf)
    out = _tiny_dps(limiter)
    assert np.all(np.asarray(out) <= 0.5 + 1e-5)
    assert np.all(np.asarray(out) >= -0.5 - 1e-5)

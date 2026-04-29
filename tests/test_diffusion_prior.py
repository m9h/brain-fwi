"""TDD spec for Phase 3 score-based prior over SIREN weights.

Reference: docs/design/phase3_diffusion_prior.md.
Each test was written failing first, then minimum code to GREEN.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest


def test_score_mlp_returns_same_shape_as_input():
    """ScoreMLP(θ_t, t) → ℝ^D matching θ_t's shape, for D=8."""
    from brain_fwi.inference.diffusion import ScoreMLP

    net = ScoreMLP(dim=8, hidden=32, depth=2, key=jr.PRNGKey(0))
    theta_t = jnp.ones((8,))
    t = jnp.array(0.5)
    out = net(theta_t, t)
    assert out.shape == (8,), f"got {out.shape}, expected (8,)"
    assert jnp.all(jnp.isfinite(out))


def test_vp_sde_schedule_endpoints():
    """VP SDE: α(0)=1, σ(0)=0, α(1)≈0, σ(1)≈1, α²+σ²=1 ∀ t."""
    from brain_fwi.inference.diffusion import VPSDE

    sde = VPSDE(beta_min=0.1, beta_max=20.0)
    a0, s0 = sde.alpha(0.0), sde.sigma(0.0)
    a1, s1 = sde.alpha(1.0), sde.sigma(1.0)
    assert float(a0) == pytest.approx(1.0, abs=1e-6)
    assert float(s0) == pytest.approx(0.0, abs=1e-6)
    assert float(a1) < 0.01
    assert float(s1) > 0.99

    ts = jnp.linspace(0.0, 1.0, 50)
    a = jax.vmap(sde.alpha)(ts)
    s = jax.vmap(sde.sigma)(ts)
    assert jnp.allclose(a ** 2 + s ** 2, 1.0, atol=1e-5)


def test_vp_sde_alpha_decreasing_sigma_increasing():
    """α(t) is monotone decreasing, σ(t) is monotone increasing."""
    from brain_fwi.inference.diffusion import VPSDE

    sde = VPSDE(beta_min=0.1, beta_max=20.0)
    ts = jnp.linspace(0.0, 1.0, 50)
    a = jax.vmap(sde.alpha)(ts)
    s = jax.vmap(sde.sigma)(ts)
    assert jnp.all(jnp.diff(a) <= 1e-6)
    assert jnp.all(jnp.diff(s) >= -1e-6)


def test_dsm_loss_zero_for_perfect_score():
    """A score net that outputs exactly -ε/σ has DSM loss = 0."""
    from brain_fwi.inference.diffusion import VPSDE, dsm_loss_for_pair

    sde = VPSDE(beta_min=0.1, beta_max=20.0)
    rng = jr.PRNGKey(0)
    k1, k2 = jr.split(rng)
    theta = jr.normal(k1, (8,))
    eps = jr.normal(k2, (8,))
    t = jnp.array(0.4)
    sigma = sde.sigma(t)

    def perfect_score(theta_t, tt):
        # Reconstruct the noise that produced θ_t under VP and return
        # its negative, scaled by σ. Cheats: knows θ at training time.
        e = (theta_t - sde.alpha(tt) * theta) / sde.sigma(tt)
        return -e / sde.sigma(tt)

    loss = dsm_loss_for_pair(perfect_score, sde, theta, eps, t)
    assert float(loss) == pytest.approx(0.0, abs=1e-10)


def test_dsm_loss_positive_for_zero_score():
    """A score net that always outputs zero has positive DSM loss
    proportional to ‖ε‖²/σ²."""
    from brain_fwi.inference.diffusion import VPSDE, dsm_loss_for_pair

    sde = VPSDE(beta_min=0.1, beta_max=20.0)
    rng = jr.PRNGKey(1)
    k1, k2 = jr.split(rng)
    theta = jr.normal(k1, (8,))
    eps = jr.normal(k2, (8,))
    t = jnp.array(0.5)

    def zero_score(theta_t, tt):
        return jnp.zeros_like(theta_t)

    loss = dsm_loss_for_pair(zero_score, sde, theta, eps, t)
    expected = float(jnp.sum(eps ** 2) / sde.sigma(t) ** 2)
    assert float(loss) == pytest.approx(expected, rel=1e-5)


def test_score_net_recovers_analytic_score_on_unit_gaussian():
    """§7 step 1 toy gate: training on N(0, I) samples drives the
    score net's output to ≈ −θ_t at any t ∈ (0, 1].

    For variance-preserving SDE on a unit Gaussian, the perturbed
    marginal is still N(0, I) (since α² + σ² = 1), so the analytic
    score is ``∇ log p_t(θ_t) = −θ_t`` everywhere. A trained
    ScoreMLP on enough samples should match within a tight tolerance.
    """
    from brain_fwi.inference.diffusion import (
        ScoreMLP,
        VPSDE,
        train_score_matching,
    )

    sde = VPSDE(beta_min=0.1, beta_max=20.0)
    rng = jr.PRNGKey(0)
    k_data, k_net, k_train, k_eval = jr.split(rng, 4)

    samples = jr.normal(k_data, (1024, 2))
    net = ScoreMLP(dim=2, hidden=64, depth=3, key=k_net)

    trained, losses = train_score_matching(
        net, samples, sde,
        n_steps=2000, batch_size=64, learning_rate=3e-3,
        key=k_train,
    )
    assert losses[-1] < losses[0], "loss did not decrease"

    # Evaluate at a mid-noise level on fresh samples.
    theta_test = jr.normal(k_eval, (256, 2))
    t = jnp.array(0.5)
    eps = jr.normal(jr.PRNGKey(99), theta_test.shape)
    theta_t = sde.alpha(t) * theta_test + sde.sigma(t) * eps

    pred_score = jax.vmap(lambda x: trained(x, t))(theta_t)
    analytic_score = -theta_t

    err = jnp.linalg.norm(pred_score - analytic_score, axis=1)
    norm = jnp.linalg.norm(analytic_score, axis=1) + 1e-9
    median_rel_err = float(jnp.median(err / norm))
    assert median_rel_err < 0.20, (
        f"trained score median rel-err {median_rel_err:.3f} > 0.20"
    )

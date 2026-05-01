"""TDD spec for Phase 3 score-based prior over SIREN weights.

Reference: docs/design/phase3_diffusion_prior.md.
Each test was written failing first, then minimum code to GREEN.
"""

from __future__ import annotations

import equinox as eqx
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


def test_ddim_step_is_identity_when_t_next_equals_t():
    """A DDIM update with t_next == t leaves θ unchanged regardless
    of the score function — the canonical sanity test for any
    reverse-step implementation.
    """
    from brain_fwi.inference.diffusion import VPSDE, ddim_step

    sde = VPSDE(beta_min=0.1, beta_max=20.0)
    theta_t = jnp.array([0.3, -0.7, 1.2, 0.0])
    t = jnp.array(0.5)

    def arbitrary_score(x, tt):
        return -x  # any nonzero output should not matter

    theta_next = ddim_step(arbitrary_score, sde, theta_t, t, t)
    assert jnp.allclose(theta_next, theta_t, atol=1e-7)


def test_ddim_sample_returns_correct_shape_and_is_deterministic():
    """``ddim_sample(score, sde, n_samples=N, dim=D, ...)`` produces
    ``(N, D)`` output and is deterministic in the supplied PRNG key.
    """
    from brain_fwi.inference.diffusion import VPSDE, ddim_sample

    sde = VPSDE(beta_min=0.1, beta_max=20.0)

    def arbitrary_score(x, t):
        return -x

    out_a = ddim_sample(
        arbitrary_score, sde,
        n_samples=32, dim=4, n_steps=20, key=jr.PRNGKey(0),
    )
    out_b = ddim_sample(
        arbitrary_score, sde,
        n_samples=32, dim=4, n_steps=20, key=jr.PRNGKey(0),
    )
    out_c = ddim_sample(
        arbitrary_score, sde,
        n_samples=32, dim=4, n_steps=20, key=jr.PRNGKey(7),
    )
    assert out_a.shape == (32, 4)
    assert jnp.allclose(out_a, out_b), "different runs with same key diverged"
    assert not jnp.allclose(out_a, out_c), "different keys produced same output"


def test_em_sample_returns_correct_shape_and_is_deterministic():
    """Stochastic Euler-Maruyama reverse sampler API + determinism."""
    from brain_fwi.inference.diffusion import VPSDE, em_sample

    sde = VPSDE(beta_min=0.1, beta_max=20.0)

    def arbitrary_score(x, t):
        return -x

    out_a = em_sample(
        arbitrary_score, sde,
        n_samples=32, dim=4, n_steps=100, key=jr.PRNGKey(0),
    )
    out_b = em_sample(
        arbitrary_score, sde,
        n_samples=32, dim=4, n_steps=100, key=jr.PRNGKey(0),
    )
    out_c = em_sample(
        arbitrary_score, sde,
        n_samples=32, dim=4, n_steps=100, key=jr.PRNGKey(7),
    )
    assert out_a.shape == (32, 4)
    assert jnp.allclose(out_a, out_b)
    assert not jnp.allclose(out_a, out_c)


def test_em_sample_with_perfect_score_recovers_unit_gaussian_moments():
    """§7 step 2 gate analogue for the toy prior: with the analytic
    score on unit Gaussian, Euler-Maruyama samples should have
    sample mean ≈ 0 and sample variance ≈ 1 (within the discretization
    error budget)."""
    from brain_fwi.inference.diffusion import VPSDE, em_sample

    sde = VPSDE(beta_min=0.1, beta_max=20.0)

    def perfect_score(theta_t, t):
        return -theta_t

    samples = em_sample(
        perfect_score, sde,
        n_samples=2048, dim=2, n_steps=400, key=jr.PRNGKey(0),
        t_min=0.01,
    )
    mean = jnp.mean(samples, axis=0)
    var = jnp.var(samples, axis=0)
    assert jnp.all(jnp.abs(mean) < 0.15), f"sample mean off: {mean}"
    assert jnp.all(jnp.abs(var - 1.0) < 0.30), f"sample var off: {var}"


def test_score_prior_grad_term_returns_negative_weighted_score():
    """``score_prior_grad_term`` returns ``−λ · s_φ(θ, t_eps)``.

    Sign convention: this is the term to ADD to ``∇L_data`` so the
    composite gradient minimises ``NLL = −log p(d|θ) − log p(θ)``.
    The score is ``∇ log p(θ)`` (uphill on density), so subtracting
    it from the data gradient turns gradient descent into MAP.
    """
    from brain_fwi.inference.diffusion import score_prior_grad_term

    def score_fn(theta, t):
        return -theta  # analytic score for N(0, I)

    theta = jnp.array([1.0, -2.0, 0.5])
    out = score_prior_grad_term(score_fn, theta, t_eps=0.01, weight=0.5)
    expected = -0.5 * (-theta)
    assert jnp.allclose(out, expected)


def test_score_prior_grad_term_zero_when_weight_zero():
    """A weight of 0 makes the regulariser term vanish."""
    from brain_fwi.inference.diffusion import score_prior_grad_term

    def score_fn(theta, t):
        return jnp.ones_like(theta) * 99.0

    theta = jnp.array([1.0, -2.0, 0.5])
    out = score_prior_grad_term(score_fn, theta, t_eps=0.01, weight=0.0)
    assert jnp.allclose(out, jnp.zeros_like(theta))


def test_compose_siren_grad_with_zero_score_returns_grads_unchanged():
    """With a zero score function, the composite gradient pytree is
    bit-identical to the input gradient pytree."""
    from brain_fwi.inference.diffusion import compose_siren_grad_with_score_prior
    from brain_fwi.inversion.param_field import SIREN

    field = SIREN(
        in_dim=3, hidden_dim=8, n_hidden=2, out_dim=1,
        omega_0=30.0, key=jr.PRNGKey(0),
    )
    grads = jax.tree.map(
        lambda x: jnp.ones_like(x) if eqx.is_inexact_array(x) else x,
        eqx.filter(field, eqx.is_inexact_array),
    )

    def zero_score(theta, t):
        return jnp.zeros_like(theta)

    composed = compose_siren_grad_with_score_prior(
        grads, field, zero_score, weight=0.5, t_eps=0.01,
    )
    leaves_in = jax.tree.leaves(grads)
    leaves_out = jax.tree.leaves(composed)
    for a, b in zip(leaves_in, leaves_out):
        if a is None or b is None:
            continue
        assert jnp.allclose(jnp.asarray(a), jnp.asarray(b))


def test_dps_sample_with_zero_zeta_matches_em_sample():
    """DPS with the likelihood weight ``ζ = 0`` reduces exactly to the
    unconditional EM sampler — the canonical sanity check on the
    posterior-sampling extension."""
    from brain_fwi.inference.diffusion import VPSDE, dps_sample, em_sample

    sde = VPSDE(beta_min=0.1, beta_max=20.0)

    def perfect_score(theta_t, t):
        return -theta_t

    def trivial_log_lik(theta):
        return jnp.array(0.0)  # log p(d|θ) = const → score_lik = 0 anyway

    em = em_sample(
        perfect_score, sde,
        n_samples=64, dim=2, n_steps=50, key=jr.PRNGKey(0),
    )
    dps = dps_sample(
        perfect_score, sde, trivial_log_lik,
        n_samples=64, dim=2, n_steps=50, zeta=0.0, key=jr.PRNGKey(0),
    )
    assert jnp.allclose(em, dps, atol=1e-5), (
        f"DPS with ζ=0 diverged from EM (max diff "
        f"{float(jnp.max(jnp.abs(em - dps))):.2e})"
    )


def test_dps_sample_recovers_posterior_mean_on_gaussian_pair():
    """Conjugate-Gaussian sanity: prior N(0, I) + likelihood
    N(d | θ, σ² I) ⇒ posterior N(d/(1+σ²), σ²/(1+σ²) I).

    DPS samples should have mean within ±0.20 of d/(1+σ²).
    """
    from brain_fwi.inference.diffusion import VPSDE, dps_sample

    sde = VPSDE(beta_min=0.1, beta_max=20.0)

    def perfect_score(theta_t, t):
        return -theta_t

    d_obs = jnp.array([1.5, -1.0])
    sigma2 = 0.1

    def log_lik(theta):
        return -0.5 * jnp.sum((d_obs - theta) ** 2) / sigma2

    samples = dps_sample(
        perfect_score, sde, log_lik,
        n_samples=2048, dim=2, n_steps=400, zeta=1.0,
        key=jr.PRNGKey(0), t_min=0.01,
    )
    expected_mean = d_obs / (1.0 + sigma2)
    sample_mean = jnp.mean(samples, axis=0)
    err = jnp.abs(sample_mean - expected_mean)
    assert jnp.all(err < 0.20), (
        f"DPS posterior mean off: got {sample_mean}, "
        f"expected ≈ {expected_mean}"
    )


def test_compose_siren_grad_recovers_correct_total_norm_shift():
    """Score = -θ (analytic for unit Gaussian) ⇒ composite_grad =
    grads + λ·θ (after sign analysis). The total l2-norm of the
    composite grads' inexact leaves should differ from the input by
    the predicted amount.
    """
    import equinox as eqx_mod
    from brain_fwi.inference.diffusion import compose_siren_grad_with_score_prior
    from brain_fwi.inversion.param_field import SIREN

    field = SIREN(
        in_dim=3, hidden_dim=8, n_hidden=2, out_dim=1,
        omega_0=30.0, key=jr.PRNGKey(0),
    )
    grads = jax.tree.map(
        lambda x: jnp.zeros_like(x) if eqx.is_inexact_array(x) else x,
        eqx.filter(field, eqx_mod.is_inexact_array),
    )

    def neg_theta_score(theta, t):
        return -theta  # → grad term = -λ · (-θ) = +λ θ

    weight = 0.3
    composed = compose_siren_grad_with_score_prior(
        grads, field, neg_theta_score, weight=weight, t_eps=0.01,
    )
    # Gather the inexact leaves of the field as a flat vector.
    field_leaves = jax.tree.leaves(eqx.filter(field, eqx_mod.is_inexact_array))
    theta_flat = jnp.concatenate([jnp.asarray(l).ravel() for l in field_leaves])
    expected_l2 = float(jnp.linalg.norm(weight * theta_flat))

    composed_leaves = jax.tree.leaves(eqx.filter(composed, eqx_mod.is_inexact_array))
    composed_flat = jnp.concatenate(
        [jnp.asarray(l).ravel() for l in composed_leaves]
    )
    assert float(jnp.linalg.norm(composed_flat)) == pytest.approx(
        expected_l2, rel=1e-5,
    )

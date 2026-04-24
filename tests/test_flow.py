"""Tests for brain_fwi.inference.flow — conditional NPE flow wrappers.

Skipped locally on platforms where flowjax can't install (e.g., the
macOS 26 x86_64 jaxlib-wheel gap). CI runs on Linux where wheels exist,
and Modal pulls fresh from pyproject.toml on each run.
"""

import pytest

flowjax = pytest.importorskip("flowjax")

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from brain_fwi.inference.flow import ConditionalFlow, train_npe


class TestConditionalFlow:
    def test_init_stores_dims(self):
        flow = ConditionalFlow(theta_dim=5, d_dim=3, key=jr.PRNGKey(0))
        assert flow.theta_dim == 5
        assert flow.d_dim == 3

    def test_log_prob_returns_scalar(self):
        flow = ConditionalFlow(theta_dim=3, d_dim=2, key=jr.PRNGKey(0))
        theta = jnp.zeros(3)
        d = jnp.zeros(2)
        lp = flow.log_prob(theta, d)
        assert lp.shape == ()
        assert jnp.isfinite(lp)

    def test_log_prob_vmaps_over_batch(self):
        flow = ConditionalFlow(theta_dim=3, d_dim=2, key=jr.PRNGKey(0))
        theta = jnp.zeros((10, 3))
        d = jnp.zeros((10, 2))
        lp = jax.vmap(flow.log_prob)(theta, d)
        assert lp.shape == (10,)
        assert jnp.all(jnp.isfinite(lp))

    def test_sample_returns_correct_shape(self):
        flow = ConditionalFlow(theta_dim=3, d_dim=2, key=jr.PRNGKey(0))
        d = jnp.zeros(2)
        samples = flow.sample(d, jr.PRNGKey(1), n_samples=5)
        assert samples.shape == (5, 3)
        assert jnp.all(jnp.isfinite(samples))


class TestTrainNPE:
    def test_training_reduces_nll_on_toy_problem(self):
        """Toy linear-Gaussian: theta ~ N(0,1), d = 2*theta + small noise.

        After training, the learned q(theta|d) should have much lower
        negative log-likelihood on held-out pairs than the untrained flow.
        """
        rng = np.random.default_rng(0)
        n = 500
        theta_train = rng.standard_normal((n, 2)).astype(np.float32)
        d_train = 2.0 * theta_train + 0.1 * rng.standard_normal((n, 2)).astype(np.float32)
        theta_test = rng.standard_normal((64, 2)).astype(np.float32)
        d_test = 2.0 * theta_test + 0.1 * rng.standard_normal((64, 2)).astype(np.float32)

        flow = ConditionalFlow(theta_dim=2, d_dim=2, key=jr.PRNGKey(0), n_transforms=3)

        initial_nll = float(
            -jnp.mean(jax.vmap(flow.log_prob)(jnp.asarray(theta_test), jnp.asarray(d_test)))
        )

        trained, losses = train_npe(
            flow,
            theta=jnp.asarray(theta_train),
            d=jnp.asarray(d_train),
            key=jr.PRNGKey(1),
            n_steps=200,
            learning_rate=1e-3,
            batch_size=64,
        )

        final_nll = float(
            -jnp.mean(jax.vmap(trained.log_prob)(jnp.asarray(theta_test), jnp.asarray(d_test)))
        )
        assert final_nll < initial_nll - 0.5, (
            f"NLL should have dropped by >0.5 nats; "
            f"initial={initial_nll:.3f}, final={final_nll:.3f}"
        )
        assert len(losses) == 200
        assert all(np.isfinite(losses))

    def test_trained_conditional_mean_tracks_data(self):
        """q(theta | d) should concentrate around theta = d / 2 (inverse of d=2*theta)."""
        rng = np.random.default_rng(1)
        n = 400
        theta_train = rng.standard_normal((n, 1)).astype(np.float32)
        d_train = 2.0 * theta_train + 0.05 * rng.standard_normal((n, 1)).astype(np.float32)

        flow = ConditionalFlow(theta_dim=1, d_dim=1, key=jr.PRNGKey(3), n_transforms=4)
        trained, _ = train_npe(
            flow,
            theta=jnp.asarray(theta_train),
            d=jnp.asarray(d_train),
            key=jr.PRNGKey(4),
            n_steps=300,
            learning_rate=2e-3,
            batch_size=64,
        )

        # Probe a few d values and check the posterior mean is near d/2
        for d_val in [-1.5, 0.0, 1.5]:
            d_probe = jnp.array([d_val])
            samples = trained.sample(d_probe, jr.PRNGKey(99), n_samples=500)
            posterior_mean = float(jnp.mean(samples))
            expected = d_val / 2.0
            assert abs(posterior_mean - expected) < 0.3, (
                f"d={d_val}: posterior mean {posterior_mean:.2f} "
                f"far from expected {expected:.2f}"
            )

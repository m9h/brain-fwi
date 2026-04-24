"""Tests for brain_fwi.inference.sbc — simulation-based calibration.

Covers the two entry points of the module:

  - ``sbc_ranks`` computes per-pair per-dim ranks from a posterior
    sampler. Values must live in [0, L] and shape matches
    ``(n_pairs, theta_dim)``.
  - ``calibration_statistic`` runs a chi-squared test on the rank
    histogram. A well-calibrated posterior's ranks are uniform on
    [0, L], so chi2 p-value should not reject the null; a biased
    posterior's ranks are concentrated, so the test should reject.

Tests use a mock sampler — no flowjax dependency. Keeps the SBC
machinery fully testable on any platform.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from brain_fwi.inference.sbc import calibration_statistic, sbc_ranks


# ---------------------------------------------------------------------------
# Mock samplers
# ---------------------------------------------------------------------------

@dataclass
class GaussianSampler:
    """Mock ``q(theta | d)`` that draws from ``N(mean_fn(d), scale*I)``."""
    mean_fn: "callable"
    scale: float

    def sample(self, d: jnp.ndarray, key: jax.Array, n_samples: int) -> jnp.ndarray:
        mean = self.mean_fn(d)
        noise = jr.normal(key, (n_samples,) + mean.shape)
        return mean + self.scale * noise


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSBCRanks:
    def test_shape(self):
        theta = jnp.zeros((10, 3))
        d = jnp.zeros((10, 3))
        sampler = GaussianSampler(mean_fn=lambda d: d, scale=1.0)
        ranks = sbc_ranks(sampler, theta, d, n_posterior_samples=50, key=jr.PRNGKey(0))
        assert ranks.shape == (10, 3)

    def test_ranks_in_valid_range(self):
        theta = jnp.zeros((20, 2))
        d = jnp.zeros((20, 2))
        sampler = GaussianSampler(mean_fn=lambda d: d, scale=1.0)
        ranks = sbc_ranks(sampler, theta, d, n_posterior_samples=64, key=jr.PRNGKey(1))
        assert int(ranks.min()) >= 0
        assert int(ranks.max()) <= 64

    def test_calibrated_posterior_gives_near_uniform_ranks(self):
        """Linear-Gaussian generative model with known true posterior:

            prior:       theta ~ N(0, 1)
            observation: d = theta + eps,  eps ~ N(0, 1)
            posterior:   theta | d ~ N(d / 2, sqrt(1/2))

        When the sampler matches this true posterior exactly, SBC ranks
        should be uniform on [0, L]. Mean ≈ L/2, std ≈ L/sqrt(12).
        """
        rng = np.random.default_rng(42)
        n_pairs, theta_dim, L = 500, 1, 100
        theta_star = rng.standard_normal((n_pairs, theta_dim)).astype(np.float32)
        eps = rng.standard_normal((n_pairs, theta_dim)).astype(np.float32)
        d = theta_star + eps

        true_posterior = GaussianSampler(
            mean_fn=lambda d: d / 2.0, scale=float(1.0 / np.sqrt(2.0)),
        )
        ranks = np.asarray(sbc_ranks(
            true_posterior, jnp.asarray(theta_star), jnp.asarray(d),
            n_posterior_samples=L, key=jr.PRNGKey(3),
        ))
        mean_rank = ranks.mean()
        std_rank = ranks.std()
        assert abs(mean_rank - L / 2) < L * 0.05, (
            f"mean rank {mean_rank:.1f} far from L/2 = {L/2}"
        )
        assert abs(std_rank - L / np.sqrt(12)) < L * 0.1

    def test_biased_posterior_gives_skewed_ranks(self):
        """Same generative model as the calibrated test, but the sampler's
        mean is shifted up by +2 (well above the true posterior mean).
        Posterior samples systematically exceed theta*, so
        ``count(samples < theta_star)`` is small → ranks skew LOW.
        """
        rng = np.random.default_rng(42)
        n_pairs, theta_dim, L = 500, 1, 100
        theta_star = rng.standard_normal((n_pairs, theta_dim)).astype(np.float32)
        eps = rng.standard_normal((n_pairs, theta_dim)).astype(np.float32)
        d = theta_star + eps

        biased_high = GaussianSampler(
            mean_fn=lambda d: d / 2.0 + 2.0, scale=float(1.0 / np.sqrt(2.0)),
        )
        ranks = np.asarray(sbc_ranks(
            biased_high, jnp.asarray(theta_star), jnp.asarray(d),
            n_posterior_samples=L, key=jr.PRNGKey(4),
        ))
        assert ranks.mean() < L * 0.2, (
            f"biased-high posterior should give LOW ranks "
            f"(samples exceed theta*); got mean {ranks.mean():.1f}"
        )


class TestCalibrationStatistic:
    def test_uniform_ranks_pass_chi2(self):
        """Genuinely uniform ranks should not trigger the chi-squared test."""
        rng = np.random.default_rng(0)
        ranks = rng.integers(0, 101, size=(500, 1))
        stat = calibration_statistic(ranks, n_bins=10)
        assert stat["p_value"] > 0.05
        assert stat["is_calibrated"] is True

    def test_heavily_skewed_ranks_fail_chi2(self):
        """All ranks in the bottom quarter should decisively fail."""
        ranks = np.zeros((500, 1), dtype=np.int32)  # all ranks = 0
        stat = calibration_statistic(ranks, n_bins=10)
        assert stat["p_value"] < 0.001
        assert stat["is_calibrated"] is False

    def test_multidim_reported_per_dim(self):
        """Per-dimension chi2 stats should be reported independently."""
        n_pairs, L = 400, 100
        uniform_col = np.random.default_rng(0).integers(0, L + 1, size=n_pairs)
        skewed_col = np.zeros(n_pairs, dtype=np.int32)
        ranks = np.stack([uniform_col, skewed_col], axis=-1)
        stat = calibration_statistic(ranks, n_bins=10)
        assert len(stat["per_dim"]) == 2
        assert stat["per_dim"][0]["is_calibrated"] is True
        assert stat["per_dim"][1]["is_calibrated"] is False
        # Aggregate 'is_calibrated' is all-per-dim
        assert stat["is_calibrated"] is False

    def test_rejects_non_integer_ranks(self):
        ranks = np.array([[0.5, 1.5]], dtype=np.float32)
        with pytest.raises(ValueError, match="integer"):
            calibration_statistic(ranks)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            calibration_statistic(np.zeros(10, dtype=np.int32))

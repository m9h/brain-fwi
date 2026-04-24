"""Simulation-based calibration (Talts et al. 2018).

The check: for a trained amortised posterior ``q(θ | d)`` and a set of
``(θ*, d*)`` pairs drawn from the same joint distribution used for
training (held-out), the rank of each ``θ*`` among L posterior samples
from ``q(· | d*)`` should be uniform on ``[0, L]`` if the posterior is
well-calibrated. Any non-uniformity indicates a calibration bug.

This module is a pure-stats / sampler-agnostic implementation: it
accepts any object with ``.sample(d, key, n_samples) -> (n_samples,
theta_dim)``. That keeps SBC testable without flowjax and reusable for
Phase 3 diffusion posteriors.

References:
    Talts, Betancourt, Simpson, Vehtari, Gelman (2018). "Validating
    Bayesian inference algorithms with simulation-based calibration."
    arXiv:1804.06788.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy import stats


class PosteriorSampler(Protocol):
    """Duck-typed interface for anything that can draw from ``q(θ | d)``."""

    def sample(self, d: jnp.ndarray, key: jax.Array, n_samples: int) -> jnp.ndarray:
        ...


def sbc_ranks(
    sampler: PosteriorSampler,
    theta_held_out: jnp.ndarray,
    d_held_out: jnp.ndarray,
    n_posterior_samples: int,
    key: jax.Array,
) -> jnp.ndarray:
    """Compute SBC ranks for held-out ``(θ*, d*)`` pairs.

    For each pair ``i`` and each θ-dim ``j``, returns the count of
    posterior samples strictly less than the true ``θ*_{i,j}``.
    A uniform distribution of ranks on ``[0, L]`` indicates a
    calibrated posterior.

    Args:
        sampler: Object with ``.sample(d, key, n_samples)``.
        theta_held_out: ``(N, theta_dim)`` ground-truth parameters.
        d_held_out: ``(N, d_dim)`` corresponding observations.
        n_posterior_samples: L — number of posterior samples per pair.
        key: JAX PRNG key, split per pair internally.

    Returns:
        ``(N, theta_dim)`` integer rank array in ``[0, L]``.
    """
    n_pairs = theta_held_out.shape[0]
    if d_held_out.shape[0] != n_pairs:
        raise ValueError(
            f"theta_held_out and d_held_out must have matching leading dim; "
            f"got {n_pairs} vs {d_held_out.shape[0]}"
        )

    keys = jr.split(key, n_pairs)
    theta_dim = theta_held_out.shape[1] if theta_held_out.ndim > 1 else 1
    ranks = np.empty((n_pairs, theta_dim), dtype=np.int32)

    for i in range(n_pairs):
        samples = np.asarray(
            sampler.sample(d_held_out[i], keys[i], n_posterior_samples)
        )
        theta_star = np.asarray(theta_held_out[i])
        # per-dim rank = count of posterior samples strictly less than theta*
        ranks[i] = np.sum(samples < theta_star[None, :], axis=0)

    return jnp.asarray(ranks)


def calibration_statistic(
    ranks: np.ndarray,
    n_bins: int = 20,
) -> Dict[str, Any]:
    """Chi-squared test of rank uniformity (per dim and aggregate).

    For each θ-dim, bin the ranks into ``n_bins`` equal-width bins and
    run a chi-squared goodness-of-fit test against the uniform
    distribution. A well-calibrated posterior should *not* reject at
    the 5% level — a failed calibration is a significant p-value.

    Args:
        ranks: ``(N, theta_dim)`` integer ranks from :func:`sbc_ranks`.
        n_bins: Number of histogram bins. Heuristic: pick so each bin
            has at least 5 expected counts (``N / n_bins >= 5``).

    Returns:
        Dict with ``'per_dim'`` (list of per-dimension stats) and
        aggregate ``'chi2'``, ``'dof'``, ``'p_value'``, and
        ``'is_calibrated'`` (True iff every dim passes at 5%).
    """
    ranks = np.asarray(ranks)
    if ranks.ndim != 2:
        raise ValueError(
            f"ranks must be 2-D (n_pairs, theta_dim); got shape {ranks.shape}"
        )
    if not np.issubdtype(ranks.dtype, np.integer):
        raise ValueError(
            f"ranks must be integer-valued; got dtype {ranks.dtype}"
        )

    n_pairs, theta_dim = ranks.shape
    L = int(ranks.max()) + 1  # rank range is [0, L_effective]

    per_dim: List[Dict[str, Any]] = []
    worst_p = 1.0
    agg_chi2 = 0.0

    for j in range(theta_dim):
        # bin edges span [0, L] inclusive
        edges = np.linspace(0, L, n_bins + 1)
        counts, _ = np.histogram(ranks[:, j], bins=edges)
        expected = n_pairs / n_bins
        chi2_stat = float(np.sum((counts - expected) ** 2 / expected))
        dof = n_bins - 1
        p = float(stats.chi2.sf(chi2_stat, dof))
        per_dim.append({
            "dim": j,
            "chi2": chi2_stat,
            "dof": dof,
            "p_value": p,
            "is_calibrated": bool(p >= 0.05),
            "bin_counts": counts.tolist(),
        })
        agg_chi2 += chi2_stat
        worst_p = min(worst_p, p)

    agg_dof = theta_dim * (n_bins - 1)
    agg_p = float(stats.chi2.sf(agg_chi2, agg_dof))

    return {
        "per_dim": per_dim,
        "chi2": agg_chi2,
        "dof": agg_dof,
        "p_value": agg_p,
        "is_calibrated": bool(worst_p >= 0.05),
        "n_pairs": n_pairs,
        "n_bins": n_bins,
        "rank_max": L,
    }

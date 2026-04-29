"""Score-based prior over SIREN-weight θ (Phase 3).

Reference: ``docs/design/phase3_diffusion_prior.md``.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class VPSDE(NamedTuple):
    """Variance-preserving SDE schedule (Song et al. 2021).

    With linear ``β(t) = β_min + t·(β_max − β_min)``::

        α(t) = exp(-½ ∫₀ᵗ β(s) ds)
             = exp(-½(β_min · t + ½(β_max − β_min) · t²))
        σ²(t) = 1 − α²(t)

    Phase 3 §3 default schedule. ``α²+σ²=1`` by construction.
    """

    beta_min: float = 0.1
    beta_max: float = 20.0

    def beta(self, t: jax.Array) -> jax.Array:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha(self, t: jax.Array) -> jax.Array:
        # ∫₀ᵗ β(s) ds = β_min·t + ½(β_max − β_min)·t²
        integ = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return jnp.exp(-0.5 * integ)

    def sigma(self, t: jax.Array) -> jax.Array:
        return jnp.sqrt(1.0 - self.alpha(t) ** 2)


class ScoreMLP(eqx.Module):
    """MLP score net ``s_φ(θ_t, t) → ℝ^D``.

    Time is encoded via a sinusoidal embedding and concatenated with
    ``θ_t`` before the MLP body. Phase 3 §2 §MVP architecture.
    """

    body: eqx.nn.MLP
    dim: int = eqx.field(static=True)
    time_embed_dim: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        *,
        hidden: int = 128,
        depth: int = 4,
        time_embed_dim: int = 16,
        key: jax.Array,
    ):
        self.dim = int(dim)
        self.time_embed_dim = int(time_embed_dim)
        self.body = eqx.nn.MLP(
            in_size=dim + time_embed_dim,
            out_size=dim,
            width_size=hidden,
            depth=depth,
            key=key,
        )

    def _time_embed(self, t: jax.Array) -> jax.Array:
        half = self.time_embed_dim // 2
        freqs = jnp.exp(
            -jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / max(half - 1, 1)
        )
        ang = jnp.atleast_1d(t).astype(jnp.float32) * freqs
        emb = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)])
        if emb.shape[0] < self.time_embed_dim:
            emb = jnp.concatenate(
                [emb, jnp.zeros(self.time_embed_dim - emb.shape[0])]
            )
        return emb

    def __call__(self, theta_t: jax.Array, t: jax.Array) -> jax.Array:
        t_emb = self._time_embed(t)
        x = jnp.concatenate([theta_t, t_emb])
        return self.body(x)


def dsm_loss_for_pair(
    score_fn,
    sde: VPSDE,
    theta: jax.Array,
    eps: jax.Array,
    t: jax.Array,
) -> jax.Array:
    """Denoising-score-matching loss for one ``(θ, ε, t)`` triple.

    ``L = ‖s_φ(θ_t, t) + ε/σ(t)‖²`` where ``θ_t = α·θ + σ·ε``.

    Args:
        score_fn: callable ``(θ_t, t) → ℝ^D``. Either a trained
            ``ScoreMLP`` or any closure with that signature.
        sde: schedule providing ``α(t)`` and ``σ(t)``.
        theta: clean ``θ`` sample of shape ``(D,)``.
        eps: standard-normal noise of shape ``(D,)``.
        t: scalar time in ``[0, 1]``.

    Returns:
        Scalar loss for this triple.
    """
    theta_t = sde.alpha(t) * theta + sde.sigma(t) * eps
    target = eps / sde.sigma(t)
    pred = score_fn(theta_t, t)
    return jnp.sum((pred + target) ** 2)

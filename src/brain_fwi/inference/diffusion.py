"""Score-based prior over SIREN-weight θ (Phase 3).

Reference: ``docs/design/phase3_diffusion_prior.md``.
"""

from __future__ import annotations

from typing import List, NamedTuple, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax


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


def train_score_matching(
    model: ScoreMLP,
    samples: jax.Array,
    sde: VPSDE,
    *,
    n_steps: int,
    batch_size: int,
    learning_rate: float = 1e-3,
    key: jax.Array,
    t_min: float = 1e-3,
) -> Tuple[ScoreMLP, List[float]]:
    """Adam-trained denoising-score-matching loop.

    Per step: sample a batch of clean ``θ`` from ``samples``, draw fresh
    ``ε ~ N(0, I)`` and ``t ~ U(t_min, 1)``, take a gradient step on the
    averaged DSM loss. ``t_min > 0`` avoids the singular ``σ(0) = 0``
    in the loss target.

    Args:
        model: ScoreMLP to train (returned updated).
        samples: ``(N, D)`` clean θ samples.
        sde: noise schedule.
        n_steps: number of gradient steps.
        batch_size: per-step batch size; sampled with replacement.
        learning_rate: Adam learning rate.
        key: PRNG key (split internally).
        t_min: lower clamp on t to avoid σ→0.

    Returns:
        (trained_model, loss_history)
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    n = int(samples.shape[0])

    def batch_loss(m, theta_batch, eps_batch, t_batch):
        # Vectorise over the batch dim. Weight by σ²(t) (the design-
        # doc λ(t) choice) so the target -ε/σ does not blow up the loss
        # near t=0 — without weighting, training is unstable.
        per_sample = jax.vmap(
            lambda th, ee, tt: dsm_loss_for_pair(m, sde, th, ee, tt)
        )(theta_batch, eps_batch, t_batch)
        weights = sde.sigma(t_batch) ** 2
        return jnp.mean(weights * per_sample)

    @eqx.filter_jit
    def step(m, st, theta_batch, eps_batch, t_batch):
        loss, grads = eqx.filter_value_and_grad(batch_loss)(
            m, theta_batch, eps_batch, t_batch,
        )
        updates, st = optimizer.update(grads, st)
        m = eqx.apply_updates(m, updates)
        return m, st, loss

    losses: List[float] = []
    for _ in range(n_steps):
        key, k_idx, k_eps, k_t = jr.split(key, 4)
        idx = jr.randint(k_idx, (batch_size,), 0, n)
        theta_batch = samples[idx]
        eps_batch = jr.normal(k_eps, theta_batch.shape)
        t_batch = jr.uniform(k_t, (batch_size,), minval=t_min, maxval=1.0)
        model, opt_state, loss = step(
            model, opt_state, theta_batch, eps_batch, t_batch,
        )
        losses.append(float(loss))

    return model, losses

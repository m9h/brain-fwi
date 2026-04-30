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


def ddim_step(
    score_fn,
    sde: VPSDE,
    theta_t: jax.Array,
    t: jax.Array,
    t_next: jax.Array,
) -> jax.Array:
    """One deterministic DDIM update from ``t`` to ``t_next``.

    Tweedie-denoise to ``θ̂_0`` using the score, then renoise to the
    next time level::

        θ̂_0          = (θ_t + σ(t)² · s) / α(t)
        ε̂            = -σ(t) · s             # implied noise
        θ_{t_next}   = α(t_next) · θ̂_0 + σ(t_next) · ε̂

    Substituting:

        θ_{t_next} = (α(t_next) / α(t)) · θ_t
                     + (α(t_next)·σ(t)² / α(t) − σ(t_next)·σ(t)) · s

    With ``t_next = t`` this collapses to identity by construction.
    """
    s = score_fn(theta_t, t)
    a_t, s_t = sde.alpha(t), sde.sigma(t)
    a_n, s_n = sde.alpha(t_next), sde.sigma(t_next)
    theta_hat_0 = (theta_t + (s_t ** 2) * s) / a_t
    eps_hat = -s_t * s
    return a_n * theta_hat_0 + s_n * eps_hat


def ddim_sample(
    score_fn,
    sde: VPSDE,
    *,
    n_samples: int,
    dim: int,
    n_steps: int,
    key: jax.Array,
    t_min: float = 1e-3,
    t_max: float = 1.0,
) -> jax.Array:
    """Deterministic DDIM reverse trajectory from ``t_max`` to ``t_min``.

    Draws ``n_samples`` initial points from ``N(0, I)`` at ``t_max``
    and runs ``n_steps`` DDIM updates along a uniform ``t``-grid down
    to ``t_min``. Output shape ``(n_samples, dim)``.

    Per-sample updates are vectorised with ``jax.vmap`` so the score
    network sees one ``θ`` at a time (matching its training-time
    signature).
    """
    theta = jr.normal(key, (n_samples, dim))
    ts = jnp.linspace(t_max, t_min, n_steps + 1)

    def body(theta, i):
        t = ts[i]
        t_next = ts[i + 1]
        new_theta = jax.vmap(
            lambda x: ddim_step(score_fn, sde, x, t, t_next)
        )(theta)
        return new_theta, None

    final, _ = jax.lax.scan(body, theta, jnp.arange(n_steps))
    return final


def em_sample(
    score_fn,
    sde: VPSDE,
    *,
    n_samples: int,
    dim: int,
    n_steps: int,
    key: jax.Array,
    t_min: float = 1e-3,
    t_max: float = 1.0,
) -> jax.Array:
    """Stochastic Euler-Maruyama sampler on the reverse VP SDE.

    Reverse-time SDE for VP: ``dθ = [-½β·θ − β·s] dt + √β dw̄``.
    Discretised backward Euler over ``n_steps`` uniform t-grid points
    from ``t_max`` down to ``t_min``::

        Δt    = t − t_next
        drift = -½β(t)·θ − β(t)·s_φ(θ, t)
        θ_next = θ − drift·Δt + √(β(t)·Δt) · z,   z ~ N(0, I)

    Output shape ``(n_samples, dim)``.
    """
    key, subkey = jr.split(key)
    theta = jr.normal(subkey, (n_samples, dim))
    ts = jnp.linspace(t_max, t_min, n_steps + 1)

    def body(carry, i):
        theta, key = carry
        t, t_next = ts[i], ts[i + 1]
        dt = t - t_next
        beta_t = sde.beta(t)

        s = jax.vmap(lambda x: score_fn(x, t))(theta)
        drift = -0.5 * beta_t * theta - beta_t * s
        key, sk = jr.split(key)
        z = jr.normal(sk, theta.shape)
        new_theta = theta - drift * dt + jnp.sqrt(beta_t * dt) * z
        return (new_theta, key), None

    (final, _), _ = jax.lax.scan(body, (theta, key), jnp.arange(n_steps))
    return final


def score_prior_grad_term(
    score_fn,
    theta: jax.Array,
    *,
    t_eps: float = 0.01,
    weight: float = 1.0,
) -> jax.Array:
    """MAP-FWI regulariser term: ``−λ · s_φ(θ, t_eps)``.

    ADD this to the data gradient to turn FWI's gradient descent
    into MAP estimation:

        effective_grad = ∇L_data − λ · s_φ(θ, t_eps)
                       = ∇L_data + score_prior_grad_term(...)

    The score is ``∇ log p(θ)`` (points uphill on density), so
    subtracting weighted score from descent gradient pushes the
    optimisation toward higher-density θ regions — Phase 3 §4.

    ``t_eps`` is the small noise level at which to evaluate the
    score; the network was trained with ``t_min ≥ 1e-3`` so
    asking it at ``t = 0`` exactly is out-of-distribution.

    Args:
        score_fn: ``(θ, t) → ℝ^D`` callable.
        theta: current SIREN-weight vector at this FWI iter.
        t_eps: noise level for the score evaluation.
        weight: Phase 3 §4 ``λ``.

    Returns:
        ``−λ · s_φ(θ, t_eps)`` — same shape as ``theta``.
    """
    if weight == 0.0:
        return jnp.zeros_like(theta)
    return -weight * score_fn(theta, jnp.asarray(t_eps))


def compose_siren_grad_with_score_prior(
    grads,
    field,
    score_fn,
    *,
    weight: float,
    t_eps: float = 0.01,
):
    """Add a score-prior regulariser term to a SIREN gradient pytree.

    The SIREN gradient pipeline operates on Equinox pytrees, but the
    score net consumes a flat θ vector. This helper:

      1. Flattens ``field``'s inexact-array leaves to a 1-D θ vector.
      2. Computes ``-λ · s_φ(θ, t_eps)`` via :func:`score_prior_grad_term`.
      3. Unravels that flat term back into the field-shaped pytree.
      4. Adds it to ``grads`` leaf-wise.

    Args:
        grads: gradient pytree (same structure as ``field``'s inexact
            leaves) returned by ``eqx.filter_value_and_grad``.
        field: current SIRENField / SIREN module.
        score_fn: ``(θ, t) → ℝ^D`` score callable.
        weight: Phase 3 §4 ``λ``. ``0`` short-circuits.
        t_eps: small noise level at which to evaluate the score.

    Returns:
        New gradient pytree.
    """
    if weight == 0.0:
        return grads

    inexact = eqx.filter(field, eqx.is_inexact_array)
    theta_flat, unravel = jax.flatten_util.ravel_pytree(inexact)
    grad_term_flat = score_prior_grad_term(
        score_fn, theta_flat, t_eps=t_eps, weight=weight,
    )
    grad_term_tree = unravel(grad_term_flat)

    def _add(g, t):
        if g is None or t is None:
            return g
        return g + t

    return jax.tree.map(_add, grads, grad_term_tree)

"""Conditional normalizing flow for amortised posterior estimation (NPE).

Thin wrapper around `flowjax`'s masked-autoregressive flow factory so
that callers can write posterior-over-SIREN-weights training loops
without learning the underlying library's whole API surface.

Design choices:

- Masked autoregressive flow with rational-quadratic spline transformer.
  Standard NPE building block (Papamakarios 2017, Durkan 2019) and the
  one flowjax exposes most cleanly. Good enough for the O(100)–O(10³)
  theta dims we'll start with; for full SIREN-weight theta
  (~5×10⁴ dims) we'll likely move to Phase 3 diffusion.
- Conditional density `q(theta | d)`: the flow is trained to match the
  true posterior by minimising `E_{theta,d} [-log q(theta|d)]` on
  simulated pairs.
- Equinox module so it composes with the rest of the project (jit,
  filter_value_and_grad, pytree serialisation).

Entry points:

  :class:`ConditionalFlow` — the flow itself (log_prob / sample).
  :func:`train_npe` — Adam training loop with mini-batching.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax


class ConditionalFlow(eqx.Module):
    """Conditional normalizing flow `q(theta | d)` for NPE.

    Attributes are carried as pytree leaves; Equinox partitioning via
    ``eqx.filter_value_and_grad`` is the canonical way to train.
    """

    flow: object
    theta_dim: int = eqx.field(static=True)
    d_dim: int = eqx.field(static=True)

    def __init__(
        self,
        theta_dim: int,
        d_dim: int,
        key: jax.Array,
        n_transforms: int = 6,
        nn_width: int = 64,
        nn_depth: int = 2,
        knots: int = 8,
        interval: float = 4.0,
    ):
        # Imports deferred so the module itself imports without flowjax
        # (useful for skipping unsupported local platforms).
        from flowjax.flows import masked_autoregressive_flow
        from flowjax.distributions import Normal
        from flowjax.bijections import RationalQuadraticSpline

        self.theta_dim = theta_dim
        self.d_dim = d_dim
        self.flow = masked_autoregressive_flow(
            key=key,
            base_dist=Normal(jnp.zeros(theta_dim)),
            cond_dim=d_dim,
            transformer=RationalQuadraticSpline(knots=knots, interval=interval),
            flow_layers=n_transforms,
            nn_width=nn_width,
            nn_depth=nn_depth,
        )

    def log_prob(self, theta: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
        """Conditional log-density. Scalar in, scalar out (use vmap for batches)."""
        return self.flow.log_prob(theta, condition=d)

    def sample(
        self,
        d: jnp.ndarray,
        key: jax.Array,
        n_samples: int = 1,
    ) -> jnp.ndarray:
        """Sample theta from `q(theta | d)`."""
        return self.flow.sample(key, (n_samples,), condition=d)


def train_npe(
    flow: ConditionalFlow,
    theta: jnp.ndarray,
    d: jnp.ndarray,
    key: jax.Array,
    n_steps: int = 1000,
    learning_rate: float = 1e-3,
    batch_size: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[ConditionalFlow, List[float]]:
    """Train `flow` to match `q(theta | d)` on simulated pairs.

    Standard NPE objective: maximise the log-likelihood of `theta` under
    `q(· | d)` for every simulated (theta, d) pair.

    Args:
        flow: :class:`ConditionalFlow` to train.
        theta: ``(N, theta_dim)`` stacked parameter samples.
        d: ``(N, d_dim)`` stacked observation samples.
        key: JAX PRNG key for minibatch shuffling.
        n_steps: Adam steps.
        learning_rate: Adam lr.
        batch_size: Minibatch size. ``None`` → full-batch (use only at
            small scale; SIREN-weight datasets will need minibatching).
        verbose: Print NLL every 50 steps.

    Returns:
        (trained_flow, loss_history).
    """
    n = theta.shape[0]
    if theta.shape[0] != d.shape[0]:
        raise ValueError(
            f"theta and d must have matching leading dim; got {theta.shape[0]} vs {d.shape[0]}"
        )
    batch_size = n if batch_size is None else min(batch_size, n)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(flow, eqx.is_inexact_array))

    @eqx.filter_jit
    def loss_fn(flow, theta_batch, d_batch):
        log_probs = jax.vmap(flow.log_prob)(theta_batch, d_batch)
        return -jnp.mean(log_probs)

    @eqx.filter_jit
    def step(flow, opt_state, theta_batch, d_batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(flow, theta_batch, d_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        flow = eqx.apply_updates(flow, updates)
        return flow, opt_state, loss

    losses: List[float] = []
    for i in range(n_steps):
        key, subkey = jr.split(key)
        idx = jr.choice(subkey, n, shape=(batch_size,), replace=False)
        flow, opt_state, loss = step(flow, opt_state, theta[idx], d[idx])
        losses.append(float(loss))
        if verbose and (i + 1) % 50 == 0:
            print(f"  step {i+1}/{n_steps}: NLL={float(loss):.4f}")

    return flow, losses

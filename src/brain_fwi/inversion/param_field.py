"""Parameterizations of the sound-speed field for FWI.

Two strategies share a common ``ParameterField`` interface:

  - ``VoxelField``: raw voxel grid with sigmoid reparameterization (the
    classical FWI parameterization used by Guasch et al. and Stride).
  - ``SIRENField``: coordinate-based sinusoidal MLP (Sitzmann et al. 2020,
    NeurIPS) that outputs a scalar per grid point. Naturally smooth, memory-
    light (~10^4 weights vs 10^6 voxels), and a better basis for learned
    priors / SBI / score-based regularization since the posterior lives in
    a low-dimensional weight space instead of a high-dimensional voxel grid.

The field is always mapped to physical sound speed via::

    c(x) = c_min + (c_max - c_min) * sigmoid(raw(x))

so bounds are enforced for both parameterizations.

Both fields are Equinox modules (pytrees). Use ``eqx.filter_value_and_grad``
when differentiating and ``eqx.filter(field, eqx.is_inexact_array)`` when
initializing Optax optimizers.
"""

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax


class ParameterField(eqx.Module):
    """Base class. Subclasses implement ``to_velocity``."""

    def to_velocity(self, c_min: float, c_max: float) -> jnp.ndarray:
        raise NotImplementedError


class VoxelField(ParameterField):
    """Voxel-grid parameterization (classical FWI)."""

    params: jnp.ndarray

    def to_velocity(self, c_min: float, c_max: float) -> jnp.ndarray:
        return c_min + (c_max - c_min) * jax.nn.sigmoid(self.params)


class SIREN(eqx.Module):
    """Sinusoidal representation network (Sitzmann et al. 2020).

    f(x) = W_L sin(omega_0 * W_{L-1} sin(... sin(omega_0 * W_1 x + b_1) ...))

    The initialization scheme preserves activation variance across depth
    so training with ω₀ ≈ 30 converges without vanishing/exploding signals.
    """

    layers: list
    omega_0: float = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_hidden: int,
        out_dim: int,
        omega_0: float,
        key: jax.Array,
    ):
        keys = jr.split(key, n_hidden + 2)
        layers = [_make_siren_linear(in_dim, hidden_dim, keys[0], is_first=True, omega_0=omega_0)]
        for i in range(n_hidden):
            layers.append(
                _make_siren_linear(hidden_dim, hidden_dim, keys[i + 1], is_first=False, omega_0=omega_0)
            )
        layers.append(_make_siren_linear(hidden_dim, out_dim, keys[-1], is_first=False, omega_0=omega_0))
        self.layers = layers
        self.omega_0 = omega_0

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = jnp.sin(self.omega_0 * self.layers[0](x))
        for layer in self.layers[1:-1]:
            h = jnp.sin(self.omega_0 * layer(h))
        return self.layers[-1](h).squeeze()


def _make_siren_linear(
    in_dim: int,
    out_dim: int,
    key: jax.Array,
    is_first: bool,
    omega_0: float,
) -> eqx.nn.Linear:
    """Build an ``eqx.nn.Linear`` with SIREN-style uniform weight init."""
    layer = eqx.nn.Linear(in_dim, out_dim, key=key)
    if is_first:
        bound = 1.0 / in_dim
    else:
        bound = jnp.sqrt(6.0 / in_dim) / omega_0
    wkey, bkey = jr.split(key)
    new_w = jr.uniform(wkey, (out_dim, in_dim), minval=-bound, maxval=bound)
    new_b = jr.uniform(bkey, (out_dim,), minval=-bound, maxval=bound)
    layer = eqx.tree_at(lambda m: m.weight, layer, new_w)
    layer = eqx.tree_at(lambda m: m.bias, layer, new_b)
    return layer


class SIRENField(ParameterField):
    """SIREN-parameterized sound-speed field.

    Coordinates are rebuilt from ``grid_shape`` on each call (constant-folded
    under ``jit``) so only the SIREN weights appear as differentiable leaves.
    Output raw field is reshaped to ``grid_shape`` before the sigmoid bound.
    """

    siren: SIREN
    grid_shape: Tuple[int, ...] = eqx.field(static=True)

    def to_velocity(self, c_min: float, c_max: float) -> jnp.ndarray:
        coords = _make_coords(self.grid_shape)
        raw = jax.vmap(self.siren)(coords).reshape(self.grid_shape)
        return c_min + (c_max - c_min) * jax.nn.sigmoid(raw)


def _make_coords(grid_shape: Tuple[int, ...]) -> jnp.ndarray:
    """Normalized grid coordinates in [-1, 1]^ndim, shape ``(prod(grid), ndim)``."""
    axes = [jnp.linspace(-1.0, 1.0, n) for n in grid_shape]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack([g.ravel() for g in mesh], axis=-1)


def init_siren_from_velocity(
    initial_velocity: jnp.ndarray,
    c_min: float,
    c_max: float,
    hidden_dim: int = 128,
    n_hidden: int = 3,
    omega_0: float = 30.0,
    pretrain_steps: int = 1000,
    learning_rate: float = 1e-4,
    key: jax.Array = None,
    verbose: bool = True,
) -> SIRENField:
    """Fit a SIREN to reproduce ``initial_velocity`` in the pre-sigmoid space.

    Works in the unconstrained (logit) space so the sigmoid bound is never
    the bottleneck during regression. After pretraining, ``field.to_velocity``
    reproduces ``initial_velocity`` up to MLP capacity.
    """
    if key is None:
        key = jr.PRNGKey(0)

    grid_shape = tuple(initial_velocity.shape)
    ndim = len(grid_shape)

    v_clipped = jnp.clip(initial_velocity, c_min + 1.0, c_max - 1.0)
    normalized = (v_clipped - c_min) / (c_max - c_min)
    target_raw = jnp.log(normalized / (1.0 - normalized)).ravel()

    key, init_key = jr.split(key)
    siren = SIREN(
        in_dim=ndim,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        out_dim=1,
        omega_0=omega_0,
        key=init_key,
    )
    coords = _make_coords(grid_shape)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(siren, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(siren, opt_state):
        def loss_fn(s):
            pred = jax.vmap(s)(coords)
            return jnp.mean((pred - target_raw) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(siren)
        updates, opt_state = optimizer.update(grads, opt_state)
        siren = eqx.apply_updates(siren, updates)
        return siren, opt_state, loss

    for i in range(pretrain_steps):
        siren, opt_state, loss = step(siren, opt_state)
        if verbose and (i + 1) % max(1, pretrain_steps // 5) == 0:
            print(f"    SIREN pretrain {i+1}/{pretrain_steps}: mse={float(loss):.4f}")

    return SIRENField(siren=siren, grid_shape=grid_shape)


def init_voxel_from_velocity(
    initial_velocity: jnp.ndarray,
    c_min: float,
    c_max: float,
) -> VoxelField:
    """Build a ``VoxelField`` whose ``to_velocity`` reproduces ``initial_velocity``."""
    v_clipped = jnp.clip(initial_velocity, c_min + 1.0, c_max - 1.0)
    normalized = (v_clipped - c_min) / (c_max - c_min)
    params = jnp.log(normalized / (1.0 - normalized))
    return VoxelField(params=params)

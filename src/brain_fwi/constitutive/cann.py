"""CANN for frequency-dependent acoustic attenuation α(ω)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class AttenuationCANN(eqx.Module):
    log_w: jax.Array
    eta: jax.Array
    n_basis: int = eqx.field(static=True)
    omega_scale: float = eqx.field(static=True)
    alpha_scale: float = eqx.field(static=True)

    def __init__(
        self,
        n_basis: int,
        *,
        key: jax.Array,
        omega_scale: float = 1.0,
        alpha_scale: float = 1.0,
    ):
        kw, ke = jr.split(key)
        self.n_basis = int(n_basis)
        self.omega_scale = float(omega_scale)
        self.alpha_scale = float(alpha_scale)
        self.log_w = jr.normal(kw, (n_basis,))
        self.eta = jr.normal(ke, (n_basis,))

    def __call__(self, omega: jax.Array) -> jax.Array:
        w = jax.nn.softplus(self.log_w)
        y = 1.0 + jax.nn.sigmoid(self.eta)
        x = jnp.abs(omega) / self.omega_scale
        omega_pow = x[..., None] ** y
        return self.alpha_scale * jnp.sum(w * omega_pow, axis=-1)

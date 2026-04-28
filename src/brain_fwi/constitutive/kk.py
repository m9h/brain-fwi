"""Kramers–Kronig dispersion: causal link between α(ω) and c(ω)."""

from __future__ import annotations

import jax.numpy as jnp


def kramers_kronig_dispersion(
    alpha_omega: jnp.ndarray,
    omega: jnp.ndarray,
    *,
    omega_ref: float,
    c_ref: float,
) -> jnp.ndarray:
    """Map ``α(ω) → c(ω)`` via the K-K relation.

    Args:
        alpha_omega: ``(N,)`` attenuation samples on the ``omega`` grid.
        omega: ``(N,)`` strictly-positive monotonic angular frequencies.
        omega_ref: calibration angular frequency at which ``c = c_ref``.
        c_ref: phase velocity at ``omega_ref`` (m/s).

    Returns:
        ``c(omega)`` array, same shape as ``omega``.
    """
    omega = jnp.asarray(omega)
    alpha_omega = jnp.asarray(alpha_omega)

    domega = jnp.gradient(omega)
    eps = 1e-9
    omega_p = omega[None, :]
    omega_q = omega[:, None]
    denom = omega_p ** 2 - omega_q ** 2
    integrand = jnp.where(jnp.abs(denom) > eps, alpha_omega[None, :] / denom, 0.0)
    delta_inv_c = (2.0 / jnp.pi) * jnp.sum(integrand * domega[None, :], axis=1)

    delta_ref = jnp.interp(jnp.asarray(omega_ref), omega, delta_inv_c)
    inv_c = (1.0 / c_ref) + delta_inv_c - delta_ref
    return 1.0 / inv_c


def kk_consistency_loss(
    alpha_omega: jnp.ndarray,
    c_omega: jnp.ndarray,
    omega: jnp.ndarray,
    *,
    omega_ref: float,
    c_ref: float,
) -> jnp.ndarray:
    """Squared error between ``c_omega`` and the K-K image of ``alpha_omega``.

    Use as a soft causality penalty when training a network that
    predicts both ``α`` and ``c`` independently.
    """
    c_kk = kramers_kronig_dispersion(
        alpha_omega, omega, omega_ref=omega_ref, c_ref=c_ref,
    )
    return jnp.mean((c_omega - c_kk) ** 2)

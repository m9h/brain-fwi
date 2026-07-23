"""Multi-relaxation (generalized-Maxwell / Prony) attenuation spectrum.

Route A of ``docs/design/cann_forward_scoping.md``: model alpha(omega) as a sum
of relaxation mechanisms, each a local Debye loss with a non-negative modulus and
a relaxation frequency. This is the time-domain-implementable constitutive form
(each mechanism is a local ODE in a solver, no fractional Laplacian) and the
frequency-domain image of the iCANN Prony spectrum (Holthusen et al. 2024).

A fixed-relaxation-time, moduli-only fit only reaches ~10-15% for a y!=1 tissue
power law; matching y=1.3 to <5% requires optimising the relaxation *times* too
(the standard viscoelastic step, Emmerich & Korn 1987; Blanch et al. 1995). This
module does the nonlinear fit of moduli AND times.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import optax


def relaxation_alpha(
    omega: jnp.ndarray,
    moduli: jnp.ndarray,
    relax_freqs_hz: jnp.ndarray,
) -> jnp.ndarray:
    """alpha(omega) from a relaxation spectrum.

    ``alpha(omega) = sum_m g_m * (omega*tau_m) / (1 + (omega*tau_m)^2)`` with
    ``tau_m = 1 / (2*pi*fr_m)``. Non-negative moduli ``g_m`` keep alpha >= 0
    (dissipation), and each term is DC-vanishing. Each mechanism has an analytic
    Kramers-Kronig dispersion partner, so causality is structural.
    """
    w = jnp.abs(jnp.asarray(omega))[..., None]
    tau = 1.0 / (2.0 * jnp.pi * jnp.asarray(relax_freqs_hz))
    wt = w * tau
    return jnp.sum(jnp.asarray(moduli) * wt / (1.0 + wt ** 2), axis=-1)


@dataclass
class RelaxationFit:
    """Result of :func:`fit_relaxation_spectrum`."""
    moduli: np.ndarray            # non-negative modulus per mechanism
    relax_freqs_hz: np.ndarray    # relaxation frequency per mechanism (Hz)
    rel_rmse: float               # ||fit - target|| / ||target|| over the band
    n_mech: int


def fit_relaxation_spectrum(
    alpha_target: jnp.ndarray,
    omega: jnp.ndarray,
    n_mech: int = 8,
    n_steps: int = 5000,
    lr: float = 3e-2,
    seed: int = 0,
) -> RelaxationFit:
    """Fit a relaxation spectrum to ``alpha_target(omega)``, optimising both the
    non-negative moduli and the relaxation frequencies (nonlinear, Adam).

    Relaxation frequencies are **bounded** log-uniformly to the band ± a decade
    (a sigmoid transform) so they cannot collapse out of band into the pure
    constant-Q (linear) regime — the failure mode that caps a naive fit at
    ~10-15% for y!=1. The target is normalised for conditioning and the moduli
    rescaled back.
    """
    omega = jnp.asarray(omega)
    target = jnp.asarray(alpha_target)
    scale = float(jnp.max(jnp.abs(target)))
    scale = scale if scale > 0 else 1.0
    t_norm = target / scale

    a = jnp.abs(omega)
    fr_lo = float(jnp.min(a[a > 0])) / (2 * np.pi) / 10.0   # band lo, minus a decade
    fr_hi = float(jnp.max(a)) / (2 * np.pi) * 10.0          # band hi, plus a decade
    log_ratio = np.log(fr_hi / fr_lo)

    # freq = fr_lo * (fr_hi/fr_lo)^sigmoid(raw_fr)  -> bounded to [fr_lo, fr_hi].
    # init raw_fr so mechanisms spread log-uniformly across the range.
    u0 = np.linspace(0.05, 0.95, n_mech)
    raw_fr0 = np.log(u0 / (1.0 - u0))                       # inverse sigmoid
    params = {
        "raw_g": jnp.full((n_mech,), -1.0),
        "raw_fr": jnp.asarray(raw_fr0, dtype=jnp.float32),
    }

    def freqs_of(p):
        return fr_lo * jnp.exp(log_ratio * jax.nn.sigmoid(p["raw_fr"]))

    def predict(p, w):
        return relaxation_alpha(w, jax.nn.softplus(p["raw_g"]), freqs_of(p))

    def loss_fn(p):
        return jnp.mean((predict(p, omega) - t_norm) ** 2)

    opt = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def step(p, s):
        g = jax.grad(loss_fn)(p)
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s

    for _ in range(n_steps):
        params, state = step(params, state)

    moduli = np.asarray(jax.nn.softplus(params["raw_g"])) * scale
    relax_freqs = np.asarray(freqs_of(params))
    pred = np.asarray(relaxation_alpha(omega, jnp.asarray(moduli), jnp.asarray(relax_freqs)))
    tgt = np.asarray(target)
    rel = float(np.sqrt(np.mean((pred - tgt) ** 2)) / (np.sqrt(np.mean(tgt ** 2)) + 1e-30))
    return RelaxationFit(moduli=moduli, relax_freqs_hz=relax_freqs, rel_rmse=rel, n_mech=n_mech)

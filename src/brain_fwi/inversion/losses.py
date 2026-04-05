"""Loss functions for Full Waveform Inversion.

Provides:
  - l2_loss: Standard L2 waveform difference (Stride default)
  - envelope_loss: Hilbert envelope matching (j-Wave FWI notebook)
  - multiscale_loss: Frequency-weighted combination

The envelope loss is more robust to cycle-skipping artifacts at low
frequencies, which is critical for transcranial imaging where skull
heterogeneity creates large phase errors.
"""

import jax
import jax.numpy as jnp


def l2_loss(predicted: jnp.ndarray, observed: jnp.ndarray) -> jnp.ndarray:
    """L2 waveform misfit.

    f = 0.5 * sum((predicted - observed)^2) / sum(observed^2)

    Normalized by observed energy for scale-invariance across frequencies.

    Args:
        predicted: (n_timesteps, n_sensors) or (n_sensors,) simulated data.
        observed: Same shape as predicted.

    Returns:
        Scalar loss value.
    """
    residual = predicted - observed
    energy = jnp.sum(observed ** 2) + 1e-30
    return 0.5 * jnp.sum(residual ** 2) / energy


def envelope_loss(predicted: jnp.ndarray, observed: jnp.ndarray) -> jnp.ndarray:
    """Hilbert envelope misfit (robust to cycle-skipping).

    Compares the amplitude envelopes rather than raw waveforms.
    Much more convex basin of attraction — better convergence from
    poor initial models (critical for brain FWI through skull).

    f = 0.5 * sum((|H(pred)| - |H(obs)|)^2) / sum(|H(obs)|^2)

    Args:
        predicted: (n_timesteps, n_sensors) simulated data.
        observed: Same shape.

    Returns:
        Scalar loss value.
    """
    env_pred = _hilbert_envelope(predicted)
    env_obs = _hilbert_envelope(observed)

    residual = env_pred - env_obs
    energy = jnp.sum(env_obs ** 2) + 1e-30
    return 0.5 * jnp.sum(residual ** 2) / energy


def multiscale_loss(
    predicted: jnp.ndarray,
    observed: jnp.ndarray,
    envelope_weight: float = 0.5,
) -> jnp.ndarray:
    """Combined L2 + envelope loss.

    Balances waveform fidelity (L2) with robustness (envelope).

    Args:
        predicted: (n_timesteps, n_sensors).
        observed: Same shape.
        envelope_weight: Weight for envelope term (0 = pure L2, 1 = pure envelope).

    Returns:
        Scalar loss value.
    """
    l2 = l2_loss(predicted, observed)
    env = envelope_loss(predicted, observed)
    return (1.0 - envelope_weight) * l2 + envelope_weight * env


def _hilbert_envelope(x: jnp.ndarray) -> jnp.ndarray:
    """Compute Hilbert envelope along time axis (axis=0).

    Uses the FFT-based analytic signal computation.
    """
    n = x.shape[0]
    X = jnp.fft.fft(x, axis=0)

    # Build the Hilbert filter
    h = jnp.zeros(n)
    h = h.at[0].set(1.0)
    if n % 2 == 0:
        h = h.at[n // 2].set(1.0)
        h = h.at[1:n // 2].set(2.0)
    else:
        h = h.at[1:(n + 1) // 2].set(2.0)

    # Apply filter and inverse FFT
    analytic = jnp.fft.ifft(X * h[:, jnp.newaxis], axis=0)
    return jnp.abs(analytic)

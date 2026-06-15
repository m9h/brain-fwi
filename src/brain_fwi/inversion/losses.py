"""Loss functions for Full Waveform Inversion.

Provides:
  - l2_loss: Standard L2 waveform difference (Stride default)
  - envelope_loss: Hilbert envelope matching (j-Wave FWI notebook)
  - multiscale_loss: Frequency-weighted combination
  - awi_loss: Adaptive Waveform Inversion (Warner & Guasch 2016)

The envelope loss is more robust to cycle-skipping artifacts at low
frequencies, which is critical for transcranial imaging where skull
heterogeneity creates large phase errors. AWI goes further: it has a
much wider basin of attraction (it is the objective behind the
successful in-silico brain FWI of Guasch et al. 2020), and is the
missing ingredient behind the plain-L2 skull cycle-skip.
"""

import jax
import jax.numpy as jnp


def l2_loss(predicted: jnp.ndarray, observed: jnp.ndarray) -> jnp.ndarray:
    """L2 waveform misfit.

    f = 0.5 * sum((predicted - observed)^2)

    Unnormalized L2 — avoids instability when bandpass filtering
    reduces observed energy unevenly between predicted and observed.

    Args:
        predicted: (n_timesteps, n_sensors) or (n_sensors,) simulated data.
        observed: Same shape as predicted.

    Returns:
        Scalar loss value.
    """
    residual = predicted - observed
    return 0.5 * jnp.mean(residual ** 2)


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


def awi_loss(
    predicted: jnp.ndarray,
    observed: jnp.ndarray,
    eps: float = 1e-2,
) -> jnp.ndarray:
    """Adaptive Waveform Inversion misfit (Warner & Guasch 2016).

    Rather than the raw residual, AWI finds — per trace — the Wiener
    matching filter ``w`` that maps the predicted trace onto the observed
    trace, and penalises the filter for energy away from zero lag:

        J = 0.5 * mean_traces  Σ_τ (τ·w(τ))²  /  Σ_τ w(τ)²

    When predicted matches observed (up to scaling) the optimal filter is
    a spike at zero lag and J → 0. A cycle-skipped misalignment of k
    samples drives w to a spike at lag k, so J ≈ 0.5·k² — a single, wide
    basin in the time shift, with NO spurious local minima at whole-cycle
    offsets (where plain L2 cycle-skips). This is what lets FWI climb the
    high-contrast skull from a poor starting model.

    The matching filter is computed by frequency-domain (regularised
    Wiener) deconvolution, so the whole objective is differentiable and
    ``jax.grad`` yields the adjoint source automatically — no hand-derived
    AWI adjoint needed (the j-Wave/JAX advantage).

    Args:
        predicted: (n_timesteps, n_sensors) simulated data.
        observed: Same shape.
        eps: Wiener regularisation, relative to each trace's peak power
            spectral density. Larger = smoother filter, more stable on
            low-energy or noisy traces.

    Returns:
        Scalar AWI loss (minimised at alignment).
    """
    nt = predicted.shape[0]
    P = jnp.fft.fft(predicted, axis=0)
    D = jnp.fft.fft(observed, axis=0)
    power = jnp.abs(P) ** 2
    reg = eps * jnp.max(power, axis=0, keepdims=True)
    # Wiener matching filter: w ⋆ predicted ≈ observed.
    W = jnp.conj(P) * D / (power + reg + 1e-30)
    w = jnp.real(jnp.fft.ifft(W, axis=0))            # (nt, n_sensors)
    # Signed lags in samples, zero-centred in FFT order (0,1,..,-2,-1).
    tau = (jnp.fft.fftfreq(nt) * nt)[:, jnp.newaxis]
    num = jnp.sum((tau * w) ** 2, axis=0)            # penalised filter energy
    den = jnp.sum(w ** 2, axis=0) + 1e-30            # total filter energy
    return 0.5 * jnp.mean(num / den)


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

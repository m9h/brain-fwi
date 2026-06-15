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


def _awi_matching_filter(predicted, observed, filter_half_len=64, eps=1e-2):
    """Limited-length least-squares (Wiener) matching filter for ONE trace.

    Solves for ``w`` (supported on lags [-L, L], L = filter_half_len) that
    minimises ``|| conv(w, predicted) - observed ||²`` via the normal
    equations (AᵀA + εI) w = Aᵀ observed, where A is the banded convolution
    matrix of ``predicted``. The LIMITED length is essential: a full-length
    filter can deconvolve predicted→observed exactly, which makes the FWI
    velocity gradient ill-conditioned/uninformative (the earlier full-FFT
    variant underperformed plain L2 on skull recovery). A short filter
    forces the misfit to localise as a genuine time lag.

    Returns ``(w, lags)`` with ``lags = -L .. L``. Differentiable end to end
    (the linear solve included), so ``jax.grad`` yields the AWI adjoint.
    """
    nt = predicted.shape[0]
    L = int(min(filter_half_len, nt // 2 - 1))
    M = 2 * L + 1
    t = jnp.arange(nt)[:, None]
    j = jnp.arange(M)[None, :]
    idx = t - j + L                              # conv(w,p)[t] = Σ_j w[j]·p[t-j+L]
    valid = (idx >= 0) & (idx < nt)
    A = jnp.where(valid, predicted[jnp.clip(idx, 0, nt - 1)], 0.0)  # (nt, M)
    AtA = A.T @ A
    Atd = A.T @ observed
    reg = eps * (jnp.trace(AtA) / M + 1e-30)
    w = jnp.linalg.solve(AtA + reg * jnp.eye(M), Atd)
    return w, jnp.arange(-L, L + 1)


def awi_loss(
    predicted: jnp.ndarray,
    observed: jnp.ndarray,
    filter_half_len: int = 64,
    eps: float = 1e-2,
) -> jnp.ndarray:
    """Adaptive Waveform Inversion misfit (Warner & Guasch 2016).

    Per trace, find the LIMITED-length Wiener matching filter ``w`` that maps
    predicted onto observed (see :func:`_awi_matching_filter`), then penalise
    its energy away from zero lag:

        J = 0.5 * mean_traces  Σ_τ (τ·w(τ))²  /  Σ_τ w(τ)²

    A cycle-skipped misalignment of k samples drives w to a spike at lag k,
    so J ≈ 0.5·k² — a single wide basin in the time shift with NO spurious
    local minima at whole-cycle offsets (where plain L2 cycle-skips).
    Differentiable, so ``jax.grad`` yields the adjoint source automatically.

    Args:
        predicted: (n_timesteps, n_sensors) or (n_timesteps,) simulated data.
        observed: Same shape.
        filter_half_len: matching-filter half-length L (lags ±L). Must cover
            the largest expected time shift; kept short for a well-conditioned
            velocity gradient.
        eps: Tikhonov regularisation on the normal equations (relative to the
            mean diagonal), for stability on low-energy/noisy traces.

    Returns:
        Scalar AWI loss (minimised at alignment).
    """
    def per_trace(p, d):
        w, lags = _awi_matching_filter(p, d, filter_half_len, eps)
        tau = lags.astype(w.dtype)
        return jnp.sum((tau * w) ** 2) / (jnp.sum(w ** 2) + 1e-30)

    if predicted.ndim == 1:
        return 0.5 * per_trace(predicted, observed)
    Js = jax.vmap(per_trace, in_axes=(1, 1))(predicted, observed)
    return 0.5 * jnp.mean(Js)


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

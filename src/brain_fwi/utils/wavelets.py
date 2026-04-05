"""Source wavelets for FWI simulation.

Provides Ricker (Mexican hat) wavelet and toneburst generators.
The Ricker wavelet is standard for FWI (used by Stride, j-Wave examples).
"""

import jax.numpy as jnp
import numpy as np


def ricker_wavelet(
    f0: float,
    dt: float,
    n_samples: int,
    delay: float = 0.0,
) -> jnp.ndarray:
    """Generate a Ricker (Mexican hat) wavelet.

    Standard source wavelet for seismic/ultrasound FWI. Zero-phase with
    peak at t=delay, dominant frequency f0.

    Args:
        f0: Central frequency in Hz.
        dt: Time step in seconds.
        n_samples: Number of time samples.
        delay: Time delay of peak in seconds.
            Default 0 = auto-set to 1.5/f0 for causal wavelet.

    Returns:
        (n_samples,) array.
    """
    if delay <= 0.0:
        delay = 1.5 / f0

    t = jnp.arange(n_samples) * dt
    tau = t - delay
    pi_f_tau = jnp.pi * f0 * tau
    return (1.0 - 2.0 * pi_f_tau ** 2) * jnp.exp(-pi_f_tau ** 2)


def toneburst(
    f0: float,
    dt: float,
    n_cycles: int = 5,
    delay: float = 0.0,
    n_samples: int = 0,
) -> jnp.ndarray:
    """Generate a windowed sinusoidal toneburst.

    Args:
        f0: Frequency in Hz.
        dt: Time step in seconds.
        n_cycles: Number of cycles in the burst.
        delay: Time delay in seconds.
        n_samples: Total number of samples. 0 = auto-size.

    Returns:
        1D array of the toneburst signal.
    """
    duration = n_cycles / f0
    if n_samples <= 0:
        total_time = delay + duration + 2.0 / f0
        n_samples = int(total_time / dt) + 1

    t = jnp.arange(n_samples) * dt
    t_shifted = t - delay

    # Hann-windowed sinusoid
    signal = jnp.sin(2.0 * jnp.pi * f0 * t_shifted)
    window = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * t_shifted / duration))
    active = (t_shifted >= 0) & (t_shifted <= duration)

    return jnp.where(active, signal * window, 0.0)

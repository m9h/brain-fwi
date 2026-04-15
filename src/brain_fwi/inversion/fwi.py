"""Full Waveform Inversion engine.

Implements gradient-based FWI using JAX automatic differentiation through
j-Wave's pseudospectral solver. Follows the design patterns of:

  - j-Wave FWI notebook: Autodiff gradients, Hilbert envelope loss,
    reparameterized velocity, gradient smoothing
  - Stride: Multi-frequency banding, stochastic source selection,
    model bounds, gradient processing pipeline

Key innovation: the entire gradient computation is handled by JAX autodiff
(jax.grad through simulate_wave_propagation), rather than hand-coded
adjoint operators as in Stride/Devito. This enables:
  1. Exact gradients with zero implementation effort
  2. Higher-order optimization (L-BFGS via Optax)
  3. Easy integration with neural networks (learned regularization)
  4. Direct use of sbi4dwi's SBI pipeline for posterior estimation

Architecture:
  FWIConfig → run_fwi() → FWIResult
  The velocity model is reparameterized as:
    c(x) = c_min + (c_max - c_min) * sigmoid(params(x))
  This enforces physical bounds and improves optimization landscape.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..simulation.forward import (
    build_domain,
    build_medium,
    build_time_axis,
    simulate_shot_sensors,
    _build_source_signal,
)
from .losses import l2_loss, envelope_loss, multiscale_loss


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FWIConfig:
    """Configuration for Full Waveform Inversion.

    Attributes:
        freq_bands: List of (f_min, f_max) frequency bands in Hz.
            FWI proceeds from low to high frequency (multi-scale).
            Stride pattern: start coarse for convexity, refine.
        n_iters_per_band: Iterations per frequency band.
        shots_per_iter: Number of sources per iteration (stochastic).
            Stride default: 16. j-Wave FWI: 1.
        learning_rate: Adam learning rate.
        c_min: Minimum sound speed (m/s). Used for reparameterization.
        c_max: Maximum sound speed (m/s).
        pml_size: PML absorbing boundary thickness (grid points).
        cfl: CFL stability number.
        gradient_smooth_sigma: Gaussian smoothing sigma for gradients
            (in grid points). 0 = no smoothing.
        loss_fn: Loss function name ('l2', 'envelope', 'multiscale').
        envelope_weight: Weight for envelope term in multiscale loss.
        mask: Optional binary mask for inversion region.
            Gradients outside mask are zeroed (e.g., mask out PML, water).
        verbose: Print iteration progress.
    """
    freq_bands: List[Tuple[float, float]] = field(default_factory=lambda: [
        (50e3, 100e3),
        (100e3, 200e3),
        (200e3, 300e3),
    ])
    n_iters_per_band: int = 30
    shots_per_iter: int = 4
    learning_rate: float = 5.0
    c_min: float = 1400.0
    c_max: float = 3200.0
    pml_size: int = 20
    cfl: float = 0.3
    gradient_smooth_sigma: float = 3.0
    loss_fn: str = "multiscale"
    envelope_weight: float = 0.5
    mask: Optional[jnp.ndarray] = None
    skip_bandpass: bool = False
    checkpoint_dir: Optional[str] = None  # Save/resume state after each band
    precondition: bool = True  # Pseudo-Hessian source illumination compensation
    max_step_m_per_s: float = 50.0  # Max velocity change per iteration (m/s)
    verbose: bool = True


@dataclass
class FWIResult:
    """Result of FWI inversion.

    Attributes:
        velocity: Final reconstructed sound speed (m/s).
        velocity_history: List of velocity snapshots (one per band).
        loss_history: Loss values per iteration.
        params: Raw optimization parameters (before reparameterization).
    """
    velocity: jnp.ndarray
    velocity_history: List[jnp.ndarray]
    loss_history: List[float]
    params: jnp.ndarray


# ---------------------------------------------------------------------------
# Reparameterization
# ---------------------------------------------------------------------------

def _params_to_velocity(
    params: jnp.ndarray,
    c_min: float,
    c_max: float,
) -> jnp.ndarray:
    """Convert unconstrained parameters to bounded sound speed.

    c(x) = c_min + (c_max - c_min) * sigmoid(params(x))
    """
    return c_min + (c_max - c_min) * jax.nn.sigmoid(params)


def _velocity_to_params(
    velocity: jnp.ndarray,
    c_min: float,
    c_max: float,
) -> jnp.ndarray:
    """Convert sound speed to unconstrained parameters (inverse sigmoid)."""
    # Clip to avoid log(0) or log(inf)
    v_clipped = jnp.clip(velocity, c_min + 1.0, c_max - 1.0)
    normalized = (v_clipped - c_min) / (c_max - c_min)
    # Inverse sigmoid = logit
    return jnp.log(normalized / (1.0 - normalized))


# ---------------------------------------------------------------------------
# Disk checkpointing (resume after preemption)
# ---------------------------------------------------------------------------

def _save_checkpoint(path: Path, band_idx: int, params, opt_state,
                     loss_history, velocity_history):
    """Save FWI state after completing a frequency band."""
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        f.attrs["completed_bands"] = band_idx + 1
        f.create_dataset("params", data=np.array(params))
        f.create_dataset("loss_history", data=np.array(loss_history))
        for i, v in enumerate(velocity_history):
            f.create_dataset(f"velocity_band_{i}", data=np.array(v))
        # Save Adam state (mu, nu, count) as flat arrays
        mu, nu = opt_state[0].mu, opt_state[0].nu
        f.create_dataset("opt_mu", data=np.array(mu))
        f.create_dataset("opt_nu", data=np.array(nu))
        f.attrs["opt_count"] = int(opt_state[0].count)


def _load_checkpoint(path: Path, learning_rate: float):
    """Load FWI state from a previous run. Returns None if no checkpoint."""
    import h5py
    if not path.exists():
        return None
    with h5py.File(str(path), "r") as f:
        completed_bands = int(f.attrs["completed_bands"])
        params = jnp.array(f["params"][:])
        loss_history = list(f["loss_history"][:])
        velocity_history = [jnp.array(f[f"velocity_band_{i}"][:])
                           for i in range(completed_bands)]
        mu = jnp.array(f["opt_mu"][:])
        nu = jnp.array(f["opt_nu"][:])
        count = jnp.array(int(f.attrs["opt_count"]))
    # Reconstruct Adam state
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    opt_state = (optax.ScaleByAdamState(count=count, mu=mu, nu=nu),
                 opt_state[1])
    return {
        "completed_bands": completed_bands,
        "params": params,
        "opt_state": opt_state,
        "loss_history": loss_history,
        "velocity_history": velocity_history,
    }


# ---------------------------------------------------------------------------
# Gradient smoothing
# ---------------------------------------------------------------------------

def _smooth_gradient(
    grad: jnp.ndarray,
    sigma: float,
) -> jnp.ndarray:
    """Apply Gaussian smoothing to gradient field.

    Prevents high-frequency artifacts in the model update.
    Standard practice in FWI (both Stride and j-Wave examples).

    Args:
        grad: Gradient array (same shape as model).
        sigma: Smoothing sigma in grid points. 0 = no smoothing.

    Returns:
        Smoothed gradient.
    """
    if sigma <= 0:
        return grad

    ndim = grad.ndim
    # Build 1D Gaussian kernel
    radius = int(3 * sigma)
    x = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    kernel_1d = jnp.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / jnp.sum(kernel_1d)

    # Apply separable convolution
    result = grad
    for axis in range(ndim):
        # Reshape kernel for this axis
        shape = [1] * ndim
        shape[axis] = len(kernel_1d)
        k = kernel_1d.reshape(shape)

        # Pad and convolve
        pad_widths = [(0, 0)] * ndim
        pad_widths[axis] = (radius, radius)
        padded = jnp.pad(result, pad_widths, mode="edge")

        # Use lax.conv for 1D convolution along axis
        # Simpler: just use jnp.convolve via vmap
        result = _convolve_along_axis(padded, kernel_1d, axis, grad.shape[axis])

    return result


def _convolve_along_axis(
    padded: jnp.ndarray,
    kernel: jnp.ndarray,
    axis: int,
    output_size: int,
) -> jnp.ndarray:
    """1D convolution along a specific axis of an N-D array."""
    k_len = len(kernel)

    # Move target axis to last position for easier indexing
    moved = jnp.moveaxis(padded, axis, -1)
    out_shape = moved.shape[:-1] + (output_size,)

    # Sliding window convolution
    def conv_1d(x):
        # x is 1D, padded
        return jnp.array([
            jnp.sum(x[i:i + k_len] * kernel)
            for i in range(output_size)
        ])

    # Flatten all non-target dims, apply conv, reshape
    flat = moved.reshape(-1, moved.shape[-1])

    # Vectorized version using jnp.convolve
    def single_conv(row):
        return jnp.convolve(row, kernel, mode="valid")[:output_size]

    result_flat = jax.vmap(single_conv)(flat)
    result = result_flat.reshape(out_shape)

    return jnp.moveaxis(result, -1, axis)


# ---------------------------------------------------------------------------
# Core FWI loop
# ---------------------------------------------------------------------------

def _get_loss_fn(name: str, envelope_weight: float) -> Callable:
    """Get loss function by name."""
    if name == "l2":
        return l2_loss
    elif name == "envelope":
        return envelope_loss
    elif name == "multiscale":
        return lambda p, o: multiscale_loss(p, o, envelope_weight)
    else:
        raise ValueError(f"Unknown loss: {name!r}. Use 'l2', 'envelope', or 'multiscale'.")


def _bandpass_signal(signal: jnp.ndarray, dt: float, fmin: float, fmax: float) -> jnp.ndarray:
    """Apply a bandpass filter to a source signal via FFT.

    Uses smooth cosine tapers at band edges to avoid Gibbs ringing.
    """
    n = signal.shape[0]
    freqs = jnp.fft.fftfreq(n, d=dt)
    S = jnp.fft.fft(signal)

    f_abs = jnp.abs(freqs)
    taper_width = (fmax - fmin) * 0.2

    # Low-frequency taper: 0 below (fmin - taper), rises to 1 at fmin
    low_taper = 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.clip(
        (fmin - f_abs) / (taper_width + 1e-30), 0.0, 1.0)))

    # High-frequency taper: 1 at fmax, drops to 0 above (fmax + taper)
    high_taper = 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.clip(
        (f_abs - fmax) / (taper_width + 1e-30), 0.0, 1.0)))

    # Product forms a smooth bandpass — no hard mask needed
    bandpass = low_taper * high_taper

    return jnp.real(jnp.fft.ifft(S * bandpass))


def run_fwi(
    observed_data: jnp.ndarray,
    initial_velocity: jnp.ndarray,
    density: jnp.ndarray,
    dx: float,
    src_positions_grid: list,
    sensor_positions_grid: Tuple,
    source_signal: jnp.ndarray,
    dt: float,
    t_end: float,
    config: Optional[FWIConfig] = None,
    key: Optional[jax.Array] = None,
) -> FWIResult:
    """Run Full Waveform Inversion.

    This is the main entry point. It iteratively updates a sound speed
    model to minimize the misfit between simulated and observed data.

    The algorithm:
    1. For each frequency band (low → high):
       2. For each iteration:
          a. Select random subset of sources
          b. For each source: simulate forward, record at sensors
          c. Compute loss (bandpass-filtered data)
          d. Backprop through j-Wave → gradient w.r.t. velocity
          e. Smooth gradient, apply mask
          f. Adam optimizer update
       3. Save velocity snapshot

    Args:
        observed_data: (n_sources, n_timesteps, n_sensors) ground truth.
        initial_velocity: (*spatial_dims) starting velocity model (m/s).
        density: (*spatial_dims) density model (held fixed during FWI).
        dx: Grid spacing (m).
        src_positions_grid: List of (ix, iy[, iz]) source positions.
        sensor_positions_grid: Tuple of receiver index arrays.
        source_signal: (n_samples,) base source wavelet.
        dt: Time step (s).
        t_end: Simulation end time (s).
        config: FWI configuration. None = defaults.
        key: JAX PRNG key for stochastic source selection.

    Returns:
        FWIResult with reconstructed velocity and diagnostics.
    """
    if config is None:
        config = FWIConfig()
    if key is None:
        key = jr.PRNGKey(0)

    grid_shape = initial_velocity.shape
    n_sources = len(src_positions_grid)
    loss_fn = _get_loss_fn(config.loss_fn, config.envelope_weight)

    # Initialize reparameterized parameters
    params = _velocity_to_params(initial_velocity, config.c_min, config.c_max)

    # Adam optimizer
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    loss_history = []
    velocity_history = []
    start_band = 0

    # Resume from checkpoint if available
    if config.checkpoint_dir:
        ckpt_path = Path(config.checkpoint_dir) / "fwi_checkpoint.h5"
        ckpt = _load_checkpoint(ckpt_path, config.learning_rate)
        if ckpt is not None:
            start_band = ckpt["completed_bands"]
            params = ckpt["params"]
            opt_state = ckpt["opt_state"]
            loss_history = ckpt["loss_history"]
            velocity_history = ckpt["velocity_history"]
            if config.verbose:
                print(f"  Resumed from checkpoint: {start_band} bands complete, "
                      f"skipping to band {start_band + 1}")

    if config.verbose:
        print(f"FWI: {len(config.freq_bands)} frequency bands, "
              f"{config.n_iters_per_band} iters/band, "
              f"{config.shots_per_iter} shots/iter")
        print(f"  Grid: {grid_shape}, dx={dx*1e3:.2f} mm")
        print(f"  Velocity bounds: [{config.c_min:.0f}, {config.c_max:.0f}] m/s")
        print(f"  Loss: {config.loss_fn}")

    # Pre-compute time axis OUTSIDE the traced function.
    # TimeAxis.from_medium() calls float() which breaks JAX tracing.
    # Use a reference medium with c_max for CFL stability guarantee.
    ref_domain = build_domain(grid_shape, dx)
    ref_medium = build_medium(ref_domain, config.c_max, 1000.0, pml_size=config.pml_size)
    fixed_time_axis = build_time_axis(ref_medium, cfl=config.cfl, t_end=t_end)

    for band_idx, (fmin, fmax) in enumerate(config.freq_bands):
        if band_idx < start_band:
            continue

        if config.verbose:
            print(f"\n  Band {band_idx+1}/{len(config.freq_bands)}: "
                  f"{fmin/1e3:.0f}-{fmax/1e3:.0f} kHz")

        # Bandpass the source signal and observed data for this frequency band
        if config.skip_bandpass:
            bp_signal = source_signal
            bp_observed = observed_data
        else:
            bp_signal = _bandpass_signal(source_signal, dt, fmin, fmax)
            bp_observed = jax.vmap(
                lambda d: jax.vmap(lambda col: _bandpass_signal(col, dt, fmin, fmax))(d.T).T
            )(observed_data)

        for it in range(config.n_iters_per_band):
            key, subkey = jr.split(key)

            # Select random sources for this iteration
            if config.shots_per_iter >= n_sources:
                shot_indices = jnp.arange(n_sources)
            else:
                shot_indices = jr.choice(
                    subkey, n_sources, shape=(config.shots_per_iter,), replace=False
                )
            shot_indices = np.array(shot_indices)

            # Compute loss and gradient, one shot at a time to save memory.
            # Accumulate gradients across shots (equivalent to batched but
            # uses O(1 shot) memory instead of O(n_shots)).
            # Also accumulate squared gradients for pseudo-Hessian preconditioning.
            total_loss = 0.0
            grad_accum = jnp.zeros_like(params)
            grad_sq_accum = jnp.zeros_like(params)

            # Use checkpointed scan for large grids (>= 128^3)
            use_checkpoint = all(s >= 128 for s in grid_shape)

            for si in shot_indices:
                src_pos = src_positions_grid[int(si)]
                obs = bp_observed[int(si)]

                def single_shot_loss(p, _src_pos=src_pos, _obs=obs):
                    velocity = _params_to_velocity(p, config.c_min, config.c_max)
                    domain = build_domain(grid_shape, dx)
                    medium = build_medium(domain, velocity, density, pml_size=config.pml_size)
                    pred = simulate_shot_sensors(
                        medium, fixed_time_axis, _src_pos, sensor_positions_grid,
                        bp_signal, dt, checkpointed=use_checkpoint,
                    )
                    min_t = min(pred.shape[0], _obs.shape[0])
                    return loss_fn(pred[:min_t], _obs[:min_t])

                shot_loss, shot_grad = jax.value_and_grad(single_shot_loss)(params)
                total_loss = total_loss + float(shot_loss)
                grad_accum = grad_accum + shot_grad
                grad_sq_accum = grad_sq_accum + shot_grad ** 2

            n_shots = len(shot_indices)
            loss_val = total_loss / n_shots
            grad = grad_accum / n_shots
            loss_history.append(loss_val)

            # Source illumination preconditioning (pseudo-Hessian).
            # Compensates for the fact that near-transducer voxels have
            # enormous gradient amplitude while interior brain voxels
            # get tiny gradients. Standard in geophysical FWI (Shin 2001).
            if config.precondition:
                illum = jnp.sqrt(grad_sq_accum / n_shots)
                grad = grad / (illum + 1e-12 * jnp.max(illum))

            # Process gradient
            if config.gradient_smooth_sigma > 0:
                grad = _smooth_gradient(grad, config.gradient_smooth_sigma)

            if config.mask is not None:
                grad = grad * config.mask

            # Adaptive step length: normalise so max velocity change
            # per iteration is bounded (like Stride's max-norm step).
            # Convert max_step from velocity-space to parameter-space.
            if config.max_step_m_per_s > 0:
                grad_max = jnp.max(jnp.abs(grad))
                step_scale = config.max_step_m_per_s / (
                    (config.c_max - config.c_min) * (grad_max + 1e-30)
                )
                grad = grad * step_scale

            # Optimizer update
            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)

            if config.verbose and (it + 1) % 5 == 0:
                vel = _params_to_velocity(params, config.c_min, config.c_max)
                print(f"    Iter {it+1}/{config.n_iters_per_band}: "
                      f"loss={loss_val:.6f}, "
                      f"c=[{float(jnp.min(vel)):.0f}, {float(jnp.max(vel)):.0f}] m/s")

        # Save velocity snapshot at end of band
        velocity_history.append(
            _params_to_velocity(params, config.c_min, config.c_max)
        )

        # Checkpoint to disk for resume after preemption
        if config.checkpoint_dir:
            ckpt_path = Path(config.checkpoint_dir) / "fwi_checkpoint.h5"
            _save_checkpoint(ckpt_path, band_idx, params, opt_state,
                           loss_history, velocity_history)
            if config.verbose:
                print(f"  Checkpoint saved: band {band_idx+1} complete")

    final_velocity = _params_to_velocity(params, config.c_min, config.c_max)

    return FWIResult(
        velocity=final_velocity,
        velocity_history=velocity_history,
        loss_history=loss_history,
        params=params,
    )

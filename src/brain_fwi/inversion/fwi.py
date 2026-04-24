"""Full Waveform Inversion engine.

Implements gradient-based FWI using JAX automatic differentiation through
j-Wave's pseudospectral solver. Two parameterisations of the sound-speed
field are supported behind a shared ``run_fwi`` entry point:

  - ``voxel`` (default): classical voxel grid. Optimises directly in m/s
    using SGD with max-norm gradient normalisation + Gaussian gradient
    smoothing + optional water mask. Bounds enforced by clip after each
    update. This is the production path at 192^3.
  - ``siren``: coordinate-based sinusoidal MLP (Sitzmann 2020). ~10^4
    weights represent the whole field. Optimised with Adam on MLP weights;
    no gradient smoothing/normalisation (architecture handles regularity,
    Adam handles per-parameter scaling). Velocity clipped inside
    ``SIRENField.to_velocity``.

Both paths share forward simulation, frequency banding, per-shot gradient
accumulation, and checkpointing. They differ only in parameterisation,
optimiser choice, and gradient post-processing.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

from ..simulation.forward import (
    build_domain,
    build_medium,
    build_time_axis,
    simulate_shot_sensors,
    _build_source_signal,
)
from .losses import l2_loss, envelope_loss, multiscale_loss
from .param_field import (
    ParameterField,
    SIRENField,
    VoxelField,
    init_siren_from_velocity,
    init_voxel_from_velocity,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FWIConfig:
    """Configuration for Full Waveform Inversion.

    Optimises directly in velocity space (m/s) using steepest descent
    with max-norm gradient normalisation. The learning rate equals the
    maximum velocity change per iteration in m/s.

    Attributes:
        freq_bands: List of (f_min, f_max) frequency bands in Hz.
            FWI proceeds from low to high frequency (multi-scale).
        n_iters_per_band: Iterations per frequency band.
        shots_per_iter: Number of sources per iteration (stochastic).
        learning_rate: Maximum velocity update per iteration (m/s).
            50 m/s is a good default for transcranial FWI.
        c_min: Minimum sound speed bound (m/s). Velocity clipped after update.
        c_max: Maximum sound speed bound (m/s).
        pml_size: PML absorbing boundary thickness (grid points).
        cfl: CFL stability number.
        gradient_smooth_sigma: Gaussian smoothing sigma for gradients
            (in grid points). 0 = no smoothing.
        loss_fn: Loss function name ('l2', 'envelope', 'multiscale').
        envelope_weight: Weight for envelope term in multiscale loss.
        mask: Optional binary mask for inversion region.
            Gradients outside mask are zeroed. Use (labels > 0) to
            exclude water coupling.
        precondition: Apply pseudo-Hessian source illumination compensation.
        verbose: Print iteration progress.
    """
    freq_bands: List[Tuple[float, float]] = field(default_factory=lambda: [
        (50e3, 100e3),
        (100e3, 200e3),
        (200e3, 300e3),
    ])
    n_iters_per_band: int = 30
    shots_per_iter: int = 4
    learning_rate: float = 50.0  # Max velocity update per iteration (m/s), voxel path
    c_min: float = 1400.0
    c_max: float = 3200.0
    pml_size: int = 20
    cfl: float = 0.3
    gradient_smooth_sigma: float = 3.0
    loss_fn: str = "l2"
    envelope_weight: float = 0.5
    mask: Optional[jnp.ndarray] = None
    skip_bandpass: bool = False
    checkpoint_dir: Optional[str] = None  # Save/resume state after each band
    precondition: bool = False  # Pseudo-Hessian source illumination compensation
    verbose: bool = True

    # --- Parameterisation ---
    # "voxel" (default): dense grid, SGD + max-norm grad + smoothing + mask.
    # "siren":           MLP over coordinates, Adam on weights. Ignores
    #                    gradient_smooth_sigma and mask (SIREN is smooth by
    #                    construction; mask doesn't have a clean analogue
    #                    on MLP weights).
    parameterization: Literal["voxel", "siren"] = "voxel"

    # SIREN knobs (unused when parameterization="voxel").
    siren_hidden: int = 128
    siren_layers: int = 3
    siren_omega: float = 30.0
    siren_pretrain_steps: int = 1000
    siren_pretrain_lr: float = 1e-3
    siren_learning_rate: float = 1e-3  # Adam lr on MLP weights during FWI
    siren_seed: int = 0


@dataclass
class FWIResult:
    """Result of FWI inversion.

    Attributes:
        velocity: Final reconstructed sound speed (m/s).
        velocity_history: List of velocity snapshots (one per band).
        loss_history: Loss values per iteration.
        params: Raw optimisation parameters. For the voxel path this is
            ``field.params`` (an m/s array); for SIREN it's a rendered
            voxel copy of the final velocity field for back-compat.
        field: Final ParameterField object. For SIREN, callers can save
            the SIREN weights directly as the compact theta representation
            for the Phase-0 dataset / downstream SBI.
    """
    velocity: jnp.ndarray
    velocity_history: List[jnp.ndarray]
    loss_history: List[float]
    params: jnp.ndarray
    field: Optional[ParameterField] = None


# ---------------------------------------------------------------------------
# Parameterisation dispatch
# ---------------------------------------------------------------------------

def _init_param_field(
    initial_velocity: jnp.ndarray,
    config: "FWIConfig",
) -> ParameterField:
    """Build a ``ParameterField`` for the configured parameterisation.

    Voxel path stores velocity directly. SIREN path runs an Adam
    pretrain to regress the MLP towards the initial velocity (normalised
    to O(1)) before FWI begins.
    """
    if config.parameterization == "voxel":
        return init_voxel_from_velocity(initial_velocity, config.c_min, config.c_max)
    if config.parameterization == "siren":
        return init_siren_from_velocity(
            initial_velocity,
            c_min=config.c_min,
            c_max=config.c_max,
            hidden_dim=config.siren_hidden,
            n_hidden=config.siren_layers,
            omega_0=config.siren_omega,
            pretrain_steps=config.siren_pretrain_steps,
            learning_rate=config.siren_pretrain_lr,
            key=jr.PRNGKey(config.siren_seed),
            verbose=config.verbose,
        )
    raise ValueError(
        f"Unknown parameterization {config.parameterization!r}; "
        f"expected 'voxel' or 'siren'."
    )


# ---------------------------------------------------------------------------
# Disk checkpointing (resume after preemption)
# ---------------------------------------------------------------------------

def _save_checkpoint(path: Path, band_idx: int, params,
                     loss_history, velocity_history,
                     grid_shape=None):
    """Save FWI state after completing a frequency band.

    SGD is memoryless, so opt_state is not persisted — resume re-initialises
    a fresh optimizer on the saved params. ``grid_shape`` is stamped as an
    HDF5 attribute so :func:`_load_checkpoint` can refuse a resume that
    targets a different configuration (as happened in jobs 917 and 919,
    where a stale 192^3 checkpoint was loaded into an unrelated run).
    """
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        f.attrs["completed_bands"] = band_idx + 1
        if grid_shape is not None:
            f.attrs["grid_shape"] = np.asarray(grid_shape, dtype=np.int32)
        f.create_dataset("params", data=np.array(params))
        f.create_dataset("loss_history", data=np.array(loss_history))
        for i, v in enumerate(velocity_history):
            f.create_dataset(f"velocity_band_{i}", data=np.array(v))


def _load_checkpoint(path: Path, expected_grid_shape=None):
    """Load FWI state from a previous run.

    Returns ``None`` if no checkpoint exists. Raises ``ValueError`` if a
    checkpoint is present but its ``params`` shape (or stamped
    ``grid_shape`` attribute) does not match ``expected_grid_shape`` —
    this is what happened in 917/919: a stale 192^3 checkpoint leaked
    into a 96^3 run (917, shape-broadcast crash) and into a fresh 192^3
    MIDA run (919, NaN from wrong-phantom initial velocity).
    """
    import h5py
    if not path.exists():
        return None
    with h5py.File(str(path), "r") as f:
        completed_bands = int(f.attrs["completed_bands"])
        params = jnp.array(f["params"][:])
        loss_history = list(f["loss_history"][:])
        velocity_history = [jnp.array(f[f"velocity_band_{i}"][:])
                           for i in range(completed_bands)]
        stamped_shape = None
        if "grid_shape" in f.attrs:
            stamped_shape = tuple(int(x) for x in f.attrs["grid_shape"])

    if expected_grid_shape is not None:
        expected = tuple(int(x) for x in expected_grid_shape)
        actual = tuple(params.shape)
        if actual != expected or (stamped_shape is not None and stamped_shape != expected):
            raise ValueError(
                f"Checkpoint at {path} has params shape {actual} "
                f"(stamped grid {stamped_shape}) but current run expects "
                f"{expected}. Delete the stale checkpoint or point "
                f"`checkpoint_dir` at a run-specific path."
            )
    return {
        "completed_bands": completed_bands,
        "params": params,
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

    if config.parameterization == "siren":
        return _run_fwi_siren(
            observed_data, initial_velocity, density, dx,
            src_positions_grid, sensor_positions_grid, source_signal,
            dt, t_end, config, key,
        )
    if config.parameterization != "voxel":
        raise ValueError(
            f"Unknown parameterization {config.parameterization!r}; "
            f"expected 'voxel' or 'siren'."
        )

    grid_shape = initial_velocity.shape
    n_sources = len(src_positions_grid)
    loss_fn = _get_loss_fn(config.loss_fn, config.envelope_weight)

    # Optimise directly in velocity space (m/s).
    # No sigmoid reparameterisation — just clip after each update.
    # This gives the learning rate physical meaning: LR=1 means Adam
    # takes ~1 m/s steps (modulated by its moment estimates).
    params = initial_velocity.copy()

    # Steepest descent with gradient normalisation (Stride-style).
    # Combined with max_step_m_per_s, the learning rate directly controls
    # the maximum velocity change per iteration in m/s.
    optimizer = optax.sgd(config.learning_rate)
    opt_state = optimizer.init(params)

    loss_history = []
    velocity_history = []
    start_band = 0

    # Resume from checkpoint if available
    if config.checkpoint_dir:
        ckpt_path = Path(config.checkpoint_dir) / "fwi_checkpoint.h5"
        ckpt = _load_checkpoint(ckpt_path, expected_grid_shape=grid_shape)
        if ckpt is not None:
            start_band = ckpt["completed_bands"]
            params = ckpt["params"]
            opt_state = optimizer.init(params)
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

                def single_shot_loss(velocity, _src_pos=src_pos, _obs=obs):
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

            # Normalise gradient by max magnitude so that the optimizer
            # learning rate directly controls max velocity change in m/s.
            # With SGD(lr=50): max update = 50 m/s per iteration.
            grad_max = jnp.max(jnp.abs(grad))
            grad = grad / (grad_max + 1e-30)

            # Optimizer update + clip to physical bounds
            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = jnp.clip(params, config.c_min, config.c_max)

            if config.verbose and (it + 1) % 5 == 0:
                print(f"    Iter {it+1}/{config.n_iters_per_band}: "
                      f"loss={loss_val:.6f}, "
                      f"c=[{float(jnp.min(params)):.0f}, {float(jnp.max(params)):.0f}] m/s")

        # Save velocity snapshot at end of band
        velocity_history.append(params)

        # Checkpoint to disk for resume after preemption
        if config.checkpoint_dir:
            ckpt_path = Path(config.checkpoint_dir) / "fwi_checkpoint.h5"
            _save_checkpoint(ckpt_path, band_idx, params,
                           loss_history, velocity_history,
                           grid_shape=grid_shape)
            if config.verbose:
                print(f"  Checkpoint saved: band {band_idx+1} complete")

    return FWIResult(
        velocity=params,
        velocity_history=velocity_history,
        loss_history=loss_history,
        params=params,
        field=VoxelField(params=params),
    )


# ---------------------------------------------------------------------------
# SIREN path
# ---------------------------------------------------------------------------

def _run_fwi_siren(
    observed_data: jnp.ndarray,
    initial_velocity: jnp.ndarray,
    density: jnp.ndarray,
    dx: float,
    src_positions_grid: list,
    sensor_positions_grid: Tuple,
    source_signal: jnp.ndarray,
    dt: float,
    t_end: float,
    config: FWIConfig,
    key: jax.Array,
) -> FWIResult:
    """FWI with a SIREN-parameterised velocity field.

    Differences from the voxel path:
      - ``field`` is a ``SIRENField`` (Equinox module) pretrained against
        ``initial_velocity``.
      - Gradients come from ``eqx.filter_value_and_grad`` and are applied
        with Adam to MLP weights (no max-norm normalisation, no Gaussian
        smoothing, no mask — SIREN is smooth by construction).
      - No disk checkpoint on this path yet (pytree serialisation via
        ``eqx.tree_serialise_leaves`` is planned).
    """
    grid_shape = initial_velocity.shape
    n_sources = len(src_positions_grid)
    loss_fn = _get_loss_fn(config.loss_fn, config.envelope_weight)

    field = _init_param_field(initial_velocity, config)

    optimizer = optax.adam(config.siren_learning_rate)
    opt_state = optimizer.init(eqx.filter(field, eqx.is_inexact_array))

    loss_history: List[float] = []
    velocity_history: List[jnp.ndarray] = []

    if config.verbose:
        print(f"FWI: {len(config.freq_bands)} frequency bands, "
              f"{config.n_iters_per_band} iters/band, "
              f"{config.shots_per_iter} shots/iter")
        print(f"  Grid: {grid_shape}, dx={dx*1e3:.2f} mm")
        print(f"  Velocity bounds: [{config.c_min:.0f}, {config.c_max:.0f}] m/s")
        print(f"  Loss: {config.loss_fn}")
        print(f"  Parameterisation: SIREN "
              f"(hidden={config.siren_hidden}, layers={config.siren_layers}, "
              f"omega={config.siren_omega:g}), Adam lr={config.siren_learning_rate:g}")

    # Pre-compute time axis with a reference medium at c_max for CFL stability.
    ref_domain = build_domain(grid_shape, dx)
    ref_medium = build_medium(ref_domain, config.c_max, 1000.0, pml_size=config.pml_size)
    fixed_time_axis = build_time_axis(ref_medium, cfl=config.cfl, t_end=t_end)

    for band_idx, (fmin, fmax) in enumerate(config.freq_bands):
        if config.verbose:
            print(f"\n  Band {band_idx+1}/{len(config.freq_bands)}: "
                  f"{fmin/1e3:.0f}-{fmax/1e3:.0f} kHz")

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

            if config.shots_per_iter >= n_sources:
                shot_indices = jnp.arange(n_sources)
            else:
                shot_indices = jr.choice(
                    subkey, n_sources, shape=(config.shots_per_iter,), replace=False
                )
            shot_indices = np.array(shot_indices)

            use_checkpoint = all(s >= 128 for s in grid_shape)

            total_loss = 0.0
            grads_accum = None

            for si in shot_indices:
                src_pos = src_positions_grid[int(si)]
                obs = bp_observed[int(si)]

                def single_shot_loss(f, _src_pos=src_pos, _obs=obs):
                    velocity = f.to_velocity(config.c_min, config.c_max)
                    domain = build_domain(grid_shape, dx)
                    medium = build_medium(domain, velocity, density, pml_size=config.pml_size)
                    pred = simulate_shot_sensors(
                        medium, fixed_time_axis, _src_pos, sensor_positions_grid,
                        bp_signal, dt, checkpointed=use_checkpoint,
                    )
                    min_t = min(pred.shape[0], _obs.shape[0])
                    return loss_fn(pred[:min_t], _obs[:min_t])

                shot_loss, shot_grad = eqx.filter_value_and_grad(single_shot_loss)(field)
                total_loss = total_loss + float(shot_loss)
                if grads_accum is None:
                    grads_accum = shot_grad
                else:
                    grads_accum = jax.tree.map(_add_if_array, grads_accum, shot_grad)

            n_shots = len(shot_indices)
            loss_val = total_loss / n_shots
            grads = jax.tree.map(
                lambda g: g / n_shots if eqx.is_inexact_array(g) else g,
                grads_accum,
            )
            loss_history.append(loss_val)

            updates, opt_state = optimizer.update(grads, opt_state)
            field = eqx.apply_updates(field, updates)

            if config.verbose and (it + 1) % 5 == 0:
                vel = field.to_velocity(config.c_min, config.c_max)
                print(f"    Iter {it+1}/{config.n_iters_per_band}: "
                      f"loss={loss_val:.6f}, "
                      f"c=[{float(jnp.min(vel)):.0f}, {float(jnp.max(vel)):.0f}] m/s")

        velocity_history.append(field.to_velocity(config.c_min, config.c_max))

        if config.checkpoint_dir and config.verbose:
            # SIREN checkpointing deferred — needs eqx.tree_serialise_leaves
            # path that handles MLP weights. Voxel path is still supported.
            print(f"  (SIREN checkpoint not yet implemented; band {band_idx+1} result held in memory)")

    final_velocity = field.to_velocity(config.c_min, config.c_max)
    return FWIResult(
        velocity=final_velocity,
        velocity_history=velocity_history,
        loss_history=loss_history,
        params=final_velocity,
        field=field,
    )


def _add_if_array(a, b):
    if eqx.is_inexact_array(a) and eqx.is_inexact_array(b):
        return a + b
    return a

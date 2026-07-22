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
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from ..simulation.forward import (
    build_domain,
    build_medium,
    build_time_axis,
    simulate_shot_sensors,
    _build_source_signal,
)
from ..constitutive import alpha_from_speed, manifold_proximal
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

    # --- Absorption-aware FWI ---
    # Known, FIXED power-law attenuation field (dB/cm/MHz^alpha_power), e.g.
    # derived from a CT skull. Folded into the forward medium each iteration
    # (Treeby-Cox absorbing EoS) so the inversion matches the attenuated
    # waveforms instead of (wrongly) explaining the missing amplitude with
    # velocity structure. None = lossless (default, unchanged behaviour). Only
    # velocity is inverted; alpha is held fixed.
    attenuation: Optional[jnp.ndarray] = None
    alpha_power: float = 1.5

    # --- Phase 6: multiparameter (velocity + attenuation) inversion ---
    # When True, alpha is a CO-INVERTED voxel field (not frozen). The GM/WM
    # contrast channel is attenuation, not sound speed (Kang 2022: WM alpha ~1.5x
    # GM, while GM/WM speed difference is at the noise floor) — velocity-only FWI
    # cannot resolve it. State becomes a ``{"c", "a"}`` pytree; when False the
    # state stays a bare velocity array and behaviour is byte-identical.
    invert_attenuation: bool = False
    attenuation_lr: float = 0.5  # Max alpha update/iter (dB/cm/MHz^alpha_power)
    # Starting alpha field. None -> ``attenuation`` if given (array), else zeros.
    attenuation_init: Optional[jnp.ndarray] = None
    attenuation_max: float = 20.0  # alpha clipped to [0, this]; 0 = energy dissipation
    attenuation_mask: Optional[jnp.ndarray] = None  # where alpha is updated

    # Phase 6.2: CANN-derived tissue-α manifold prior (issue #45). Projects the
    # inverted α toward the nearest physical tissue archetype after each step —
    # "a few coefficients + a spatial field" instead of a smeared image. Ramped
    # in LATE (after the data has grown α past the archetype midpoint) because a
    # sub-midpoint value snaps toward water. None = no prior (free-voxel α).
    attenuation_archetypes: Optional[jnp.ndarray] = None
    attenuation_prior_weight: float = 0.0    # proximal pull strength beta in [0,1]
    attenuation_prior_ramp: float = 0.5      # fraction of a band's iters before prior engages

    # Constitutive velocity→attenuation coupling (issue #45). The well-resolved
    # sound speed predicts α through the same-tissue (c, α) relation (KK/causal
    # premise), letting the strong channel inform the weak one. Provide the
    # (c_anchors, alpha_anchors) from ``constitutive.speed_alpha_anchors``.
    # Degenerate for GM/WM (equal c) — helps c-contrasted tissues only.
    attenuation_speed_anchors: Optional[tuple] = None
    attenuation_speed_weight: float = 0.0    # pull toward c-predicted α in [0,1]

    # Hierarchical c-first schedule (issue #45 recipe). α is FROZEN at its init
    # until the global iteration reaches this fraction of the total, then
    # released. This avoids the c-α crosstalk where a free-from-iter-0 α absorbs
    # the amplitude misfit and starves velocity recovery — recover c first, then
    # release α (with the constitutive coupling) into a well-resolved c. 0.0 =
    # co-invert from the start (unchanged default); ~0.5 = c-only first half.
    attenuation_release_frac: float = 0.0

    checkpoint_dir: Optional[str] = None  # Save/resume state after each band
    precondition: bool = False  # Pseudo-Hessian source illumination compensation
    # Illumination "water level" for preconditioning, as a fraction of peak
    # illumination. The gradient is divided by ``illum + precondition_floor *
    # max(illum)``. A near-zero floor (the old 1e-12) fully normalises every
    # voxel, which AMPLIFIES noise in the poorly-lit deep brain into speckle;
    # too large a floor under-corrects the bright near-skull periphery, leaving
    # a ring. ~0.05 (5 %) tempers both — normalise the periphery, damp (not
    # amplify) the interior. Default preserves the old full-normalisation.
    precondition_floor: float = 1e-12
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

    # --- Phase 3 score-prior regulariser (SIREN path only) ---
    # When ``score_prior_fn`` is provided AND ``score_prior_weight > 0``,
    # the FWI gradient at each iter is composed with a Phase-3 score
    # prior: ``effective_grad = ∇L_data − λ · s_φ(θ, t_eps)``. Defaults
    # leave existing callers unaffected.
    score_prior_fn: Optional[Any] = None  # (theta_flat, t) -> ℝ^D
    score_prior_weight: float = 0.0
    score_prior_t_eps: float = 0.01


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
    # Phase 6: recovered attenuation field (dB/cm/MHz^alpha_power) when
    # ``invert_attenuation=True``; None for the velocity-only path.
    attenuation: Optional[jnp.ndarray] = None


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


def _finalize_gradient(grad, grad_sq_accum, n_shots, mask, config):
    """Precondition → smooth → mask → max-normalise a single-field gradient.

    Shared by the velocity-only and multiparameter (Phase 6) paths so every
    inverted field (c and, when enabled, α) gets identical treatment. After
    max-normalisation the field's max update per iteration is controlled solely
    by its optimiser step size.
    """
    if config.precondition:
        illum = jnp.sqrt(grad_sq_accum / n_shots)
        grad = grad / (illum + config.precondition_floor * jnp.max(illum))
    if config.gradient_smooth_sigma > 0:
        grad = _smooth_gradient(grad, config.gradient_smooth_sigma)
    if mask is not None:
        grad = grad * mask
    return grad / (jnp.max(jnp.abs(grad)) + 1e-30)


def _velocity_of(params):
    """Extract the velocity array from the (possibly multiparameter) state."""
    return params["c"] if isinstance(params, dict) else params


def _alpha_released(band_idx: int, it: int, config: "FWIConfig") -> bool:
    """Hierarchical c-first schedule: has attenuation inversion engaged yet?

    α is frozen until the global iteration (across bands) reaches
    ``attenuation_release_frac`` of the total. Default frac 0.0 → released from
    the first iteration (co-inversion, unchanged behaviour).
    """
    total = max(1, config.n_iters_per_band * len(config.freq_bands))
    global_iter = band_idx * config.n_iters_per_band + it
    return bool(global_iter >= config.attenuation_release_frac * total)


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

    # Phase 6: co-invert attenuation. State becomes a {"c", "a"} pytree; the
    # velocity-only path below is byte-identical (params stays a bare array).
    if config.invert_attenuation:
        if config.attenuation_init is not None:
            a0 = jnp.asarray(config.attenuation_init, dtype=params.dtype)
        elif config.attenuation is not None and not isinstance(
                config.attenuation, (int, float)):
            a0 = jnp.asarray(config.attenuation, dtype=params.dtype)
        else:
            a0 = jnp.zeros_like(params)
        params = {"c": params, "a": a0}

    # Steepest descent with gradient normalisation (Stride-style).
    # Combined with max_step_m_per_s, the learning rate directly controls
    # the maximum velocity change per iteration in m/s.
    optimizer = optax.sgd(config.learning_rate)
    opt_state = optimizer.init(params)

    loss_history = []
    velocity_history = []
    start_band = 0

    # Resume from checkpoint if available. Multiparameter (Phase 6) state is not
    # yet checkpointed (velocity-only save/resume) — tracked in issue #44.
    if config.checkpoint_dir and not config.invert_attenuation:
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
            grad_accum = jax.tree_util.tree_map(jnp.zeros_like, params)
            grad_sq_accum = jax.tree_util.tree_map(jnp.zeros_like, params)

            # Use checkpointed scan for large grids (>= 128^3)
            use_checkpoint = all(s >= 128 for s in grid_shape)

            for si in shot_indices:
                src_pos = src_positions_grid[int(si)]
                obs = bp_observed[int(si)]

                def single_shot_loss(p, _src_pos=src_pos, _obs=obs):
                    if config.invert_attenuation:
                        velocity, atten = p["c"], p["a"]
                    else:
                        velocity, atten = p, config.attenuation
                    domain = build_domain(grid_shape, dx)
                    medium = build_medium(
                        domain, velocity, density, pml_size=config.pml_size,
                        attenuation=atten, alpha_power=config.alpha_power,
                    )
                    pred = simulate_shot_sensors(
                        medium, fixed_time_axis, _src_pos, sensor_positions_grid,
                        bp_signal, dt, checkpointed=use_checkpoint,
                    )
                    min_t = min(pred.shape[0], _obs.shape[0])
                    return loss_fn(pred[:min_t], _obs[:min_t])

                shot_loss, shot_grad = jax.value_and_grad(single_shot_loss)(params)
                total_loss = total_loss + float(shot_loss)
                grad_accum = jax.tree_util.tree_map(
                    lambda a, g: a + g, grad_accum, shot_grad)
                grad_sq_accum = jax.tree_util.tree_map(
                    lambda a, g: a + g ** 2, grad_sq_accum, shot_grad)

            n_shots = len(shot_indices)
            loss_val = total_loss / n_shots
            grad = jax.tree_util.tree_map(lambda g: g / n_shots, grad_accum)
            loss_history.append(loss_val)

            # Precondition → smooth → mask → max-normalise. For the velocity-only
            # path this is a single field; for Phase 6 each leaf (c, α) is
            # finalised independently with its own mask, and α is scaled to its
            # own step size (attenuation_lr) since the shared SGD carries lr=c-step.
            alpha_released = config.invert_attenuation and _alpha_released(
                band_idx, it, config)
            if config.invert_attenuation:
                gc = _finalize_gradient(
                    grad["c"], grad_sq_accum["c"], n_shots, config.mask, config)
                if alpha_released:
                    ga = _finalize_gradient(
                        grad["a"], grad_sq_accum["a"], n_shots,
                        config.attenuation_mask, config)
                    ga = ga * (config.attenuation_lr / config.learning_rate)
                else:
                    ga = jnp.zeros_like(grad["a"])   # c-first: α frozen
                grad = {"c": gc, "a": ga}
            else:
                grad = _finalize_gradient(
                    grad, grad_sq_accum, n_shots, config.mask, config)

            # Optimizer update + clip to physical bounds
            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            if config.invert_attenuation:
                c_new = jnp.clip(params["c"], config.c_min, config.c_max)
                a_new = jnp.clip(params["a"], 0.0, config.attenuation_max)
                # Regularisers engage only once α is released and past the
                # per-band ramp (while frozen, α is held exactly at its init).
                past_ramp = alpha_released and (
                    it >= config.attenuation_prior_ramp * config.n_iters_per_band)
                # Tissue-α manifold prior (issue #45): ramp in late so the data
                # has grown α past the archetype midpoint before we project.
                if (config.attenuation_archetypes is not None
                        and config.attenuation_prior_weight > 0.0 and past_ramp):
                    a_new = manifold_proximal(
                        a_new, config.attenuation_archetypes,
                        beta=config.attenuation_prior_weight)
                # Constitutive c→α coupling (issue #45): the well-resolved
                # velocity predicts α (same-tissue relation), pulling weak α
                # toward the strong channel's implication. Ramped like the prior.
                if (config.attenuation_speed_anchors is not None
                        and config.attenuation_speed_weight > 0.0 and past_ramp):
                    c_anch, a_anch = config.attenuation_speed_anchors
                    pred = alpha_from_speed(c_new, c_anch, a_anch)
                    a_new = a_new + config.attenuation_speed_weight * (pred - a_new)
                    a_new = jnp.clip(a_new, 0.0, config.attenuation_max)
                params = {"c": c_new, "a": a_new}
            else:
                params = jnp.clip(params, config.c_min, config.c_max)

            if config.verbose and (it + 1) % 5 == 0:
                _c = _velocity_of(params)
                print(f"    Iter {it+1}/{config.n_iters_per_band}: "
                      f"loss={loss_val:.6f}, "
                      f"c=[{float(jnp.min(_c)):.0f}, {float(jnp.max(_c)):.0f}] m/s")

        # Save velocity snapshot at end of band
        velocity_history.append(_velocity_of(params))

        # Checkpoint to disk for resume after preemption (velocity-only path)
        if config.checkpoint_dir and not config.invert_attenuation:
            ckpt_path = Path(config.checkpoint_dir) / "fwi_checkpoint.h5"
            _save_checkpoint(ckpt_path, band_idx, params,
                           loss_history, velocity_history,
                           grid_shape=grid_shape)
            if config.verbose:
                print(f"  Checkpoint saved: band {band_idx+1} complete")

    final_velocity = _velocity_of(params)
    return FWIResult(
        velocity=final_velocity,
        velocity_history=velocity_history,
        loss_history=loss_history,
        params=final_velocity,
        field=VoxelField(params=final_velocity),
        attenuation=params["a"] if isinstance(params, dict) else None,
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
                    medium = build_medium(
                        domain, velocity, density, pml_size=config.pml_size,
                        attenuation=config.attenuation, alpha_power=config.alpha_power,
                    )
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

            # Phase 3 score-prior regulariser. With ``score_prior_fn=None``
            # or ``weight=0`` this is a no-op and the original FWI gradient
            # pipeline runs unchanged.
            if config.score_prior_fn is not None and config.score_prior_weight > 0:
                from brain_fwi.inference.diffusion import (
                    compose_siren_grad_with_score_prior,
                )
                grads = compose_siren_grad_with_score_prior(
                    grads, field, config.score_prior_fn,
                    weight=config.score_prior_weight,
                    t_eps=config.score_prior_t_eps,
                )

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

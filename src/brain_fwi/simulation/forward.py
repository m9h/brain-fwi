"""Forward acoustic simulation via j-Wave for brain FWI.

Wraps j-Wave's pseudospectral time-domain (PSTD) solver to provide:
  - Domain and medium construction from acoustic property arrays
  - Single-shot simulation with sensor recording
  - Batched data generation (all sources → all receivers)

The forward operator is fully differentiable via JAX autodiff, which is
the key advantage over Stride's Devito-based adjoint approach.

References:
    - Stanziola et al. (2022). j-Wave. arXiv:2207.01499.
    - Treeby & Cox (2010). k-Wave: MATLAB toolbox.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Optional, Tuple, Union
from functools import partial

from ..utils.wavelets import ricker_wavelet


# ---------------------------------------------------------------------------
# Domain / Medium construction
# ---------------------------------------------------------------------------

def build_domain(grid_shape: Tuple[int, ...], dx: float):
    """Create a j-Wave computational domain.

    Args:
        grid_shape: (nx, ny) or (nx, ny, nz).
        dx: Uniform grid spacing in metres.

    Returns:
        jaxdf.geometry.Domain
    """
    from jaxdf.geometry import Domain
    ndim = len(grid_shape)
    return Domain(grid_shape, tuple([dx] * ndim))


def build_medium(
    domain,
    sound_speed: Union[float, jnp.ndarray],
    density: Union[float, jnp.ndarray],
    pml_size: int = 20,
    attenuation: Union[float, jnp.ndarray, None] = None,
    alpha_power: float = 1.5,
):
    """Create a j-Wave medium from acoustic property arrays.

    Args:
        domain: jaxdf Domain.
        sound_speed: Scalar or array matching domain.N. Units: m/s.
        density: Scalar or array. Units: kg/m^3.
        pml_size: PML absorbing boundary thickness in grid points.
        attenuation: Scalar or array. Units: dB/cm/MHz^alpha_power.
            None = lossless.
        alpha_power: Power-law exponent ``y`` for absorption,
            ``α(ω) ∝ ω^y``. Cortical bone ≈ 1.5; soft tissue ≈ 1.1.
            j-Wave's Stokes default is 2.0, but that's singular for the
            time-domain Treeby-Cox absorber (1/sin(πy/2)) so we default
            to 1.5 — appropriate for skull-imaging FWI.

    Returns:
        jwave.geometry.Medium
    """
    from jwave import FourierSeries
    from jwave.geometry import Medium

    def to_field(val):
        if isinstance(val, (int, float)):
            return val
        arr = jnp.asarray(val, dtype=jnp.float32)
        if arr.ndim == len(domain.N):
            arr = arr[..., jnp.newaxis]
        return FourierSeries(arr, domain)

    kwargs = dict(
        domain=domain,
        sound_speed=to_field(sound_speed),
        density=to_field(density),
        pml_size=pml_size,
        alpha_power=alpha_power,
    )
    if attenuation is not None:
        kwargs["attenuation"] = to_field(attenuation)
    else:
        # Explicit zero so the time-domain absorber treats lossless as
        # "skip" rather than picking up Medium's default scalar α=1.0.
        kwargs["attenuation"] = 0.0

    return Medium(**kwargs)


def build_time_axis(medium, cfl: float = 0.3, t_end: Optional[float] = None):
    """Compute stable time axis from medium properties.

    Args:
        medium: j-Wave Medium.
        cfl: CFL stability number (< 1).
        t_end: End time in seconds. None = auto from domain traversal.

    Returns:
        jwave.geometry.TimeAxis
    """
    from jwave.geometry import TimeAxis
    return TimeAxis.from_medium(medium, cfl=cfl, t_end=t_end)


# ---------------------------------------------------------------------------
# Source construction
# ---------------------------------------------------------------------------

def _build_source_signal(
    freq: float,
    dt: float,
    n_samples: int,
) -> jnp.ndarray:
    """Build a Ricker wavelet source signal."""
    return ricker_wavelet(freq, dt, n_samples)


def _build_sources(
    domain,
    src_position_grid: Tuple[int, ...],
    signal: jnp.ndarray,
    dt: float,
):
    """Create j-Wave Sources for a single point source.

    Args:
        domain: jaxdf Domain.
        src_position_grid: Grid indices (ix, iy) or (ix, iy, iz).
        signal: (n_samples,) source signal.
        dt: Time step.

    Returns:
        jwave.geometry.Sources
    """
    from jwave.geometry import Sources

    ndim = len(domain.N)
    pos_tuple = tuple([src_position_grid[d]] for d in range(ndim))
    signals = signal[jnp.newaxis, :]  # (1, n_samples)

    return Sources(
        positions=pos_tuple,
        signals=signals,
        dt=dt,
        domain=domain,
    )


def _build_sensors(domain, sensor_positions_grid: Tuple):
    """Create j-Wave Sensors at grid positions.

    Args:
        domain: jaxdf Domain.
        sensor_positions_grid: Tuple of arrays, one per dimension.
            Each array is (n_sensors,) of int indices.

    Returns:
        jwave.geometry.Sensors
    """
    from jwave.geometry import Sensors
    positions = tuple(
        [int(sensor_positions_grid[d][i]) for i in range(len(sensor_positions_grid[0]))]
        for d in range(len(domain.N))
    )
    return Sensors(positions=positions)


# ---------------------------------------------------------------------------
# Single-shot simulation
# ---------------------------------------------------------------------------

def simulate_shot(
    medium,
    time_axis,
    src_position_grid: Tuple[int, ...],
    freq: float,
    checkpoint: bool = True,
) -> jnp.ndarray:
    """Simulate a single source and return the full pressure field.

    Returns the final-timestep pressure field (for visualization).
    For FWI, use simulate_shot_sensors instead.

    Args:
        medium: j-Wave Medium.
        time_axis: j-Wave TimeAxis.
        src_position_grid: Source position as grid indices.
        freq: Source frequency in Hz.
        checkpoint: Use gradient checkpointing to save memory.

    Returns:
        Pressure field array with shape matching domain.
    """
    from jwave.acoustics.time_varying import simulate_wave_propagation

    dt = float(time_axis.dt)
    n_samples = int(float(time_axis.t_end) / dt)
    signal = _build_source_signal(freq, dt, n_samples)
    sources = _build_sources(medium.domain, src_position_grid, signal, dt)

    settings = None
    if checkpoint:
        from jwave.acoustics.time_varying import TimeWavePropagationSettings
        settings = TimeWavePropagationSettings(checkpoint=True)

    p_final = simulate_wave_propagation(
        medium, time_axis, sources=sources, settings=settings,
    )
    return p_final


def simulate_shot_sensors(
    medium,
    time_axis,
    src_position_grid: Tuple[int, ...],
    sensor_positions_grid: Tuple,
    source_signal: jnp.ndarray,
    dt: float,
    checkpointed: bool = False,
) -> jnp.ndarray:
    """Simulate a single source and record at sensor positions.

    This is the core forward operator for FWI. It runs a full time-domain
    simulation and extracts the pressure time series at sensor locations.

    Args:
        medium: j-Wave Medium.
        time_axis: j-Wave TimeAxis.
        src_position_grid: Source grid indices (ix, iy[, iz]).
        sensor_positions_grid: Tuple of (n_sensors,) index arrays.
        source_signal: (n_samples,) source wavelet.
        dt: Time step.
        checkpointed: Use segmented gradient checkpointing for large grids.
            Trades ~3x compute for O(sqrt(N)) memory on the backward pass.

    Returns:
        (n_timesteps, n_sensors) array of recorded pressure.
    """
    if checkpointed:
        return _simulate_shot_sensors_checkpointed(
            medium, time_axis, src_position_grid, sensor_positions_grid,
            source_signal, dt,
        )

    from jwave.acoustics.time_varying import (
        simulate_wave_propagation,
        TimeWavePropagationSettings,
    )

    sources = _build_sources(medium.domain, src_position_grid, source_signal, dt)
    sensors = _build_sensors(medium.domain, sensor_positions_grid)

    settings = TimeWavePropagationSettings(checkpoint=True)

    # Pass sensors so scan output is (n_timesteps, n_sensors) directly,
    # instead of the full 3D pressure field at every timestep.
    ys = simulate_wave_propagation(
        medium, time_axis, sources=sources, sensors=sensors, settings=settings,
    )

    # ys is (n_timesteps, n_sensors) — sensor values at each step
    return _to_array(ys)


def _extract_sensor_data(
    pressure_output,
    sensor_positions: Tuple,
) -> jnp.ndarray:
    """Extract time series at sensor positions from j-Wave output.

    j-Wave returns different types depending on version:
    - List of FourierSeries (one per timestep)
    - Stacked array (n_timesteps, *spatial_dims)
    - Single FourierSeries with time dimension

    Args:
        pressure_output: j-Wave simulation output.
        sensor_positions: Tuple of index arrays (one per spatial dim).

    Returns:
        (n_timesteps, n_sensors) array.
    """
    # Convert to raw array
    p_data = _to_array(pressure_output)

    if p_data.ndim == len(sensor_positions) + 1:
        # Shape: (n_timesteps, *spatial_dims)
        # Index into spatial dims at each sensor position
        return p_data[:, sensor_positions[0], sensor_positions[1]] \
            if len(sensor_positions) == 2 \
            else p_data[:, sensor_positions[0], sensor_positions[1], sensor_positions[2]]
    else:
        # Single snapshot — expand time dim
        if len(sensor_positions) == 2:
            return p_data[sensor_positions[0], sensor_positions[1]][jnp.newaxis, :]
        return p_data[sensor_positions[0], sensor_positions[1], sensor_positions[2]][jnp.newaxis, :]


def _to_array(pressure_output) -> jnp.ndarray:
    """Convert j-Wave pressure output to a plain JAX array."""
    if isinstance(pressure_output, (list, tuple)):
        frames = []
        for frame in pressure_output:
            if hasattr(frame, "params"):
                data = frame.params
            elif hasattr(frame, "on_grid"):
                data = frame.on_grid
            else:
                data = jnp.asarray(frame)
            if data.ndim > 0 and data.shape[-1] == 1:
                data = data[..., 0]
            frames.append(data)
        return jnp.stack(frames, axis=0)
    elif hasattr(pressure_output, "params"):
        data = pressure_output.params
        if data.shape[-1] == 1:
            data = data[..., 0]
        return data
    else:
        data = jnp.asarray(pressure_output)
        if data.ndim > 1 and data.shape[-1] == 1:
            data = data[..., 0]
        return data


# ---------------------------------------------------------------------------
# Batched data generation
# ---------------------------------------------------------------------------

def generate_observed_data(
    sound_speed: jnp.ndarray,
    density: jnp.ndarray,
    dx: float,
    src_positions_grid: list,
    sensor_positions_grid: Tuple,
    freq: float,
    pml_size: int = 20,
    cfl: float = 0.3,
    t_end: Optional[float] = None,
    time_axis=None,
    source_signal: Optional[jnp.ndarray] = None,
    dt: Optional[float] = None,
    verbose: bool = True,
) -> jnp.ndarray:
    """Generate synthetic observed data for all source-receiver pairs.

    Runs one forward simulation per source, records at all sensors.
    This is used to create the 'ground truth' data for FWI.

    IMPORTANT: For consistency with FWI, pass the same time_axis,
    source_signal, and dt that will be used during inversion. If not
    provided, they are computed from the medium (which may produce
    a different dt than the FWI uses).

    Args:
        sound_speed: (*spatial_dims) sound speed array (m/s).
        density: (*spatial_dims) density array (kg/m^3).
        dx: Grid spacing (m).
        src_positions_grid: List of tuples, each (ix, iy[, iz]).
        sensor_positions_grid: Tuple of index arrays for receivers.
        freq: Source frequency (Hz).
        pml_size: PML thickness.
        cfl: CFL number.
        t_end: Simulation end time. None = auto.
        time_axis: Pre-computed TimeAxis. None = compute from medium.
        source_signal: Pre-computed source wavelet. None = build Ricker.
        dt: Time step matching source_signal. Required if source_signal given.
        verbose: Print progress.

    Returns:
        (n_sources, n_timesteps, n_sensors) array of recorded data.
    """
    grid_shape = sound_speed.shape
    domain = build_domain(grid_shape, dx)
    medium = build_medium(domain, sound_speed, density, pml_size=pml_size)

    if time_axis is None:
        time_axis = build_time_axis(medium, cfl=cfl, t_end=t_end)

    if dt is None:
        dt = float(time_axis.dt)

    if source_signal is None:
        n_samples = int(float(time_axis.t_end) / dt)
        source_signal = _build_source_signal(freq, dt, n_samples)

    n_sources = len(src_positions_grid)
    all_data = []

    for i, src_pos in enumerate(src_positions_grid):
        if verbose:
            print(f"  Forward shot {i+1}/{n_sources}", end="\r")

        d = simulate_shot_sensors(
            medium, time_axis, src_pos, sensor_positions_grid,
            source_signal, dt,
        )
        all_data.append(d)

    if verbose:
        print(f"  Forward shots: {n_sources}/{n_sources} done")

    return jnp.stack(all_data, axis=0)


# ---------------------------------------------------------------------------
# Checkpointed simulation for large grids (192^3+)
# ---------------------------------------------------------------------------

def _simulate_shot_sensors_checkpointed(
    medium,
    time_axis,
    src_position_grid: Tuple[int, ...],
    sensor_positions_grid: Tuple,
    source_signal: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Memory-efficient forward simulation using segmented checkpointing.

    Reimplements j-Wave's time-stepping loop with a two-level nested scan
    so that autodiff stores only O(sqrt(N)) intermediate carries instead
    of all N. This reduces backward-pass memory from ~218 GB to ~13 GB
    at 192^3 with 1102 timesteps.

    The physics is identical to j-Wave's simulate_wave_propagation —
    we use the same RHS functions and k-space operators.
    """
    from jwave import FourierSeries
    from jwave.acoustics.time_varying import (
        fourier_wave_prop_params,
        momentum_conservation_rhs,
        mass_conservation_rhs,
        pressure_from_density,
        TimeWavePropagationSettings,
    )
    from .checkpointed_scan import checkpointed_scan

    sources = _build_sources(medium.domain, src_position_grid, source_signal, dt)
    sensors = _build_sensors(medium.domain, sensor_positions_grid)

    settings = TimeWavePropagationSettings(checkpoint=False)

    # Get PML and k-space parameters (same as j-Wave internally computes)
    params = fourier_wave_prop_params(medium, time_axis, settings=settings)
    c_ref = params["c_ref"]
    pml_rho = params["pml_rho"]
    pml_u = params["pml_u"]

    dt_val = time_axis.dt
    output_steps = jnp.arange(0, time_axis.Nt, 1)

    # Initialize fields — mirrors j-Wave's initialization (time_varying.py:582-613)
    ndim = len(medium.domain.N)
    shape = tuple(list(medium.domain.N) + [ndim])
    shape_one = tuple(list(medium.domain.N) + [1])

    p0 = pml_rho.replace_params(jnp.zeros(shape_one))
    u0 = pml_u.replace_params(jnp.zeros(shape))

    rho = p0.replace_params(
        jnp.stack([p0.params[..., i] for i in range(ndim)], axis=-1)
    ) / ndim
    rho = rho / (medium.sound_speed ** 2)

    fields = [p0, u0, rho]

    # Step function — identical to j-Wave's scan_fun (time_varying.py:615-640)
    def scan_fun(fields, n):
        p, u, rho_f = fields
        mass_src_field = sources.on_grid(n)

        du = momentum_conservation_rhs(
            p, u, medium, c_ref=c_ref, dt=dt_val, params=params["fourier"],
        )
        u = pml_u * (pml_u * u + dt_val * du)

        drho = mass_conservation_rhs(
            p, u, mass_src_field, medium,
            c_ref=c_ref, dt=dt_val, params=params["fourier"],
        )
        rho_f = pml_rho * (pml_rho * rho_f + dt_val * drho)

        p = pressure_from_density(rho_f, medium)
        return [p, u, rho_f], sensors(p, u, rho_f)

    # Run with segmented checkpointing
    _, ys = checkpointed_scan(scan_fun, fields, output_steps)

    return _to_array(ys)

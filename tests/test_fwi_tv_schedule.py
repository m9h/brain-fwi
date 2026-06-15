"""Per-band TV weight schedule.

Once the band-1 stage has smoothed the model with TV, band 2 should
be free to sharpen inclusion edges without TV's edge-smoothing penalty
holding it back. This test pins the contract that
`FWIConfig.tv_weight_schedule = [w1, w2, ...]` (one entry per freq
band) is honoured by `run_fwi`, overriding the scalar `tv_weight`.
"""

import jax.numpy as jnp
import pytest

from brain_fwi.transducers import helmet_array_3d, transducer_positions_to_grid
from brain_fwi.simulation.forward import generate_observed_data
from brain_fwi.utils.wavelets import ricker_wavelet
from brain_fwi.inversion.fwi import FWIConfig, run_fwi


def _build_problem():
    grid_shape = (20, 20, 20)
    dx = 1.0e-3
    zz, yy, xx = jnp.meshgrid(
        jnp.arange(grid_shape[0]), jnp.arange(grid_shape[1]),
        jnp.arange(grid_shape[2]), indexing="ij",
    )
    r2 = (xx - 10) ** 2 + (yy - 10) ** 2 + (zz - 10) ** 2
    c_true = jnp.where(r2 < 6 ** 2, 1650.0, 1500.0).astype(jnp.float32)
    rho = jnp.full(grid_shape, 1000.0, dtype=jnp.float32)
    return c_true, rho, dx, grid_shape


def _run(c_true, rho, dx, grid_shape, tv_weight=0.0, tv_weight_schedule=None):
    n_elements = 12
    cx, cy, cz = (grid_shape[i] * dx / 2 for i in range(3))
    positions = helmet_array_3d(
        n_elements=n_elements, center=(cx, cy, cz),
        radius_ap=(grid_shape[0] * dx - 4e-3) / 2,
        radius_lr=(grid_shape[1] * dx - 4e-3) / 2,
        radius_si=(grid_shape[2] * dx - 4e-3) / 2,
        standoff=0.0, coverage_angle=3.14, exclude_face=False,
    )
    pos_grid = transducer_positions_to_grid(positions, dx, grid_shape)
    n_actual = len(pos_grid[0])
    src_list = [
        (int(pos_grid[0][i]), int(pos_grid[1][i]), int(pos_grid[2][i]))
        for i in range(n_actual)
    ]
    sensor_pos = pos_grid

    dt = 80e-9
    n_t = 400
    t_end = dt * n_t
    source_signal = ricker_wavelet(f0=200e3, dt=dt, n_samples=n_t)

    observed = generate_observed_data(
        sound_speed=c_true, density=rho, dx=dx,
        src_positions_grid=src_list, sensor_positions_grid=sensor_pos,
        freq=0.5e6, pml_size=6, cfl=0.3, t_end=t_end,
        source_signal=source_signal, dt=dt, verbose=False,
    )

    c_init = jnp.full(grid_shape, 1500.0, dtype=jnp.float32)
    config = FWIConfig(
        freq_bands=[(0.0, 0.3e6), (0.0, 0.5e6)],
        n_iters_per_band=4,
        shots_per_iter=n_actual,
        learning_rate=5.0,
        c_min=1400.0,
        c_max=1700.0,
        pml_size=6,
        gradient_smooth_sigma=0.5,
        loss_fn="l2",
        optimizer="adam",
        tv_weight=tv_weight,
        tv_weight_schedule=tv_weight_schedule,
        verbose=False,
    )
    result = run_fwi(
        observed_data=observed,
        initial_velocity=c_init,
        density=rho, dx=dx,
        src_positions_grid=src_list,
        sensor_positions_grid=sensor_pos,
        source_signal=source_signal,
        dt=dt, t_end=t_end, config=config,
    )
    return result.velocity


def test_tv_weight_schedule_overrides_scalar():
    """Running with tv_weight_schedule=[2.0, 0.0] vs tv_weight=2.0
    (constant) should produce *different* reconstructions: the
    schedule lets band 2 sharpen edges where the constant keeps
    smoothing. If they're identical, the schedule isn't being used.
    """
    c_true, rho, dx, grid_shape = _build_problem()

    recon_const = _run(c_true, rho, dx, grid_shape, tv_weight=2.0)
    recon_sched = _run(
        c_true, rho, dx, grid_shape,
        tv_weight=0.0,
        tv_weight_schedule=[2.0, 0.0],
    )

    diff = float(jnp.max(jnp.abs(recon_sched - recon_const)))
    print(f"\nMax abs diff between constant-TV and scheduled-TV recons: "
          f"{diff:.3f} m/s")

    assert diff > 5.0, (
        f"Schedule had no effect: max abs diff {diff:.3f} m/s. "
        f"tv_weight_schedule must override the scalar tv_weight on a "
        f"per-band basis."
    )

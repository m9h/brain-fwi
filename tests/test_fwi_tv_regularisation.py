"""TV regularisation in brain-fwi reduces water-region noise.

Context: the v2 3D breast FWI run (2026-05-10) recovered inclusion
amplitudes correctly but produced salt-and-pepper noise across the
entire water background — voxels were driven toward the [1400, 1700]
clip bounds. Total Variation regularisation penalises non-physical
high-frequency oscillations and should reduce that noise.

This test runs the same small 3D phantom from
`test_breast_3d_fwi_descent.py` twice — once with tv_weight=0, once
with tv_weight>0 — and asserts that the water-region std-dev of the
reconstruction is meaningfully lower with TV on.
"""

import jax.numpy as jnp
import pytest

from brain_fwi.transducers import helmet_array_3d, transducer_positions_to_grid
from brain_fwi.simulation.forward import generate_observed_data
from brain_fwi.utils.wavelets import ricker_wavelet
from brain_fwi.inversion.fwi import FWIConfig, run_fwi


@pytest.fixture
def small_3d_phantom():
    grid_shape = (24, 24, 24)
    dx = 1.0e-3
    zz, yy, xx = jnp.meshgrid(
        jnp.arange(grid_shape[0]),
        jnp.arange(grid_shape[1]),
        jnp.arange(grid_shape[2]),
        indexing="ij",
    )
    r2 = (xx - 12) ** 2 + (yy - 12) ** 2 + (zz - 12) ** 2
    c_true = jnp.where(r2 < 7 ** 2, 1650.0, 1500.0).astype(jnp.float32)
    rho = jnp.full(grid_shape, 1000.0, dtype=jnp.float32)
    return c_true, rho, dx, grid_shape


def _run_one(c_true, rho, dx, grid_shape, tv_weight):
    n_elements = 16
    cx = grid_shape[0] * dx / 2
    cy = grid_shape[1] * dx / 2
    cz = grid_shape[2] * dx / 2
    positions = helmet_array_3d(
        n_elements=n_elements,
        center=(cx, cy, cz),
        radius_ap=(grid_shape[0] * dx - 4e-3) / 2,
        radius_lr=(grid_shape[1] * dx - 4e-3) / 2,
        radius_si=(grid_shape[2] * dx - 4e-3) / 2,
        standoff=0.0,
        coverage_angle=3.14,
        exclude_face=False,
    )
    pos_grid = transducer_positions_to_grid(positions, dx, grid_shape)
    n_actual = len(pos_grid[0])
    src_list = [
        (int(pos_grid[0][i]), int(pos_grid[1][i]), int(pos_grid[2][i]))
        for i in range(n_actual)
    ]
    sensor_pos = pos_grid

    dt = 80e-9
    n_t = 500
    t_end = dt * n_t
    source_signal = ricker_wavelet(f0=200e3, dt=dt, n_samples=n_t)

    observed = generate_observed_data(
        sound_speed=c_true, density=rho, dx=dx,
        src_positions_grid=src_list, sensor_positions_grid=sensor_pos,
        freq=0.5e6, pml_size=8, cfl=0.3, t_end=t_end,
        source_signal=source_signal, dt=dt, verbose=False,
    )

    c_init = jnp.full(grid_shape, 1500.0, dtype=jnp.float32)
    config = FWIConfig(
        freq_bands=[(0.0, 0.3e6)],
        n_iters_per_band=8,
        shots_per_iter=n_actual,
        learning_rate=5.0,
        c_min=1400.0,
        c_max=1700.0,
        pml_size=8,
        gradient_smooth_sigma=0.5,
        loss_fn="l2",
        optimizer="adam",
        tv_weight=tv_weight,
        verbose=False,
    )
    result = run_fwi(
        observed_data=observed,
        initial_velocity=c_init,
        density=rho,
        dx=dx,
        src_positions_grid=src_list,
        sensor_positions_grid=sensor_pos,
        source_signal=source_signal,
        dt=dt,
        t_end=t_end,
        config=config,
    )
    return result.velocity


def test_tv_regularisation_reduces_water_region_noise(small_3d_phantom):
    """With TV regularisation, the std-dev of recon vp inside the water
    region (where true vp is uniformly 1500) should be at least 30%
    lower than without TV. TV penalises high-frequency oscillations,
    so water voxels — which have no signal — should stay closer to a
    smooth average instead of wandering.
    """
    c_true, rho, dx, grid_shape = small_3d_phantom
    water_mask = jnp.abs(c_true - 1500.0) < 1.0

    recon_no_tv = _run_one(c_true, rho, dx, grid_shape, tv_weight=0.0)
    recon_tv = _run_one(c_true, rho, dx, grid_shape, tv_weight=5.0)

    std_water_no_tv = float(jnp.std(recon_no_tv[water_mask]))
    std_water_tv = float(jnp.std(recon_tv[water_mask]))

    print(f"\nWater-region recon std-dev:")
    print(f"  no TV (tv_weight=0):  {std_water_no_tv:.2f} m/s")
    print(f"  with TV (weight=5):   {std_water_tv:.2f} m/s "
          f"({(1-std_water_tv/std_water_no_tv)*100:+.1f}% reduction)")

    assert std_water_tv < std_water_no_tv * 0.7, (
        f"TV regularisation should reduce water-region noise by ≥30%. "
        f"Got no-TV std {std_water_no_tv:.2f}, TV std {std_water_tv:.2f} "
        f"(only {(1-std_water_tv/std_water_no_tv)*100:.1f}% reduction)."
    )

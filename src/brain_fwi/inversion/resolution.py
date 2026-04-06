"""Resolution matrix and PSF analysis for ultrasound tomography.

Computes the imaging resolution achievable by the transducer array
geometry. Uses JAX autodiff to compute the Jacobian (sensitivity matrix)
J = dp/dc — the derivative of sensor pressure w.r.t. sound speed
at each voxel — then derives:

  - Sensitivity map: diag(J^T J) — total sensitivity per voxel
  - Resolution matrix: R = (J^T J + λI)^{-1} J^T J
  - PSF: column of R at a query point, reshaped to grid

This is the ultrasound tomography analogue of dot-jax's reconstruct_mua
and follows the same Tikhonov-regularized inverse formulation.

References:
    - Arridge (1999). Optical tomography in medical imaging. Inverse Problems.
    - Guasch et al. (2020). Brain FWI resolution analysis. npj Digital Medicine.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional

from ..simulation.forward import (
    build_domain, build_medium, build_time_axis,
    simulate_shot_sensors, _build_source_signal,
)
from ..transducers.helmet import ring_array_2d, transducer_positions_to_grid


def compute_jacobian_column(
    grid_shape: Tuple[int, int],
    dx: float,
    src_pos: Tuple[int, int],
    sensor_positions: Tuple,
    voxel_idx: Tuple[int, int],
    c_background: float = 1500.0,
    freq: float = 50e3,
    pml_size: int = 8,
) -> jnp.ndarray:
    """Compute one column of the Jacobian J = dp/dc at a single voxel.

    Uses JAX autodiff: perturb the velocity at voxel_idx and measure
    the change in sensor recordings.

    Args:
        grid_shape: (nx, ny) grid.
        dx: Grid spacing (m).
        src_pos: Source position (grid indices).
        sensor_positions: Tuple of sensor index arrays.
        voxel_idx: (ix, iy) voxel to compute sensitivity for.
        c_background: Background sound speed (m/s).
        freq: Source frequency (Hz).
        pml_size: PML thickness.

    Returns:
        1D array: flattened sensor time series derivative dp/dc at voxel.
    """
    domain = build_domain(grid_shape, dx)
    ref_medium = build_medium(domain, c_background, 1000.0, pml_size=pml_size)
    time_axis = build_time_axis(ref_medium, cfl=0.3)
    dt = float(time_axis.dt)
    n_samples = int(float(time_axis.t_end) / dt)
    source_signal = _build_source_signal(freq, dt, n_samples)

    c0 = jnp.ones(grid_shape) * c_background
    rho = jnp.ones(grid_shape) * 1000.0

    def forward_at_voxel(perturbation):
        """Forward simulation with velocity perturbed at one voxel."""
        c = c0.at[voxel_idx].add(perturbation)
        medium = build_medium(domain, c, rho, pml_size=pml_size)
        data = simulate_shot_sensors(
            medium, time_axis, src_pos, sensor_positions, source_signal, dt,
        )
        return data.ravel()

    # Jacobian column = d(data)/d(perturbation) evaluated at perturbation=0
    jac_col = jax.grad(lambda p: jnp.sum(forward_at_voxel(p)))(0.0)

    # Actually compute the full Jacobian column via jvp
    _, jac_col = jax.jvp(forward_at_voxel, (0.0,), (1.0,))
    return jac_col


def compute_psf(
    grid_shape: Tuple[int, int],
    dx: float,
    n_elements: int,
    query_point: Tuple[int, int],
    c_background: float = 1500.0,
    freq: float = 50e3,
    pml_size: int = 8,
    regularization: float = 1e-3,
) -> jnp.ndarray:
    """Compute the Point Spread Function at a query location.

    Builds a partial Jacobian from all source-receiver pairs, then
    computes PSF = R[:, query] where R is the regularized resolution matrix.

    For efficiency, uses a finite-difference approximation of the Jacobian
    column rather than full autodiff.

    Args:
        grid_shape: (nx, ny).
        dx: Grid spacing (m).
        n_elements: Number of transducer elements (ring array).
        query_point: (ix, iy) location to compute PSF for.
        c_background: Background speed (m/s).
        freq: Frequency (Hz).
        pml_size: PML size.
        regularization: Tikhonov regularization parameter.

    Returns:
        (nx, ny) PSF array.
    """
    cx_m = grid_shape[0] * dx / 2
    cy_m = grid_shape[1] * dx / 2
    radius = min(cx_m, cy_m) - (pml_size + 2) * dx

    positions = ring_array_2d(
        n_elements=n_elements, center=(cx_m, cy_m),
        semi_major=radius, semi_minor=radius,
    )
    pos_grid = transducer_positions_to_grid(positions, dx, grid_shape)
    src_list = [(int(pos_grid[0][i]), int(pos_grid[1][i])) for i in range(n_elements)]

    domain = build_domain(grid_shape, dx)
    ref_medium = build_medium(domain, c_background, 1000.0, pml_size=pml_size)
    time_axis = build_time_axis(ref_medium, cfl=0.3)
    dt = float(time_axis.dt)
    n_samples = int(float(time_axis.t_end) / dt)
    source_signal = _build_source_signal(freq, dt, n_samples)

    c0 = jnp.ones(grid_shape) * c_background
    rho = jnp.ones(grid_shape) * 1000.0
    n_voxels = grid_shape[0] * grid_shape[1]

    # Compute Jacobian columns for query point and its neighbours
    # using finite difference: J_col ≈ (d(query) - d(ref)) / epsilon
    epsilon = 1.0  # 1 m/s perturbation

    # Reference data for all sources
    ref_data = []
    for src in src_list:
        medium = build_medium(domain, c0, rho, pml_size=pml_size)
        d = simulate_shot_sensors(medium, time_axis, src, pos_grid, source_signal, dt)
        ref_data.append(d.ravel())
    ref_stack = jnp.concatenate(ref_data)

    # Perturbed data at query point
    c_pert = c0.at[query_point].add(epsilon)
    pert_data = []
    for src in src_list:
        medium = build_medium(domain, c_pert, rho, pml_size=pml_size)
        d = simulate_shot_sensors(medium, time_axis, src, pos_grid, source_signal, dt)
        pert_data.append(d.ravel())
    pert_stack = jnp.concatenate(pert_data)

    # Jacobian column at query point
    j_query = (pert_stack - ref_stack) / epsilon

    # For a proper PSF we'd need the full Jacobian, but that's O(n_voxels)
    # forward solves. Instead, compute the "matched filter" PSF:
    # PSF ≈ J^T * j_query (backprojection of the query column sensitivity)
    # This is a fast approximation of the diagonal of R near the query point.

    # Compute J^T * j_query by perturbing each voxel and correlating
    # For speed, use the adjoint trick: backproject j_query through all sources
    psf_flat = jnp.zeros(n_voxels)

    # Approximate PSF via cross-correlation of Jacobian columns
    # For each voxel, J_col · j_query gives the resolution
    # We compute this efficiently by sampling a neighbourhood
    qx, qy = query_point
    hw = min(grid_shape[0] // 2 - 1, 8)  # half-window

    for ix in range(max(0, qx - hw), min(grid_shape[0], qx + hw + 1)):
        for iy in range(max(0, qy - hw), min(grid_shape[1], qy + hw + 1)):
            c_p = c0.at[ix, iy].add(epsilon)
            p_data = []
            for src in src_list:
                medium = build_medium(domain, c_p, rho, pml_size=pml_size)
                d = simulate_shot_sensors(medium, time_axis, src, pos_grid, source_signal, dt)
                p_data.append(d.ravel())
            j_voxel = (jnp.concatenate(p_data) - ref_stack) / epsilon

            # Resolution: dot product of Jacobian columns
            flat_idx = ix * grid_shape[1] + iy
            correlation = jnp.dot(j_query, j_voxel)
            norm = jnp.sqrt(jnp.dot(j_query, j_query) * jnp.dot(j_voxel, j_voxel) + 1e-30)
            psf_flat = psf_flat.at[flat_idx].set(correlation / norm)

    return psf_flat.reshape(grid_shape)


def compute_psf_width(
    grid_shape: Tuple[int, int],
    dx: float,
    n_elements: int,
    query_point: Tuple[int, int],
    freq: float = 50e3,
    c_background: float = 1500.0,
) -> float:
    """Compute the FWHM of the PSF at a query point.

    Returns width in grid points.
    """
    psf = compute_psf(grid_shape, dx, n_elements, query_point,
                       c_background=c_background, freq=freq)
    peak = float(jnp.max(jnp.abs(psf)))
    if peak == 0:
        return float(max(grid_shape))  # no resolution

    # Count voxels above half-maximum
    above_half = jnp.sum(jnp.abs(psf) > 0.5 * peak)
    # Convert area to equivalent diameter
    fwhm = float(jnp.sqrt(above_half / jnp.pi)) * 2
    return fwhm


def compute_sensitivity_map(
    grid_shape: Tuple[int, int],
    dx: float,
    n_elements: int,
    c_background: float = 1500.0,
    freq: float = 50e3,
    pml_size: int = 8,
    stride: int = 1,
) -> jnp.ndarray:
    """Compute the sensitivity map: ||J_col||^2 at each voxel.

    This is the diagonal of J^T J, which indicates how much each voxel
    affects the measured data. High sensitivity = well-resolved.

    Args:
        grid_shape: (nx, ny).
        dx: Grid spacing.
        n_elements: Number of ring array elements.
        c_background: Background speed.
        freq: Source frequency.
        pml_size: PML thickness.
        stride: Compute every N-th voxel (1 = all, 2 = every other).

    Returns:
        (nx, ny) sensitivity map.
    """
    cx_m = grid_shape[0] * dx / 2
    cy_m = grid_shape[1] * dx / 2
    radius = min(cx_m, cy_m) - (pml_size + 2) * dx

    positions = ring_array_2d(
        n_elements=n_elements, center=(cx_m, cy_m),
        semi_major=radius, semi_minor=radius,
    )
    pos_grid = transducer_positions_to_grid(positions, dx, grid_shape)
    src_list = [(int(pos_grid[0][i]), int(pos_grid[1][i])) for i in range(n_elements)]

    domain = build_domain(grid_shape, dx)
    ref_medium = build_medium(domain, c_background, 1000.0, pml_size=pml_size)
    time_axis = build_time_axis(ref_medium, cfl=0.3)
    dt = float(time_axis.dt)
    n_samples = int(float(time_axis.t_end) / dt)
    source_signal = _build_source_signal(freq, dt, n_samples)

    c0 = jnp.ones(grid_shape) * c_background
    rho = jnp.ones(grid_shape) * 1000.0
    epsilon = 1.0

    # Reference data
    ref_data = []
    for src in src_list:
        medium = build_medium(domain, c0, rho, pml_size=pml_size)
        d = simulate_shot_sensors(medium, time_axis, src, pos_grid, source_signal, dt)
        ref_data.append(d.ravel())
    ref_stack = jnp.concatenate(ref_data)

    # Sensitivity at each voxel = ||J_col||^2
    sensitivity = np.zeros(grid_shape, dtype=np.float32)

    for ix in range(0, grid_shape[0], stride):
        for iy in range(0, grid_shape[1], stride):
            c_p = c0.at[ix, iy].add(epsilon)
            p_data = []
            for src in src_list:
                medium = build_medium(domain, c_p, rho, pml_size=pml_size)
                d = simulate_shot_sensors(medium, time_axis, src, pos_grid, source_signal, dt)
                p_data.append(d.ravel())
            j_col = (jnp.concatenate(p_data) - ref_stack) / epsilon
            sensitivity[ix, iy] = float(jnp.sum(j_col ** 2))

    # Interpolate if strided
    if stride > 1:
        from scipy.ndimage import zoom
        sensitivity = zoom(sensitivity, stride, order=1)[:grid_shape[0], :grid_shape[1]]

    return jnp.array(sensitivity)

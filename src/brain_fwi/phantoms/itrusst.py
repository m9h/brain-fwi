"""ITRUSST transcranial-ultrasound benchmark phantoms.

Procedural phantoms reproducing the geometry and acoustic properties of
the ITRUSST benchmark series (Aubry et al. 2022, JASA 152(2):1003-1019).
These are the reference problems the transcranial-ultrasound community
uses to compare solvers and inversion methods, so having them as
importable phantoms lets brain-fwi be compared directly against the
published reference numbers.

All phantoms are generated from simple geometric primitives — no
external data is required. Each returns a dict with ``sound_speed``,
``density``, and ``attenuation`` volumes plus a ``labels`` integer map
compatible with the rest of the phantom pipeline.

Benchmarks implemented here:

  - **BM1** (water-only): homogeneous water box. Smoke test for any
    forward/inverse solver.
  - **BM2** (single-layer skull): water bath with a single-layer
    cortical-bone plate — isolates the effect of a uniform skull on
    transmission.
  - **BM3** (three-layer skull plate): cortical / diploe / cortical
    sandwich — the spec our synthetic head already targets.
  - **BM4** (curved three-layer skull plate): BM3 with a spherical
    curvature matching a typical adult calvarium — tests solver handling
    of a curved bone--water interface.

The full-head benchmarks BM5–BM7 in Aubry 2022 require a specific
MRI/CT-derived skull; for those use the MIDA v1.0 loader
(``phantoms/mida.py``) or the synthetic three-layer head
(``phantoms/synthetic.py``).
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np

# ITRUSST BM3 acoustic property values (Aubry 2022, Table III).
#
# Units: (c [m/s], rho [kg/m^3], alpha [dB/cm/MHz])
_WATER = (1500.0, 1000.0, 0.0)
_CORTICAL = (2800.0, 1850.0, 4.0)
_TRABECULAR = (2300.0, 1700.0, 8.0)
_BRAIN = (1560.0, 1040.0, 0.6)


def _props_volume(
    labels: np.ndarray,
    label_map: Dict[int, Tuple[float, float, float]],
) -> Dict[str, jnp.ndarray]:
    """Map a label volume through an explicit (c, rho, alpha) table."""
    n_max = max(label_map.keys()) + 1
    c_lookup = np.zeros(n_max, dtype=np.float32)
    rho_lookup = np.zeros(n_max, dtype=np.float32)
    alpha_lookup = np.zeros(n_max, dtype=np.float32)
    for lab, (c, rho, alpha) in label_map.items():
        c_lookup[lab] = c
        rho_lookup[lab] = rho
        alpha_lookup[lab] = alpha
    safe = np.clip(labels, 0, n_max - 1).astype(np.int32)
    return {
        "sound_speed": jnp.asarray(c_lookup[safe]),
        "density": jnp.asarray(rho_lookup[safe]),
        "attenuation": jnp.asarray(alpha_lookup[safe]),
        "labels": jnp.asarray(labels.astype(np.int32)),
    }


def make_bm1_water_box(
    grid_shape: Tuple[int, int, int],
) -> Dict[str, jnp.ndarray]:
    """ITRUSST BM1: homogeneous water box.

    Returns ``sound_speed=1500`` everywhere — the analytic baseline for
    any transcranial-ultrasound forward solver.
    """
    labels = np.zeros(grid_shape, dtype=np.int32)  # 0 = water
    return _props_volume(labels, {0: _WATER})


def make_bm2_single_layer_plate(
    grid_shape: Tuple[int, int, int],
    dx: float,
    plate_thickness_m: float = 0.007,
    plate_normal_axis: int = 2,
) -> Dict[str, jnp.ndarray]:
    """ITRUSST BM2: water with a single-layer uniform cortical-bone plate.

    The plate is a slab of cortical bone at the centre of the grid,
    normal to ``plate_normal_axis``, with thickness ``plate_thickness_m``.
    Everything else is water.

    Args:
        grid_shape: ``(nx, ny, nz)``.
        dx: Grid spacing (m).
        plate_thickness_m: Plate thickness in metres. Default 7 mm
            matches adult calvarium.
        plate_normal_axis: Which axis is normal to the plate (0/1/2).
    """
    labels = np.zeros(grid_shape, dtype=np.int32)  # 0 = water
    thickness_vox = int(round(plate_thickness_m / dx))
    mid = grid_shape[plate_normal_axis] // 2
    lo = mid - thickness_vox // 2
    hi = lo + thickness_vox
    slicer = [slice(None)] * 3
    slicer[plate_normal_axis] = slice(lo, hi)
    labels[tuple(slicer)] = 1  # 1 = cortical bone

    return _props_volume(labels, {0: _WATER, 1: _CORTICAL})


def make_bm3_three_layer_plate(
    grid_shape: Tuple[int, int, int],
    dx: float,
    outer_thickness_m: float = 0.002,
    diploe_thickness_m: float = 0.003,
    inner_thickness_m: float = 0.002,
    plate_normal_axis: int = 2,
) -> Dict[str, jnp.ndarray]:
    """ITRUSST BM3: water with a three-layer cortical/diploe/cortical plate.

    Default thicknesses match Aubry 2022 BM3: 2 mm outer cortical table,
    3 mm diploe, 2 mm inner cortical table. Total 7 mm.
    """
    labels = np.zeros(grid_shape, dtype=np.int32)
    total_vox = int(round((outer_thickness_m + diploe_thickness_m + inner_thickness_m) / dx))
    outer_vox = int(round(outer_thickness_m / dx))
    diploe_vox = int(round(diploe_thickness_m / dx))
    inner_vox = total_vox - outer_vox - diploe_vox  # absorbs rounding drift

    mid = grid_shape[plate_normal_axis] // 2
    lo = mid - total_vox // 2

    def _set(start: int, n: int, lab: int) -> None:
        slicer = [slice(None)] * 3
        slicer[plate_normal_axis] = slice(start, start + n)
        labels[tuple(slicer)] = lab

    _set(lo, outer_vox, 1)                            # outer cortical table
    _set(lo + outer_vox, diploe_vox, 2)               # diploe / trabecular
    _set(lo + outer_vox + diploe_vox, inner_vox, 1)   # inner cortical table

    return _props_volume(
        labels,
        {0: _WATER, 1: _CORTICAL, 2: _TRABECULAR},
    )


def make_bm4_curved_three_layer_plate(
    grid_shape: Tuple[int, int, int],
    dx: float,
    skull_radius_m: float = 0.085,
    outer_thickness_m: float = 0.002,
    diploe_thickness_m: float = 0.003,
    inner_thickness_m: float = 0.002,
    centre_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.085),
) -> Dict[str, jnp.ndarray]:
    """ITRUSST BM4: three-layer skull plate with a realistic spherical curvature.

    A spherical shell approximating an adult calvarium (radius ~85 mm by
    default). Layers have the same thicknesses as BM3. Tests the solver
    with a curved bone--water interface.

    Args:
        grid_shape: ``(nx, ny, nz)``.
        dx: Grid spacing (m).
        skull_radius_m: Outer radius of the skull sphere (m).
        outer_thickness_m, diploe_thickness_m, inner_thickness_m: layer
            thicknesses.
        centre_offset_m: Sphere centre relative to the grid centre (m).
            Default places the centre below the grid so only the top
            of the sphere intersects the volume — mimicking the standard
            benchmark geometry (transducer above, curved calvarium
            across the middle of the domain).
    """
    nx, ny, nz = grid_shape
    cx = nx / 2 + centre_offset_m[0] / dx
    cy = ny / 2 + centre_offset_m[1] / dx
    cz = nz / 2 + centre_offset_m[2] / dx

    x, y, z = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) * dx

    r_outer = skull_radius_m
    r_diploe = r_outer - outer_thickness_m
    r_inner = r_diploe - diploe_thickness_m
    r_stop = r_inner - inner_thickness_m

    labels = np.zeros(grid_shape, dtype=np.int32)  # 0 = water
    labels = np.where((r <= r_outer) & (r > r_diploe), 1, labels)  # outer cortical
    labels = np.where((r <= r_diploe) & (r > r_inner), 2, labels)  # diploe
    labels = np.where((r <= r_inner) & (r > r_stop), 1, labels)    # inner cortical

    return _props_volume(
        labels,
        {0: _WATER, 1: _CORTICAL, 2: _TRABECULAR},
    )

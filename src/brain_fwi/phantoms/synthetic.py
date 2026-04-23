"""Synthetic head phantom with three-layer ITRUSST-BM3 skull.

Produces an ellipsoidal head label volume with:

  - scalp (~3 mm)
  - outer cortical table (~2 mm)
  - diploe / trabecular (~3 mm)
  - inner cortical table (~2 mm)
  - CSF (~2 mm)
  - cortical grey-matter ribbon (~4 mm)
  - white-matter interior
  - optional lateral ventricles (two ellipsoids inside WM)
  - optional haemorrhagic lesion (small blood-filled sphere)

Labels follow the BrainWeb convention and compose with
``brain_fwi.phantoms.properties.map_labels_to_all`` and
``phantoms.augment.jittered_properties`` for canonical / jittered
acoustic property assignment.

This module is the canonical synthetic-phantom source for both the
FWI runner scripts (``run_full_usct.py``) and the Phase-0 dataset
generator (``scripts/gen_phase0.py``). Values chosen to match the
ITRUSST benchmark (Aubry et al. 2022, JASA 152(2):1003-1019), which
specifies the three-layer skull structure.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# BrainWeb-style label codes (see phantoms/properties.py for full table).
SCALP = 6
CORTICAL_BONE = 7
TRABECULAR_BONE = 11
CSF = 1
GREY_MATTER = 2
WHITE_MATTER = 3
BLOOD = 8


def make_three_layer_head(
    grid_shape: Tuple[int, int, int],
    dx: float,
    add_ventricles: bool = True,
    add_lesion: bool = True,
    lesion_offset_m: Tuple[float, float, float] = (-0.02, -0.01, 0.0),
    lesion_radius_m: float = 0.005,
) -> np.ndarray:
    """Build a three-layer head label volume.

    Args:
        grid_shape: ``(nx, ny, nz)``.
        dx: Grid spacing in metres.
        add_ventricles: If True, add two lateral ventricles filled with CSF.
        add_lesion: If True, add a small blood-filled sphere inside cortex.
        lesion_offset_m: Offset of lesion centre from head centre (m).
        lesion_radius_m: Lesion radius (m).

    Returns:
        ``(nx, ny, nz)`` int32 label volume.
    """
    nx, ny, nz = grid_shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    head_a = min(0.095 / dx, cx - 3)
    head_b = min(0.075 / dx, cy - 3)
    head_c = min(0.090 / dx, cz - 3)

    x, y, z = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    r = np.sqrt(
        ((x - cx) / head_a) ** 2
        + ((y - cy) / head_b) ** 2
        + ((z - cz) / head_c) ** 2
    )

    # Layer thicknesses (normalised by head semi-axis).
    scalp_t = 0.003 / (head_a * dx)
    outer_bone_t = 0.002 / (head_a * dx)
    diploe_t = 0.003 / (head_a * dx)
    inner_bone_t = 0.002 / (head_a * dx)
    csf_t = 0.002 / (head_a * dx)
    cortex_t = 0.004 / (head_a * dx)

    r_scalp = 1.0
    r_outer_bone = r_scalp - scalp_t
    r_diploe = r_outer_bone - outer_bone_t
    r_inner_bone = r_diploe - diploe_t
    r_csf = r_inner_bone - inner_bone_t
    r_csf_i = r_csf - csf_t
    r_cortex_i = r_csf_i - cortex_t

    labels = np.zeros(grid_shape, dtype=np.int32)
    labels = np.where(r <= r_scalp, SCALP, labels)
    labels = np.where(r <= r_outer_bone, CORTICAL_BONE, labels)
    labels = np.where(r <= r_diploe, TRABECULAR_BONE, labels)
    labels = np.where(r <= r_inner_bone, CORTICAL_BONE, labels)
    labels = np.where(r <= r_csf, CSF, labels)
    labels = np.where(r <= r_csf_i, GREY_MATTER, labels)
    labels = np.where(r <= r_cortex_i, WHITE_MATTER, labels)

    if add_ventricles:
        for y_off in (-0.015 / dx, 0.015 / dx):
            vr = np.sqrt(
                ((x - cx) / (0.010 / dx)) ** 2
                + ((y - cy - y_off) / (0.008 / dx)) ** 2
                + ((z - cz + 0.005 / dx) / (0.025 / dx)) ** 2
            )
            labels = np.where((vr <= 1.0) & (r <= r_cortex_i), CSF, labels)

    if add_lesion:
        lx, ly, lz = lesion_offset_m
        lr_ = np.sqrt(
            ((x - cx - lx / dx) / (lesion_radius_m / dx)) ** 2
            + ((y - cy - ly / dx) / (lesion_radius_m / dx)) ** 2
            + ((z - cz - lz / dx) / (lesion_radius_m / dx)) ** 2
        )
        labels = np.where((lr_ <= 1.0) & (r <= r_cortex_i), BLOOD, labels)

    return labels

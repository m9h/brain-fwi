"""Anatomical augmentation for Phase-0 dataset generation.

Two independent augmentations, intended to be composed per sample:

  - ``jittered_properties`` — draws one multiplicative perturbation
    per tissue from a Gaussian and applies it to the canonical
    acoustic-property table. Models inter-subject biological
    variability (e.g., skull density ~±8%) while preserving anatomy.

  - ``random_deformation_warp`` — generates a smooth random
    displacement field and applies it to a label volume with
    nearest-neighbor interpolation. Models registration / inter-
    subject shape variability without breaking tissue topology.

Both functions run on CPU with NumPy/SciPy because augmentation
happens once per sample and is dwarfed by the j-Wave forward cost.
"""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .properties import (
    TISSUE_PROPERTIES,
    _MAX_LABEL,
)


# Per-tissue *fractional* standard deviations for property jitter.
# Derived from inter-subject variability in Aubry 2022 (skull), Duck 1990
# (soft tissues), and Mast 2000 (blood). Air is held exactly at nominal.
# Format: label -> (sigma_c, sigma_rho, sigma_alpha).
_DEFAULT_FRAC_STD: Dict[int, Tuple[float, float, float]] = {
    0:  (0.00, 0.00, 0.00),   # air
    1:  (0.01, 0.01, 0.20),   # CSF
    2:  (0.02, 0.02, 0.20),   # grey matter
    3:  (0.02, 0.02, 0.20),   # white matter
    4:  (0.04, 0.03, 0.30),   # fat
    5:  (0.03, 0.03, 0.25),   # muscle
    6:  (0.03, 0.03, 0.25),   # muscle/skin
    7:  (0.07, 0.07, 0.30),   # skull cortical — high inter-subject
    8:  (0.02, 0.02, 0.20),   # blood
    9:  (0.04, 0.03, 0.30),   # connective tissue
    10: (0.03, 0.03, 0.25),   # dura
    11: (0.10, 0.10, 0.40),   # skull trabecular — highest variability
}


def jittered_properties(
    labels: jnp.ndarray,
    key: jax.Array,
    intensity: float = 1.0,
    fractional_std: Optional[Dict[int, Tuple[float, float, float]]] = None,
) -> Dict[str, jnp.ndarray]:
    """Return acoustic-property volumes with per-tissue Gaussian jitter.

    One multiplier is drawn per tissue per property (not per voxel), so
    the spatial structure is preserved. Negative draws are clipped to
    tiny positive values to guarantee physically admissible properties.

    Args:
        labels: ``(Z, Y, X)`` integer tissue labels.
        key: JAX PRNG key (split internally).
        intensity: Scalar scaling of all sigmas (0 = no jitter).
        fractional_std: Override table. Keys are labels, values are
            ``(sigma_c, sigma_rho, sigma_alpha)`` fractional stddevs.

    Returns:
        ``{"sound_speed", "density", "attenuation"}`` — each an array
        matching ``labels.shape`` (dtype float32). Also includes
        ``"multipliers"``: ``(n_labels, 3)`` per-tissue draw log for
        provenance.
    """
    table = dict(_DEFAULT_FRAC_STD if fractional_std is None else fractional_std)

    n_labels = _MAX_LABEL + 1
    key_c, key_rho, key_alpha = jr.split(key, 3)

    sig_c = np.array([intensity * table.get(i, (0, 0, 0))[0] for i in range(n_labels)], dtype=np.float32)
    sig_rho = np.array([intensity * table.get(i, (0, 0, 0))[1] for i in range(n_labels)], dtype=np.float32)
    sig_alpha = np.array([intensity * table.get(i, (0, 0, 0))[2] for i in range(n_labels)], dtype=np.float32)

    mult_c = 1.0 + jnp.array(sig_c) * jr.normal(key_c, (n_labels,))
    mult_rho = 1.0 + jnp.array(sig_rho) * jr.normal(key_rho, (n_labels,))
    mult_alpha = 1.0 + jnp.array(sig_alpha) * jr.normal(key_alpha, (n_labels,))

    mult_c = jnp.clip(mult_c, 0.5, 1.5)
    mult_rho = jnp.clip(mult_rho, 0.5, 1.5)
    mult_alpha = jnp.clip(mult_alpha, 0.1, 3.0)

    base_c = jnp.array([TISSUE_PROPERTIES.get(i, (0.0, 0.0, 0.0))[0] for i in range(n_labels)], dtype=jnp.float32)
    base_rho = jnp.array([TISSUE_PROPERTIES.get(i, (0.0, 0.0, 0.0))[1] for i in range(n_labels)], dtype=jnp.float32)
    base_alpha = jnp.array([TISSUE_PROPERTIES.get(i, (0.0, 0.0, 0.0))[2] for i in range(n_labels)], dtype=jnp.float32)

    c_lookup = base_c * mult_c
    rho_lookup = base_rho * mult_rho
    alpha_lookup = base_alpha * mult_alpha

    safe_labels = jnp.clip(labels, 0, _MAX_LABEL).astype(jnp.int32)
    multipliers = jnp.stack([mult_c, mult_rho, mult_alpha], axis=-1)

    return {
        "sound_speed": c_lookup[safe_labels],
        "density": rho_lookup[safe_labels],
        "attenuation": alpha_lookup[safe_labels],
        "multipliers": multipliers,
    }


def random_deformation_warp(
    labels: np.ndarray,
    rng: np.random.Generator,
    max_displacement_voxels: float = 3.0,
    smoothness_voxels: float = 10.0,
) -> np.ndarray:
    """Apply a smooth random displacement field to a label volume.

    Builds a random Gaussian-smoothed 3D displacement field, scales it
    to ``max_displacement_voxels`` peak, and warps ``labels`` with
    nearest-neighbor interpolation (so label values are preserved
    exactly — no interpolated / fractional labels).

    Args:
        labels: ``(Z, Y, X)`` integer label volume.
        rng: NumPy random generator.
        max_displacement_voxels: peak displacement magnitude.
        smoothness_voxels: Gaussian-smoothing sigma applied to the
            random field. Larger → smoother, lower-freq deformation.

    Returns:
        Warped ``labels`` volume, same shape and dtype.
    """
    if labels.ndim != 3:
        raise ValueError(f"expected 3D label volume, got shape {labels.shape}")

    shape = labels.shape
    disp = rng.standard_normal((3,) + shape).astype(np.float32)
    for i in range(3):
        disp[i] = gaussian_filter(disp[i], sigma=smoothness_voxels, mode="nearest")

    peak = float(np.max(np.abs(disp)))
    if peak > 0:
        disp *= max_displacement_voxels / peak

    z, y, x = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([z + disp[0], y + disp[1], x + disp[2]], axis=0)

    warped = map_coordinates(
        labels.astype(np.float32),
        coords,
        order=0,
        mode="nearest",
    )
    return warped.astype(labels.dtype)

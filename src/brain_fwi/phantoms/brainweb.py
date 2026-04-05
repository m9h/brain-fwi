"""BrainWeb tissue phantom loading for FWI.

Downloads and processes the BrainWeb 20 Normal Anatomical Models
(Collins et al. 1998, McConnell Brain Imaging Centre, McGill).

Provides:
  - load_brainweb_phantom(): Full 3D 181x217x181 tissue label volume
  - load_brainweb_slice(): Single 2D axial/coronal/sagittal slice
  - make_synthetic_head(): Procedural head phantom (no download needed)

BrainWeb download uses the brainweb-dl package. If unavailable, falls
back to the synthetic phantom generator.
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Optional, Tuple

from .properties import map_labels_to_all


def load_brainweb_phantom(
    subject: int = 4,
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    """Load a BrainWeb discrete tissue phantom (181x217x181, 1mm iso).

    Args:
        subject: BrainWeb subject ID (0-19). Default 4 = Colin27-like.
        cache_dir: Cache directory. Defaults to ~/.cache/brainweb.

    Returns:
        (181, 217, 181) int32 array of tissue labels (0-11).
    """
    try:
        import brainweb_dl
    except ImportError:
        raise ImportError(
            "brainweb-dl required for BrainWeb loading. "
            "Install with: uv add brainweb-dl"
        )

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "brainweb"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # brainweb-dl returns the tissue label volume directly
    phantom = brainweb_dl.get_mri(
        subject_id=subject,
        contrast="crisp",  # discrete tissue labels
    )
    # Ensure integer labels
    labels = np.round(phantom).astype(np.int32)
    return labels


def load_brainweb_slice(
    axis: str = "axial",
    slice_idx: Optional[int] = None,
    subject: int = 4,
    cache_dir: Optional[Path] = None,
    pad_to_square: bool = True,
) -> Tuple[np.ndarray, float]:
    """Load a single 2D slice from the BrainWeb phantom.

    Args:
        axis: 'axial' (z), 'coronal' (y), or 'sagittal' (x).
        slice_idx: Slice index. None = middle slice.
        subject: BrainWeb subject ID.
        cache_dir: Cache directory.
        pad_to_square: Pad to square dimensions for j-Wave.

    Returns:
        (labels_2d, dx_mm) where labels_2d is int32 and dx_mm = 1.0.
    """
    vol = load_brainweb_phantom(subject=subject, cache_dir=cache_dir)

    if axis == "axial":
        idx = slice_idx if slice_idx is not None else vol.shape[2] // 2
        sl = vol[:, :, idx]
    elif axis == "coronal":
        idx = slice_idx if slice_idx is not None else vol.shape[1] // 2
        sl = vol[:, idx, :]
    elif axis == "sagittal":
        idx = slice_idx if slice_idx is not None else vol.shape[0] // 2
        sl = vol[idx, :, :]
    else:
        raise ValueError(f"axis must be 'axial', 'coronal', or 'sagittal', got {axis!r}")

    if pad_to_square:
        n = max(sl.shape)
        padded = np.zeros((n, n), dtype=np.int32)
        y0 = (n - sl.shape[0]) // 2
        x0 = (n - sl.shape[1]) // 2
        padded[y0:y0 + sl.shape[0], x0:x0 + sl.shape[1]] = sl
        sl = padded

    return sl, 1.0  # BrainWeb is 1mm isotropic


def make_synthetic_head(
    grid_shape: Tuple[int, int] = (256, 256),
    dx: float = 0.001,
    skull_thickness: float = 0.007,
    scalp_thickness: float = 0.003,
    csf_thickness: float = 0.002,
) -> Tuple[jnp.ndarray, Dict]:
    """Generate a synthetic 2D head phantom (no download required).

    Creates concentric elliptical layers: scalp, skull, CSF, brain.
    Useful for quick testing and validation.

    Args:
        grid_shape: (nx, ny) grid dimensions.
        dx: Grid spacing in metres.
        skull_thickness: Skull layer thickness (m).
        scalp_thickness: Scalp thickness (m).
        csf_thickness: CSF gap thickness (m).

    Returns:
        (labels, properties) where labels is (nx, ny) int32 and
        properties is a dict with 'sound_speed', 'density', 'attenuation'.
    """
    nx, ny = grid_shape
    cx, cy = nx // 2, ny // 2

    # Head dimensions (semi-axes in grid units)
    # Approximate adult head: 19cm AP x 15cm LR
    head_a = 0.095 / dx  # AP semi-axis
    head_b = 0.075 / dx  # LR semi-axis

    # Coordinate grids
    y, x = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx))
    # Normalized elliptical distance from centre
    r = jnp.sqrt(((x - cx) / head_a) ** 2 + ((y - cy) / head_b) ** 2)

    # Layer boundaries (from outside in)
    scalp_outer = 1.0
    skull_outer = 1.0 - scalp_thickness / (head_a * dx)
    skull_inner = skull_outer - skull_thickness / (head_a * dx)
    csf_outer = skull_inner
    csf_inner = csf_outer - csf_thickness / (head_a * dx)

    # Assign labels (BrainWeb convention)
    labels = jnp.zeros(grid_shape, dtype=jnp.int32)  # background (air)
    labels = jnp.where(r <= scalp_outer, 6, labels)   # scalp
    labels = jnp.where(r <= skull_outer, 7, labels)    # skull
    labels = jnp.where(r <= skull_inner, 1, labels)    # CSF
    labels = jnp.where(r <= csf_inner, 2, labels)      # grey matter

    # Simple brain structure: WM core
    wm_a = 0.5 * csf_inner
    wm_r = jnp.sqrt(((x - cx) / (wm_a * head_a / csf_inner)) ** 2 +
                     ((y - cy) / (wm_a * head_b / csf_inner)) ** 2)
    labels = jnp.where((wm_r <= 1.0) & (r <= csf_inner), 3, labels)

    # Add ventricles (lateral ventricles as small ellipses)
    for offset in [-0.015 / dx, 0.015 / dx]:
        vent_r = jnp.sqrt(((x - cx - offset) / (0.012 / dx)) ** 2 +
                          ((y - cy) / (0.025 / dx)) ** 2)
        labels = jnp.where((vent_r <= 1.0) & (r <= csf_inner), 1, labels)

    properties = map_labels_to_all(labels)
    return labels, properties



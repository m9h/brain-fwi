"""TFUScapes dataset loader and CT-to-acoustic mapping.

TFUScapes (CAMMA-public, arXiv:2505.12998) is a large-scale dataset of
3D transcranial focused-ultrasound simulations produced with k-Wave.
Each subject sample contains:

  - ``ct``:        ``(256, 256, 256) float32`` CT volume in Hounsfield
                   Units (background zeroed rather than -1000)
  - ``pmap``:      ``(256, 256, 256) float32`` steady-state pressure map
  - ``tr_coords``: ``(N, 3) int64`` transducer element coordinates on
                   the CT grid

For FWI purposes we need only the CT volume and transducer coordinates;
the pressure map is the reference output of the paper's forward model
and is useful as a validation target when we later train our own
surrogate.

The CT volume is mapped to (sound_speed, density, attenuation) via a
piecewise-linear interpolation between ITRUSST-BM3 anchor points, so
TFUScapes samples compose with the rest of the phantom pipeline.

Dataset: https://github.com/CAMMA-public/TFUScapes
Paper: Sharma et al. (2025), "A Skull-Adaptive Framework for AI-Based
3D Transcranial Focused Ultrasound Simulation." arXiv:2505.12998
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# CT → acoustic mapping
# ---------------------------------------------------------------------------
# Piecewise-linear interpolation between these anchor points (HU, c, rho,
# alpha). Values derived from ITRUSST BM3 (Aubry et al. 2022) and Pichardo
# et al. (2011)'s porosity-based skull model:
#
#   0    HU -> water / soft tissue (c=1500, rho=1000, alpha=0.002)
#   500  HU -> trabecular bone edge (c=1800, rho=1200, alpha=2.0)
#   1000 HU -> trabecular bone      (c=2300, rho=1700, alpha=8.0)
#   1500 HU -> cortical bone        (c=2800, rho=1850, alpha=4.0)
#
# Note: TFUScapes CT volumes zero the background rather than setting it to
# the standard -1000 HU, so HU <= 0 is treated as soft tissue here. Callers
# that need a water coupling medium around the head should pass
# ``background="water"`` to replace the zero-HU region with water explicitly.

_HU_ANCHORS = np.array([0.0, 500.0, 1000.0, 1500.0, 3000.0], dtype=np.float32)
_C_ANCHORS = np.array([1500.0, 1800.0, 2300.0, 2800.0, 2800.0], dtype=np.float32)
_RHO_ANCHORS = np.array([1000.0, 1200.0, 1700.0, 1850.0, 1850.0], dtype=np.float32)
_ALPHA_ANCHORS = np.array([0.002, 2.0, 8.0, 4.0, 4.0], dtype=np.float32)


def ct_to_acoustic(
    ct_hu: np.ndarray,
    background: str = "water",
) -> Dict[str, jnp.ndarray]:
    """Map a CT volume in Hounsfield Units to acoustic properties.

    Args:
        ct_hu: CT volume in HU. Any shape.
        background: Handling of the ambient medium around the head.
            ``"water"`` (default) replaces HU=0 voxels with water
            (c=1500, rho=1000). ``"air"`` replaces them with air
            (c=343, rho=1.225). ``"interp"`` treats HU=0 as soft tissue
            per the anchor table.

    Returns:
        ``{"sound_speed": c, "density": rho, "attenuation": alpha}`` —
        each a JAX array matching ``ct_hu.shape``.
    """
    ct = np.asarray(ct_hu, dtype=np.float32)

    c = np.interp(ct, _HU_ANCHORS, _C_ANCHORS).astype(np.float32)
    rho = np.interp(ct, _HU_ANCHORS, _RHO_ANCHORS).astype(np.float32)
    alpha = np.interp(ct, _HU_ANCHORS, _ALPHA_ANCHORS).astype(np.float32)

    if background == "water":
        bg = ct == 0.0
        c = np.where(bg, 1500.0, c)
        rho = np.where(bg, 1000.0, rho)
        alpha = np.where(bg, 0.0, alpha)
    elif background == "air":
        bg = ct == 0.0
        c = np.where(bg, 343.0, c)
        rho = np.where(bg, 1.225, rho)
        alpha = np.where(bg, 0.0, alpha)
    elif background != "interp":
        raise ValueError(
            f"background={background!r}; expected 'water', 'air', or 'interp'."
        )

    return {
        "sound_speed": jnp.asarray(c),
        "density": jnp.asarray(rho),
        "attenuation": jnp.asarray(alpha),
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_tfuscapes_sample(
    path: Path,
    background: str = "water",
) -> Dict[str, object]:
    """Load a single TFUScapes ``.npz`` sample.

    Args:
        path: Path to the .npz file (e.g. ``A00028185/exp_0.npz``).
        background: Passed through to :func:`ct_to_acoustic`.

    Returns:
        Dict with keys:
          - ``ct``: original HU volume (numpy float32)
          - ``sound_speed``, ``density``, ``attenuation``: JAX arrays (m/s,
            kg/m^3, dB/cm/MHz)
          - ``transducer_positions_grid``: ``(N, 3) int32`` element
            coordinates on the CT grid
          - ``reference_pressure_map``: ``(256, 256, 256) float32``, the
            k-Wave steady-state pressure field from the paper
          - ``subject_id``: parent directory name
          - ``experiment_id``: filename stem
    """
    path = Path(path)
    data = np.load(str(path))

    ct = np.asarray(data["ct"], dtype=np.float32)
    tr_coords = np.asarray(data["tr_coords"], dtype=np.int32)
    pmap = np.asarray(data["pmap"], dtype=np.float32)

    acoustic = ct_to_acoustic(ct, background=background)

    return {
        "ct": ct,
        "sound_speed": acoustic["sound_speed"],
        "density": acoustic["density"],
        "attenuation": acoustic["attenuation"],
        "transducer_positions_grid": tr_coords,
        "reference_pressure_map": pmap,
        "subject_id": path.parent.name,
        "experiment_id": path.stem,
    }


def discover_tfuscapes_samples(root: Path) -> list[Path]:
    """Enumerate all ``.npz`` samples under a TFUScapes data root.

    Expected layout (per the HuggingFace release)::

        root/
          data/
            A00028185/exp_0.npz
            A00028185/exp_1.npz
            ...
            A00034856/exp_0.npz

    Accepts either the dataset root (containing ``data/``) or the
    ``data/`` directory directly.

    Args:
        root: Dataset root.

    Returns:
        Sorted list of sample paths.
    """
    root = Path(root)
    if (root / "data").is_dir():
        root = root / "data"
    return sorted(root.glob("*/exp_*.npz"))


def head_mask_from_ct(ct_hu: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Boolean mask of voxels inside the head (CT > threshold).

    Useful for restricting FWI updates to the head proper (exclude the
    water coupling region the same way ``(labels > 0)`` does for the
    synthetic phantom).
    """
    return np.asarray(ct_hu, dtype=np.float32) > float(threshold)

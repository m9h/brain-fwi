"""SCI Institute head model loader and acoustic property mapping.

The SCI (Scientific Computing and Imaging) head model is an 8-tissue
head/brain segmentation published by the University of Utah SCI
Institute, widely used for EEG / EIT / tDCS forward modelling. The
segmentation is distributed as an NRRD file produced by Seg3D.

Default label convention (neurojax-compatible, matches Seg3D export):

    1 = Scalp
    2 = Skull
    3 = Sinus / Air cavities
    4 = CSF
    5 = Gray matter
    6 = White matter
    7 = Eyes
    8 = Background (exterior)

Acoustic properties follow the ITRUSST BM3 benchmark (Aubry et al.
2022) where applicable, with eyes treated as aqueous humour (~water)
per Duck (1990).

The NRRD loader is a thin wrapper over ``pynrrd``; install it with
``uv add pynrrd`` (already in the brain-fwi environment).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Default 8-tissue label mapping
# ---------------------------------------------------------------------------

SCI_LABEL_NAMES: Dict[int, str] = {
    1: "Scalp",
    2: "Skull",
    3: "Sinus",
    4: "CSF",
    5: "Gray Matter",
    6: "White Matter",
    7: "Eyes",
    8: "Background",
}


# Acoustic properties per SCI tissue (c [m/s], rho [kg/m^3], alpha [dB/cm/MHz]).
# Water coupling is the standard USCT setup, so Sinus + Background map to
# water rather than air (see ct_to_acoustic in tfuscapes.py for the same
# pattern).
SCI_ACOUSTIC_PROPERTIES: Dict[int, Dict[str, float]] = {
    1: {"sound_speed": 1610.0, "density": 1090.0, "attenuation": 0.8},   # Scalp
    2: {"sound_speed": 2800.0, "density": 1850.0, "attenuation": 4.0},   # Skull (cortical)
    3: {"sound_speed": 1500.0, "density": 1000.0, "attenuation": 0.0},   # Sinus -> water coupling
    4: {"sound_speed": 1500.0, "density": 1007.0, "attenuation": 0.002}, # CSF
    5: {"sound_speed": 1560.0, "density": 1040.0, "attenuation": 0.6},   # Gray matter
    6: {"sound_speed": 1560.0, "density": 1040.0, "attenuation": 0.6},   # White matter
    7: {"sound_speed": 1532.0, "density": 1005.0, "attenuation": 0.1},   # Eyes
    8: {"sound_speed": 1500.0, "density": 1000.0, "attenuation": 0.0},   # Background -> water
}


def map_sci_labels_to_acoustic(
    labels: np.ndarray,
    properties: Optional[Dict[int, Dict[str, float]]] = None,
) -> Dict[str, jnp.ndarray]:
    """Map SCI tissue labels to (sound_speed, density, attenuation).

    Args:
        labels: Integer label volume, any shape. Values 1–8 (or a
            subset). Unknown labels are treated as water coupling.
        properties: Override table ``{label: {"sound_speed", "density",
            "attenuation"}}``. Merged on top of
            :data:`SCI_ACOUSTIC_PROPERTIES`.

    Returns:
        Dict with ``sound_speed``, ``density``, ``attenuation``; each a
        JAX array matching ``labels.shape``.
    """
    table = dict(SCI_ACOUSTIC_PROPERTIES)
    if properties is not None:
        table.update({k: dict(v) for k, v in properties.items()})

    n_max = max(table.keys()) + 1
    c_lookup = np.full(n_max, 1500.0, dtype=np.float32)
    rho_lookup = np.full(n_max, 1000.0, dtype=np.float32)
    alpha_lookup = np.zeros(n_max, dtype=np.float32)
    for lab, props in table.items():
        c_lookup[lab] = props["sound_speed"]
        rho_lookup[lab] = props["density"]
        alpha_lookup[lab] = props["attenuation"]

    safe = np.clip(np.asarray(labels), 0, n_max - 1).astype(np.int32)
    return {
        "sound_speed": jnp.asarray(c_lookup[safe]),
        "density": jnp.asarray(rho_lookup[safe]),
        "attenuation": jnp.asarray(alpha_lookup[safe]),
    }


# ---------------------------------------------------------------------------
# NRRD loader
# ---------------------------------------------------------------------------

def load_sci_head_segmentation(path: Path) -> np.ndarray:
    """Load the SCI head model segmentation from an NRRD file.

    Args:
        path: Path to ``HeadSegmentation.nrrd`` (or equivalent).

    Returns:
        ``(nx, ny, nz) int32`` label volume.
    """
    try:
        import nrrd
    except ImportError as e:
        raise ImportError(
            "Loading the SCI head model requires pynrrd. Install with "
            "`uv add pynrrd` (already in the brain-fwi environment)."
        ) from e

    data, _header = nrrd.read(str(Path(path)))
    return data.astype(np.int32)


def load_sci_head_acoustic(
    path: Path,
    properties: Optional[Dict[int, Dict[str, float]]] = None,
) -> Dict[str, jnp.ndarray]:
    """Load SCI head segmentation and map to acoustic properties.

    Returns a dict with ``sound_speed``, ``density``, ``attenuation``,
    and ``labels``.
    """
    labels = load_sci_head_segmentation(path)
    props = map_sci_labels_to_acoustic(labels, properties=properties)
    props["labels"] = jnp.asarray(labels)
    return props

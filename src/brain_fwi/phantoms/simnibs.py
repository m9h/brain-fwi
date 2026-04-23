"""SimNIBS (CHARM) head-model loader and acoustic-property mapping.

SimNIBS 4+ uses CHARM to produce a standardised 11-tissue head
segmentation from T1/T2 MRI. The segmentation is distributed as
``tissue_labeling.nii.gz`` inside an ``m2m_<subject>/`` directory.

Label convention (from ``simnibs/segmentation/atlases/charm_atlas_mni_v1-0/
charm_atlas_mni_v1-0.ini``):

    0  = Background
    1  = White Matter
    2  = Gray Matter
    3  = CSF
    5  = Scalp (skin)
    6  = Eyes
    7  = Compact Bone (cortical skull)
    8  = Spongy Bone (trabecular / diploe)
    9  = Blood
    10 = Muscle
    11 = Air pockets

Note: label 4 is intentionally unused in the CHARM output.

Acoustic properties follow ITRUSST BM3 (Aubry et al. 2022). Compact
and spongy bone map to the cortical / trabecular values consistent
with the rest of the phantom pipeline. Air pockets are remapped to
water coupling by default (transcranial USCT is water-immersed);
pass ``use_air=True`` to preserve the real air properties instead
(useful for simulations where sinus air matters).

This loader does not require the SimNIBS Python package — it only
needs nibabel and the standard CHARM output directory layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import jax.numpy as jnp
import numpy as np


SIMNIBS_LABEL_NAMES: Dict[int, str] = {
    0: "Background",
    1: "White Matter",
    2: "Gray Matter",
    3: "CSF",
    5: "Scalp",
    6: "Eyes",
    7: "Compact Bone",
    8: "Spongy Bone",
    9: "Blood",
    10: "Muscle",
    11: "Air pockets",
}


def _default_acoustic_table(use_air: bool) -> Dict[int, Dict[str, float]]:
    air = (
        {"sound_speed": 343.0, "density": 1.225, "attenuation": 0.0}
        if use_air
        else {"sound_speed": 1500.0, "density": 1000.0, "attenuation": 0.0}
    )
    return {
        0:  {"sound_speed": 1500.0, "density": 1000.0, "attenuation": 0.0},    # Background -> water
        1:  {"sound_speed": 1560.0, "density": 1040.0, "attenuation": 0.6},    # White matter
        2:  {"sound_speed": 1560.0, "density": 1040.0, "attenuation": 0.6},    # Gray matter
        3:  {"sound_speed": 1500.0, "density": 1007.0, "attenuation": 0.002},  # CSF
        5:  {"sound_speed": 1610.0, "density": 1090.0, "attenuation": 0.8},    # Scalp
        6:  {"sound_speed": 1532.0, "density": 1005.0, "attenuation": 0.1},    # Eyes
        7:  {"sound_speed": 2800.0, "density": 1850.0, "attenuation": 4.0},    # Compact bone
        8:  {"sound_speed": 2300.0, "density": 1700.0, "attenuation": 8.0},    # Spongy bone
        9:  {"sound_speed": 1584.0, "density": 1060.0, "attenuation": 0.2},    # Blood
        10: {"sound_speed": 1547.0, "density": 1050.0, "attenuation": 1.0},    # Muscle
        11: air,                                                               # Air pockets
    }


def simnibs_acoustic_table(use_air: bool = False) -> Dict[int, Dict[str, float]]:
    """Return the default SimNIBS tissue -> acoustic property table.

    Args:
        use_air: If True, Air_pockets maps to real air (c=343).
            Default False maps it to water coupling for USCT settings.
    """
    return _default_acoustic_table(use_air)


def map_simnibs_labels_to_acoustic(
    labels: np.ndarray,
    properties: Optional[Dict[int, Dict[str, float]]] = None,
    use_air: bool = False,
) -> Dict[str, jnp.ndarray]:
    """Map CHARM labels to (sound_speed, density, attenuation)."""
    table = _default_acoustic_table(use_air)
    if properties is not None:
        for k, v in properties.items():
            table[k] = dict(v)

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
# Directory layout + NIfTI loading
# ---------------------------------------------------------------------------

_CANDIDATE_FILENAMES = (
    "tissue_labeling.nii.gz",
    "tissue_labeling_upsampled.nii.gz",
    "final_tissues.nii.gz",
    "segmentation/labeling.nii.gz",
    "label_prep/tissue_labeling_upsampled.nii.gz",
)


def find_simnibs_tissue_labeling(m2m_dir: Path) -> Path:
    """Locate the tissue_labeling NIfTI inside an ``m2m_<subject>/`` dir.

    Args:
        m2m_dir: SimNIBS CHARM output directory (e.g. ``m2m_ernie/``).

    Returns:
        Absolute path to the first matching labeling file.

    Raises:
        FileNotFoundError: if none of the expected filenames are present.
    """
    m2m_dir = Path(m2m_dir)
    for rel in _CANDIDATE_FILENAMES:
        candidate = m2m_dir / rel
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No SimNIBS tissue-labeling NIfTI found under {m2m_dir}. "
        f"Expected one of: {list(_CANDIDATE_FILENAMES)}."
    )


def load_simnibs_segmentation(m2m_or_path: Path) -> np.ndarray:
    """Load SimNIBS tissue segmentation as an integer label volume.

    Args:
        m2m_or_path: Either an ``m2m_<subject>/`` directory (this function
            finds the label file inside) or a direct path to a
            ``tissue_labeling*.nii.gz`` file.

    Returns:
        ``(nx, ny, nz) int32`` label volume.
    """
    import nibabel as nib

    path = Path(m2m_or_path)
    if path.is_dir():
        path = find_simnibs_tissue_labeling(path)
    img = nib.load(str(path))
    return np.round(img.get_fdata()).astype(np.int32)


def load_simnibs_acoustic(
    m2m_or_path: Path,
    properties: Optional[Dict[int, Dict[str, float]]] = None,
    use_air: bool = False,
) -> Dict[str, jnp.ndarray]:
    """Load SimNIBS segmentation and map to acoustic properties.

    Returns a dict with ``sound_speed``, ``density``, ``attenuation``,
    and ``labels``.
    """
    labels = load_simnibs_segmentation(m2m_or_path)
    props = map_simnibs_labels_to_acoustic(
        labels, properties=properties, use_air=use_air,
    )
    props["labels"] = jnp.asarray(labels)
    return props

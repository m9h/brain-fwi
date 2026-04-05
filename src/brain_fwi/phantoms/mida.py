"""MIDA head model loading and acoustic property mapping.

The MIDA (Multimodal Imaging-Based Detailed Anatomical) model provides
153 anatomical structures at 500um isotropic resolution. This module
maps MIDA tissue labels to acoustic properties using ITRUSST benchmark
values and literature compilations.

This fills the gap left by Sonus (neurotech-berkeley/Sonus) which
provides a pre-baked velocity HDF5 but no tissue-to-acoustic mapping
code, and only velocity (no density or attenuation).

The MIDA model requires a license from IT'IS Foundation (not open source):
    https://itis.swiss/virtual-population/regional-human-models/mida-model/
    Contact: MIDAmodel@fda.hhs.gov

For open alternatives, use BrainWeb (brainweb.py) or Colin27 with
tissue segmentation from SimNIBS CHARM/headreco.

References:
    - Iacono et al. (2015). MIDA. PLOS ONE.
    - Aubry et al. (2022). ITRUSST benchmark. JASA 152(2):1003-1019.
    - Guasch et al. (2020). Brain FWI. npj Digital Medicine 3:28.
    - Duck (1990). Physical Properties of Tissues.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Optional, Tuple
from pathlib import Path


# ---------------------------------------------------------------------------
# MIDA tissue group definitions
# ---------------------------------------------------------------------------
# MIDA has 153 labels. We group them into acoustic categories since many
# structures share similar acoustic properties. The label→group mapping
# is based on the MIDA documentation and Guasch et al. supplementary data.

MIDA_TISSUE_GROUPS: Dict[str, list] = {
    "air": [0],  # background/air
    "skin": [1, 2],  # skin, subcutaneous fat
    "fat": [3, 4, 5],  # adipose, periorbital fat, fat pads
    "muscle": list(range(6, 32)),  # 26 individual muscles
    "cortical_bone": [
        32, 33, 34,  # skull outer table, inner table, teeth
        35, 36, 37, 38, 39, 40, 41, 42,  # vertebrae, mandible, bones
    ],
    "trabecular_bone": [
        43, 44,  # diploe (skull spongy layer), bone marrow
    ],
    "cartilage": [45, 46, 47, 48],  # nasal, ear, laryngeal, etc.
    "csf": [49, 50, 51],  # CSF, ventricles, subarachnoid space
    "grey_matter": [52, 53, 54, 55, 56, 57, 58, 59, 60],  # cortex, cerebellum GM, thalamic nuclei (9 groups)
    "white_matter": [61, 62, 63, 64, 65],  # cerebral WM, cerebellar WM, corpus callosum, brainstem, tracts
    "blood_vessels": list(range(66, 90)),  # arteries, veins, sinuses (24 structures)
    "dura": [90, 91],  # dura mater, falx cerebri
    "eye": [92, 93, 94, 95, 96, 97],  # lens, cornea, vitreous, retina, sclera, optic nerve
    "nerve": list(range(98, 122)),  # 12 cranial nerve pairs (24 structures)
    "gland": [122, 123, 124, 125],  # pituitary, pineal, thyroid, salivary
    "mucosa": list(range(126, 140)),  # nasal mucosa, oral mucosa, etc.
    "connective": list(range(140, 154)),  # tendons, ligaments, fascia, etc.
}

# Build reverse lookup: label_id → group_name
_LABEL_TO_GROUP: Dict[int, str] = {}
for _group, _labels in MIDA_TISSUE_GROUPS.items():
    for _lab in _labels:
        _LABEL_TO_GROUP[_lab] = _group


# ---------------------------------------------------------------------------
# Acoustic properties per tissue group
# ---------------------------------------------------------------------------
# Sources: ITRUSST benchmark (Aubry 2022), Duck (1990), Mast (2000),
# Connor (2002), Guasch et al. (2020) supplementary Table 1.

MIDA_ACOUSTIC_PROPERTIES: Dict[str, Dict[str, float]] = {
    #                    c [m/s]    rho [kg/m³]  alpha [dB/cm/MHz]
    "air":             {"sound_speed": 343.0,  "density": 1.225,  "attenuation": 0.0},
    "skin":            {"sound_speed": 1610.0, "density": 1090.0, "attenuation": 0.8},
    "fat":             {"sound_speed": 1478.0, "density": 950.0,  "attenuation": 0.4},
    "muscle":          {"sound_speed": 1547.0, "density": 1050.0, "attenuation": 1.0},
    "cortical_bone":   {"sound_speed": 2800.0, "density": 1850.0, "attenuation": 4.0},
    "trabecular_bone": {"sound_speed": 2300.0, "density": 1700.0, "attenuation": 8.0},
    "cartilage":       {"sound_speed": 1665.0, "density": 1100.0, "attenuation": 1.0},
    "csf":             {"sound_speed": 1500.0, "density": 1007.0, "attenuation": 0.002},
    "grey_matter":     {"sound_speed": 1560.0, "density": 1040.0, "attenuation": 0.6},
    "white_matter":    {"sound_speed": 1560.0, "density": 1040.0, "attenuation": 0.6},
    "blood_vessels":   {"sound_speed": 1584.0, "density": 1060.0, "attenuation": 0.2},
    "dura":            {"sound_speed": 1560.0, "density": 1080.0, "attenuation": 0.5},
    "eye":             {"sound_speed": 1532.0, "density": 1005.0, "attenuation": 0.1},
    "nerve":           {"sound_speed": 1560.0, "density": 1040.0, "attenuation": 0.6},
    "gland":           {"sound_speed": 1560.0, "density": 1050.0, "attenuation": 0.5},
    "mucosa":          {"sound_speed": 1540.0, "density": 1050.0, "attenuation": 0.5},
    "connective":      {"sound_speed": 1520.0, "density": 1030.0, "attenuation": 0.5},
}

# Default for unknown labels
_DEFAULT_PROPS = {"sound_speed": 1500.0, "density": 1000.0, "attenuation": 0.0}

# Pre-build lookup arrays (max label → property value) for fast vectorized mapping
_MAX_MIDA_LABEL = 153
_C_MIDA = np.zeros(_MAX_MIDA_LABEL + 1, dtype=np.float32)
_RHO_MIDA = np.zeros(_MAX_MIDA_LABEL + 1, dtype=np.float32)
_ALPHA_MIDA = np.zeros(_MAX_MIDA_LABEL + 1, dtype=np.float32)

for _i in range(_MAX_MIDA_LABEL + 1):
    _group = _LABEL_TO_GROUP.get(_i, None)
    _props = MIDA_ACOUSTIC_PROPERTIES.get(_group, _DEFAULT_PROPS) if _group else _DEFAULT_PROPS
    _C_MIDA[_i] = _props["sound_speed"]
    _RHO_MIDA[_i] = _props["density"]
    _ALPHA_MIDA[_i] = _props["attenuation"]

# Label 0 (background/air) → remap to water for acoustic coupling medium
_C_MIDA[0] = 1500.0
_RHO_MIDA[0] = 1000.0
_ALPHA_MIDA[0] = 0.0

_C_MIDA_JAX = jnp.array(_C_MIDA)
_RHO_MIDA_JAX = jnp.array(_RHO_MIDA)
_ALPHA_MIDA_JAX = jnp.array(_ALPHA_MIDA)


# ---------------------------------------------------------------------------
# Mapping functions
# ---------------------------------------------------------------------------

def map_mida_labels_to_acoustic(
    labels: np.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Map MIDA tissue labels to acoustic properties.

    Args:
        labels: Integer array of any shape. Values 0-153.

    Returns:
        Dict with 'sound_speed' (m/s), 'density' (kg/m^3),
        'attenuation' (dB/cm/MHz). Same shape as labels.
    """
    safe = jnp.clip(jnp.asarray(labels), 0, _MAX_MIDA_LABEL).astype(jnp.int32)
    return {
        "sound_speed": _C_MIDA_JAX[safe],
        "density": _RHO_MIDA_JAX[safe],
        "attenuation": _ALPHA_MIDA_JAX[safe],
    }


# ---------------------------------------------------------------------------
# MIDA file loading
# ---------------------------------------------------------------------------

def load_mida_volume(
    path: Path,
    format: str = "auto",
) -> np.ndarray:
    """Load MIDA tissue label volume from file.

    Supports:
      - .mat (MATLAB, key 'tissuedistrib')
      - .nii / .nii.gz (NIfTI via nibabel)
      - .h5 / .hdf5 (HDF5, key 'labels' or 'vp' for velocity)

    Args:
        path: Path to MIDA data file.
        format: 'mat', 'nifti', 'hdf5', or 'auto' (detect from extension).

    Returns:
        Integer array of tissue labels (typically 480x480x350 at 500um).
    """
    path = Path(path)

    if format == "auto":
        suffix = path.suffix.lower()
        if suffix == ".mat":
            format = "mat"
        elif suffix in (".nii", ".gz"):
            format = "nifti"
        elif suffix in (".h5", ".hdf5"):
            format = "hdf5"
        else:
            raise ValueError(f"Cannot detect format from extension {suffix!r}")

    if format == "mat":
        from scipy.io import loadmat
        data = loadmat(str(path))
        if "tissuedistrib" in data:
            return data["tissuedistrib"].astype(np.int32)
        # Try first array-valued key
        for key, val in data.items():
            if isinstance(val, np.ndarray) and val.ndim == 3:
                return val.astype(np.int32)
        raise KeyError("No 3D array found in .mat file")

    elif format == "nifti":
        import nibabel as nib
        img = nib.load(str(path))
        return np.round(img.get_fdata()).astype(np.int32)

    elif format == "hdf5":
        import h5py
        with h5py.File(str(path), "r") as f:
            for key in ["labels", "tissuedistrib", "tissue_labels"]:
                if key in f:
                    return np.array(f[key], dtype=np.int32)
            # If 'vp' exists, this is a pre-processed velocity volume (like Sonus)
            if "vp" in f:
                return np.array(f["vp"], dtype=np.float32)
        raise KeyError("No label array found in HDF5 file")

    else:
        raise ValueError(f"Unsupported format: {format!r}")


def load_mida_acoustic(
    path: Path,
    target_shape: Optional[Tuple[int, int, int]] = None,
    target_dx: Optional[float] = None,
) -> Dict[str, jnp.ndarray]:
    """Load MIDA model and convert to acoustic properties.

    Combines load_mida_volume + map_mida_labels_to_acoustic with
    optional resampling to a target grid.

    Args:
        path: Path to MIDA data file.
        target_shape: Resample to this shape. None = native resolution.
        target_dx: Target grid spacing in m. Ignored if target_shape given.
            Native MIDA spacing is 0.5mm.

    Returns:
        Dict with 'sound_speed', 'density', 'attenuation', 'labels'.
    """
    labels = load_mida_volume(path)

    # If velocity volume (from Sonus-style HDF5), wrap directly
    if labels.dtype == np.float32 and np.max(labels) > 200:
        # This is a pre-processed velocity volume, not labels
        c = labels
        if target_shape is not None:
            c = resample_volume(c, target_shape)
        return {
            "sound_speed": jnp.array(c),
            "density": jnp.full(c.shape, 1000.0),  # no density info
            "attenuation": jnp.zeros(c.shape),
        }

    if target_shape is not None:
        labels = resample_volume(labels, target_shape, order=0)
    elif target_dx is not None:
        native_dx = 0.5e-3  # MIDA is 500um
        scale = native_dx / target_dx
        new_shape = tuple(int(s * scale) for s in labels.shape)
        labels = resample_volume(labels, new_shape, order=0)

    props = map_mida_labels_to_acoustic(labels)
    props["labels"] = jnp.array(labels)
    return props


# ---------------------------------------------------------------------------
# Volume resampling
# ---------------------------------------------------------------------------

def resample_volume(
    volume: np.ndarray,
    target_shape: Tuple[int, ...],
    order: int = 1,
) -> np.ndarray:
    """Resample a 3D volume to a target shape.

    Args:
        volume: Input array.
        target_shape: Desired output shape.
        order: Interpolation order (0=nearest for labels, 1=linear for fields).

    Returns:
        Resampled array with target_shape.
    """
    from scipy.ndimage import zoom

    if volume.shape == target_shape:
        return volume

    scale = tuple(t / s for t, s in zip(target_shape, volume.shape))
    result = zoom(volume, scale, order=order)

    # Preserve dtype for integer arrays (labels)
    if volume.dtype in (np.int32, np.int64, np.uint8):
        result = np.round(result).astype(volume.dtype)

    return result

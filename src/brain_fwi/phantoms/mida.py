"""MIDA head model loading and acoustic property mapping.

The MIDA (Multimodal Imaging-Based Detailed Anatomical) v1.0 model
provides 116 anatomical structures at 500um isotropic resolution
(grid 480 x 480 x 350). This module maps MIDA tissue labels to
acoustic properties using ITRUSST benchmark values and literature
compilations.

Label mapping is derived from `MIDA_v1.txt` (IT'IS Foundation,
distributed with the dataset). The per-label grouping below is a
hand-curated bucketing of those 116 structures into acoustic
categories that share a common (c, rho, alpha) triplet.

The MIDA model requires a license from IT'IS Foundation (not open source):
    https://itis.swiss/virtual-population/regional-human-models/mida-model/

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
# MIDA v1.0 label names (from MIDA_v1.txt, IT'IS Foundation)
# ---------------------------------------------------------------------------

MIDA_LABEL_NAMES: Dict[int, str] = {
    1: "Dura",
    2: "Cerebellum Gray Matter",
    3: "Pineal Body",
    4: "Amygdala",
    5: "Hippocampus",
    6: "CSF Ventricles",
    7: "Caudate Nucleus",
    8: "Putamen",
    9: "Cerebellum White Matter",
    10: "Brain Gray Matter",
    11: "Brainstem Midbrain",
    12: "Brain White Matter",
    13: "Spinal Cord",
    14: "Brainstem Pons",
    15: "Brainstem Medulla",
    16: "Nucleus Accumbens",
    17: "Globus Pallidus",
    18: "Optic Tract",
    19: "Hypophysis (Pituitary Gland)",
    20: "Mammillary Body",
    21: "Hypothalamus",
    22: "Commissura (Anterior)",
    23: "Commissura (Posterior)",
    24: "Blood Arteries",
    25: "Blood Veins",
    26: "Air Internal - Ethmoidal Sinus",
    27: "Air Internal - Frontal Sinus",
    28: "Air Internal - Maxillary Sinus",
    29: "Air Internal - Sphenoidal Sinus",
    30: "Air Internal - Mastoid",
    31: "Air Internal - Nasal/Pharynx",
    32: "CSF General",
    33: "Ear Cochlea",
    34: "Ear Semicircular Canals",
    35: "Ear Auricular Cartilage (Pinna)",
    36: "Mandible",
    37: "Mucosa",
    38: "Muscle (General)",
    39: "Nasal Septum (Cartilage)",
    40: "Skull",
    41: "Teeth",
    42: "Tongue",
    43: "Adipose Tissue",
    44: "Vertebra - C1 (atlas)",
    45: "Vertebra - C2 (axis)",
    46: "Vertebra - C3",
    47: "Vertebra - C4",
    48: "Vertebra - C5",
    49: "Intervertebral Discs",
    50: "Background",
    51: "Epidermis/Dermis",
    52: "Skull Diploe",
    53: "Skull Inner Table",
    54: "Skull Outer Table",
    55: "Eye Lens",
    56: "Eye Retina/Choroid/Sclera",
    57: "Eye Vitreous",
    58: "Eye Cornea",
    59: "Eye Aqueous",
    60: "Muscle - Platysma",
    61: "Tendon - Galea Aponeurotica",
    62: "Subcutaneous Adipose Tissue",
    63: "Muscle - Temporalis/Temporoparietalis",
    64: "Muscle - Occipitiofrontalis - Frontal Belly",
    65: "Muscle - Lateral Pterygoid",
    66: "Muscle - Masseter",
    67: "Muscle - Splenius Capitis",
    68: "Muscle - Sternocleidomastoid",
    69: "Muscle - Occipitiofrontalis - Occipital Belly",
    70: "Muscle - Trapezius",
    71: "Muscle - Mentalis",
    72: "Muscle - Depressor Anguli Oris",
    73: "Muscle - Depressor Labii",
    74: "Muscle - Nasalis",
    75: "Muscle - Orbicularis Oris",
    76: "Muscles - Procerus",
    77: "Muscle - Levator Labii Superioris",
    78: "Muscle - Zygomaticus Major",
    79: "Muscle - Orbicularis Oculi",
    80: "Muscle - Levator Scapulae",
    81: "Muscle - Medial Pterygoid",
    82: "Muscle - Zygomaticus Minor",
    83: "Muscles - Risorius",
    84: "Muscle - Buccinator",
    85: "Ear Auditory Canal",
    86: "Ear Pharyngotympanic Tube",
    87: "Hyoid Bone",
    88: "Submandibular Gland",
    89: "Parotid Gland",
    90: "Sublingual Gland",
    91: "Muscle - Superior Rectus",
    92: "Muscle - Medial Rectus",
    93: "Muscle - Lateral Rectus",
    94: "Muscle - Inferior Rectus",
    95: "Muscle - Superior Oblique",
    96: "Muscle - Inferior Oblique",
    97: "Air Internal - Oral Cavity",
    98: "Tendon - Temporalis Tendon",
    99: "Substantia Nigra",
    100: "Cerebral Peduncles",
    101: "Optic Chiasm",
    102: "Cranial Nerve I - Olfactory",
    103: "Cranial Nerve II - Optic",
    104: "Cranial Nerve III - Oculomotor",
    105: "Cranial Nerve IV - Trochlear",
    106: "Cranial Nerve V - Trigeminal",
    107: "Cranial Nerve V2 - Maxillary Division",
    108: "Cranial Nerve V3 - Mandibular Division",
    109: "Cranial Nerve VI - Abducens",
    110: "Cranial Nerve VII - Facial",
    111: "Cranial Nerve VIII - Vestibulocochlear",
    112: "Cranial Nerve IX - Glossopharyngeal",
    113: "Cranial Nerve X - Vagus",
    114: "Cranial Nerve XI - Accessory",
    115: "Cranial Nerve XII - Hypoglossal",
    116: "Thalamus",
}


# ---------------------------------------------------------------------------
# Per-label acoustic grouping
# ---------------------------------------------------------------------------
# Each of the 116 MIDA v1.0 labels is assigned to one acoustic group.
# Grouping follows tissue similarity in (c, rho, alpha) — not anatomy —
# so e.g. all cortical-bone structures (skull tables, mandible, vertebrae,
# teeth, hyoid) share a single group.
#
# Notes on edge cases:
#   - Skull Diploe (52) is trabecular; Inner/Outer Tables (53,54) are cortical.
#     Label 40 "Skull" is an aggregate/fallback that MIDA sometimes uses for
#     voxels not subdivided into diploe/tables — treated as cortical.
#   - Ear Cochlea (33) and Semicircular Canals (34) are bony labyrinth → cortical.
#   - Ear Auditory Canal (85) and Oral Cavity (97) are air-filled.
#   - Intervertebral Discs (49) are fibrocartilage → cartilage.
#   - Background (50) is remapped to water for the acoustic coupling medium.
#   - Label 0 has no MIDA meaning but is treated as water for padding robustness.

MIDA_LABEL_TO_GROUP: Dict[int, str] = {
    # Padding / background → water coupling medium
    0: "water",
    50: "water",
    # Dura
    1: "dura",
    # CSF
    6: "csf", 32: "csf",
    # Grey matter + deep nuclei + brainstem
    2: "grey_matter", 3: "grey_matter", 4: "grey_matter",
    5: "grey_matter", 7: "grey_matter", 8: "grey_matter",
    10: "grey_matter", 11: "grey_matter", 14: "grey_matter",
    15: "grey_matter", 16: "grey_matter", 17: "grey_matter",
    20: "grey_matter", 21: "grey_matter", 99: "grey_matter",
    100: "grey_matter", 116: "grey_matter",
    # White matter + spinal cord + major tracts
    9: "white_matter", 12: "white_matter", 13: "white_matter",
    18: "white_matter", 22: "white_matter", 23: "white_matter",
    # Glands
    19: "gland", 88: "gland", 89: "gland", 90: "gland",
    # Blood
    24: "blood_vessels", 25: "blood_vessels",
    # Air cavities
    26: "air", 27: "air", 28: "air", 29: "air",
    30: "air", 31: "air", 85: "air", 97: "air",
    # Cortical bone (skull aggregate + inner/outer tables + mandible
    # + vertebrae + teeth + hyoid + bony ear labyrinth)
    33: "cortical_bone", 34: "cortical_bone", 36: "cortical_bone",
    40: "cortical_bone", 41: "cortical_bone",
    44: "cortical_bone", 45: "cortical_bone", 46: "cortical_bone",
    47: "cortical_bone", 48: "cortical_bone",
    53: "cortical_bone", 54: "cortical_bone", 87: "cortical_bone",
    # Trabecular bone (skull diploe)
    52: "trabecular_bone",
    # Cartilage
    35: "cartilage", 39: "cartilage", 49: "cartilage",
    # Mucosa
    37: "mucosa", 86: "mucosa",
    # Muscle (general + platysma + facial/masticatory + extraocular + tongue)
    38: "muscle", 42: "muscle", 60: "muscle",
    63: "muscle", 64: "muscle", 65: "muscle", 66: "muscle",
    67: "muscle", 68: "muscle", 69: "muscle", 70: "muscle",
    71: "muscle", 72: "muscle", 73: "muscle", 74: "muscle",
    75: "muscle", 76: "muscle", 77: "muscle", 78: "muscle",
    79: "muscle", 80: "muscle", 81: "muscle", 82: "muscle",
    83: "muscle", 84: "muscle",
    91: "muscle", 92: "muscle", 93: "muscle", 94: "muscle",
    95: "muscle", 96: "muscle",
    # Skin (epidermis/dermis)
    51: "skin",
    # Fat (adipose + subcutaneous)
    43: "fat", 62: "fat",
    # Connective tissue (tendons)
    61: "connective", 98: "connective",
    # Eye sub-components
    55: "eye", 56: "eye", 57: "eye", 58: "eye", 59: "eye",
    # Nerves (cranial nerves + optic chiasm)
    101: "nerve",
    102: "nerve", 103: "nerve", 104: "nerve", 105: "nerve",
    106: "nerve", 107: "nerve", 108: "nerve", 109: "nerve",
    110: "nerve", 111: "nerve", 112: "nerve", 113: "nerve",
    114: "nerve", 115: "nerve",
}


# Derived: group_name → list of label IDs (preserves MIDA_TISSUE_GROUPS API)
def _build_tissue_groups() -> Dict[str, list]:
    groups: Dict[str, list] = {}
    for label, group in sorted(MIDA_LABEL_TO_GROUP.items()):
        groups.setdefault(group, []).append(label)
    return groups


MIDA_TISSUE_GROUPS: Dict[str, list] = _build_tissue_groups()


# ---------------------------------------------------------------------------
# Acoustic properties per tissue group
# ---------------------------------------------------------------------------
# Sources: ITRUSST benchmark (Aubry 2022), Duck (1990), Mast (2000),
# Connor (2002), Guasch et al. (2020) supplementary Table 1.

MIDA_ACOUSTIC_PROPERTIES: Dict[str, Dict[str, float]] = {
    #                    c [m/s]    rho [kg/m³]  alpha [dB/cm/MHz]
    "water":           {"sound_speed": 1500.0, "density": 1000.0, "attenuation": 0.0},
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

# Default for unknown labels (out-of-range) — behaves as water
_DEFAULT_PROPS = MIDA_ACOUSTIC_PROPERTIES["water"]

# Pre-build lookup arrays for fast vectorized mapping
_MAX_MIDA_LABEL = max(MIDA_LABEL_NAMES.keys())  # 116
_C_MIDA = np.full(_MAX_MIDA_LABEL + 1, _DEFAULT_PROPS["sound_speed"], dtype=np.float32)
_RHO_MIDA = np.full(_MAX_MIDA_LABEL + 1, _DEFAULT_PROPS["density"], dtype=np.float32)
_ALPHA_MIDA = np.full(_MAX_MIDA_LABEL + 1, _DEFAULT_PROPS["attenuation"], dtype=np.float32)

for _lab, _group in MIDA_LABEL_TO_GROUP.items():
    if _lab > _MAX_MIDA_LABEL:
        continue
    _props = MIDA_ACOUSTIC_PROPERTIES[_group]
    _C_MIDA[_lab] = _props["sound_speed"]
    _RHO_MIDA[_lab] = _props["density"]
    _ALPHA_MIDA[_lab] = _props["attenuation"]

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
        labels: Integer array of any shape. Valid MIDA v1.0 values are 1-116;
            label 50 (Background) and any unknown label map to water.

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


def center_crop_to_cube(volume: np.ndarray) -> np.ndarray:
    """Centre-crop a 3D volume to a cubic bounding box of ``min(shape)``.

    Preserves the head's native aspect ratio when the subsequent resample
    to a cubic target grid (e.g. 192^3) would otherwise stretch it.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")
    n = min(volume.shape)
    offsets = tuple((s - n) // 2 for s in volume.shape)
    slicer = tuple(slice(o, o + n) for o in offsets)
    return volume[slicer]


# ---------------------------------------------------------------------------
# FWI-ready phantom helper
# ---------------------------------------------------------------------------

# BrainWeb-style label used to mark the injected haemorrhage (same as the
# synthetic three-layer phantom in phantoms/synthetic.py).
_LESION_LABEL_BRAINWEB = 8
_LESION_C = 1584.0
_LESION_RHO = 1060.0
_LESION_ALPHA = 0.2

# Set of MIDA label IDs that represent brain parenchyma where a lesion
# may be placed — grey matter, white matter, and deep nuclei.
_MIDA_BRAIN_PARENCHYMA_LABELS = frozenset({
    2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 20, 21, 99, 100, 116,
})


# MIDA labels that correspond to internal air cavities (sinuses, oral cavity,
# mastoid, etc). Useful when you want to water-fill them for solver stability.
_MIDA_INTERNAL_AIR_LABELS = frozenset({26, 27, 28, 29, 30, 31, 85, 97})


def make_mida_phantom(
    path: Path,
    grid_shape: Tuple[int, int, int],
    dx: float,
    add_lesion: bool = True,
    lesion_offset_m: Tuple[float, float, float] = (-0.02, -0.01, 0.0),
    lesion_radius_m: float = 0.005,
    crop_cube: bool = True,
    water_fill_internal_air: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load MIDA and return an FWI-ready ``(labels, c, rho, alpha)`` tuple.

    The pipeline:
      1. Load the NIfTI label volume.
      2. Optionally centre-crop to cube so resample doesn't stretch the head.
      3. Resample (nearest-neighbour) to the requested ``grid_shape``.
      4. Inject a simulated haemorrhage inside the brain parenchyma if
         ``add_lesion=True`` (skipped if the requested location is outside
         the brain, e.g. the volume has been cropped too aggressively).
      5. Map labels to acoustic properties via
         :func:`map_mida_labels_to_acoustic`.
      6. Replace the background (label 50) with water coupling.

    Returns arrays matching the signature of
    ``run_full_usct.create_head_phantom`` so this is a drop-in
    replacement.

    Args:
        path: MIDA label NIfTI (e.g.
            ``/data/datasets/MIDAv1-0/MIDA_v1.0/MIDA_v1_voxels/MIDA_v1.nii``).
        grid_shape: ``(nx, ny, nz)`` target FWI grid.
        dx: Target grid spacing (m).
        add_lesion: Whether to inject a parametric haemorrhage.
        lesion_offset_m: Centre of the lesion relative to the grid centre.
        lesion_radius_m: Lesion radius in metres.
        crop_cube: If True, centre-crop to ``min(native_shape)`` before
            resampling. Prevents aspect-ratio distortion when the target
            grid is cubic but MIDA native is 480x480x350.
        water_fill_internal_air: If True, remap MIDA's internal air
            cavities (sinuses, oral cavity, etc. — labels 26-31, 85, 97)
            to water. Reduces impedance contrasts that can destabilise
            pseudospectral solvers at coarse grids. The air is
            anatomically correct but FWI-unfriendly; default False
            preserves realism.
    """
    labels = load_mida_volume(path)
    if crop_cube:
        labels = center_crop_to_cube(labels)
    labels = resample_volume(labels, tuple(grid_shape), order=0)

    if add_lesion:
        nx, ny, nz = grid_shape
        cx, cy, cz = nx / 2, ny / 2, nz / 2
        lx, ly, lz = lesion_offset_m
        x, y, z = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij",
        )
        lr = np.sqrt(
            ((x - cx - lx / dx) / (lesion_radius_m / dx)) ** 2
            + ((y - cy - ly / dx) / (lesion_radius_m / dx)) ** 2
            + ((z - cz - lz / dx) / (lesion_radius_m / dx)) ** 2
        )
        brain_mask = np.isin(labels, list(_MIDA_BRAIN_PARENCHYMA_LABELS))
        lesion_mask = (lr <= 1.0) & brain_mask
        if lesion_mask.any():
            # Use BrainWeb's blood label (8) — outside the MIDA 1-116 range
            # but honoured by map_mida_labels_to_acoustic's clip, and we
            # override the properties explicitly below to keep the
            # interpretation unambiguous.
            labels = np.where(lesion_mask, _LESION_LABEL_BRAINWEB, labels)

    props = map_mida_labels_to_acoustic(labels)
    c = np.asarray(props["sound_speed"])
    rho = np.asarray(props["density"])
    alpha = np.asarray(props["attenuation"])

    # Explicit lesion acoustic values (BrainWeb blood, matches synthetic phantom)
    if add_lesion:
        lesion_mask_np = labels == _LESION_LABEL_BRAINWEB
        c = np.where(lesion_mask_np, _LESION_C, c)
        rho = np.where(lesion_mask_np, _LESION_RHO, rho)
        alpha = np.where(lesion_mask_np, _LESION_ALPHA, alpha)

    # Background (MIDA label 50) -> water coupling. Already handled in
    # map_mida_labels_to_acoustic via the "water" group, but force it here
    # in case any out-of-range labels (e.g. interpolation artefacts) slip
    # through.
    bg_mask = labels == 50
    c = np.where(bg_mask, 1500.0, c)
    rho = np.where(bg_mask, 1000.0, rho)
    alpha = np.where(bg_mask, 0.0, alpha)

    if water_fill_internal_air:
        air_mask = np.isin(labels, list(_MIDA_INTERNAL_AIR_LABELS))
        c = np.where(air_mask, 1500.0, c)
        rho = np.where(air_mask, 1000.0, rho)
        alpha = np.where(air_mask, 0.0, alpha)

    return (
        jnp.asarray(labels),
        jnp.asarray(c),
        jnp.asarray(rho),
        jnp.asarray(alpha),
    )

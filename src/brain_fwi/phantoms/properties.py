"""Acoustic tissue property mapping for transcranial ultrasound FWI.

Extended property table covering all 12 BrainWeb tissue classes plus
support for MIDA and SCI Institute head models. Values sourced from
ITRUSST benchmark (Aubry et al. 2022, JASA 152(2):1003-1019) and
literature compilations by Mast (2000), Connor (2002), and Duck (1990).

The BrainWeb discrete labels are:
    0  = Background (air/outside)
    1  = CSF
    2  = Grey matter
    3  = White matter
    4  = Fat
    5  = Muscle
    6  = Muscle/Skin
    7  = Skull (cortical bone)
    8  = Blood vessels
    9  = Connective tissue (around fat)
    10 = Dura mater
    11 = Bone marrow (trabecular)
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Acoustic property table
# ---------------------------------------------------------------------------
# Each entry: (sound_speed_m_s, density_kg_m3, attenuation_db_cm_mhz)
#
# References per tissue:
#   Water/CSF:    ITRUSST benchmark BM1 (Aubry 2022)
#   Brain (GM/WM): Aubry 2022, Table III
#   Skull cortical: Aubry 2022 BM3 (cortical layer, 2800 m/s)
#   Skull trabecular: Aubry 2022 BM3 (diploe layer, 2300 m/s)
#   Soft tissues:  Duck (1990) "Physical Properties of Tissues"
#   Blood:         Mast (2000) JASA 108(3):1023

TISSUE_PROPERTIES: Dict[int, Tuple[float, float, float]] = {
    #       (c [m/s], rho [kg/m³], alpha [dB/cm/MHz])
    0:  (  343.0, 1.225,  0.0),    # air / background
    1:  ( 1500.0, 1007.0, 0.002),  # CSF (≈ water)
    2:  ( 1560.0, 1040.0, 0.6),    # grey matter
    3:  ( 1560.0, 1040.0, 0.6),    # white matter
    4:  ( 1478.0,  950.0, 0.4),    # fat
    5:  ( 1547.0, 1050.0, 1.0),    # muscle
    6:  ( 1540.0, 1090.0, 0.8),    # muscle/skin (scalp)
    7:  ( 2800.0, 1850.0, 4.0),    # skull — cortical bone (ITRUSST BM3)
    8:  ( 1584.0, 1060.0, 0.2),    # blood vessels
    9:  ( 1520.0, 1030.0, 0.5),    # connective tissue
    10: ( 1560.0, 1080.0, 0.5),    # dura mater
    11: ( 2300.0, 1700.0, 8.0),    # bone marrow / trabecular (ITRUSST BM3)
}

# SCI Institute head model label mapping (6 labels):
#   0=background, 1=scalp, 2=skull, 3=CSF, 4=GM, 5=WM
SCI_TO_BRAINWEB = {0: 0, 1: 6, 2: 7, 3: 1, 4: 2, 5: 3}


# Pre-build numpy lookup arrays for fast indexing
_MAX_LABEL = max(TISSUE_PROPERTIES.keys())
_C_LOOKUP = np.zeros(_MAX_LABEL + 1, dtype=np.float32)
_RHO_LOOKUP = np.zeros(_MAX_LABEL + 1, dtype=np.float32)
_ALPHA_LOOKUP = np.zeros(_MAX_LABEL + 1, dtype=np.float32)

for _lab, (_c, _rho, _alpha) in TISSUE_PROPERTIES.items():
    _C_LOOKUP[_lab] = _c
    _RHO_LOOKUP[_lab] = _rho
    _ALPHA_LOOKUP[_lab] = _alpha

# Convert to JAX arrays (immutable, on-device)
_C_JAX = jnp.array(_C_LOOKUP)
_RHO_JAX = jnp.array(_RHO_LOOKUP)
_ALPHA_JAX = jnp.array(_ALPHA_LOOKUP)


# ---------------------------------------------------------------------------
# Mapping functions
# ---------------------------------------------------------------------------

def map_labels_to_speed(labels: jnp.ndarray) -> jnp.ndarray:
    """Map integer tissue labels to sound speed (m/s)."""
    return _C_JAX[jnp.clip(labels, 0, _MAX_LABEL).astype(jnp.int32)]


def map_labels_to_density(labels: jnp.ndarray) -> jnp.ndarray:
    """Map integer tissue labels to density (kg/m^3)."""
    return _RHO_JAX[jnp.clip(labels, 0, _MAX_LABEL).astype(jnp.int32)]


def map_labels_to_attenuation(labels: jnp.ndarray) -> jnp.ndarray:
    """Map integer tissue labels to attenuation (dB/cm/MHz)."""
    return _ALPHA_JAX[jnp.clip(labels, 0, _MAX_LABEL).astype(jnp.int32)]


def map_labels_to_all(
    labels: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Map tissue labels to all acoustic properties.

    Returns dict with keys: 'sound_speed', 'density', 'attenuation'.
    """
    safe = jnp.clip(labels, 0, _MAX_LABEL).astype(jnp.int32)
    return {
        "sound_speed": _C_JAX[safe],
        "density": _RHO_JAX[safe],
        "attenuation": _ALPHA_JAX[safe],
    }


def remap_sci_labels(sci_labels: jnp.ndarray) -> jnp.ndarray:
    """Remap SCI Institute head model labels to BrainWeb convention."""
    lookup = jnp.array([SCI_TO_BRAINWEB.get(i, 0) for i in range(6)],
                       dtype=jnp.int32)
    return lookup[jnp.clip(sci_labels, 0, 5).astype(jnp.int32)]

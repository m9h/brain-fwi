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


# ---------------------------------------------------------------------------
# GM/WM contrast (issue #47)
# ---------------------------------------------------------------------------
# The ITRUSST table above assigns grey and white matter identical properties
# (both 1560 m/s, 0.6 dB/cm/MHz) — faithful to the benchmark but zero contrast,
# so a reconstruction cannot be scored for GM/WM.
#
# IMPORTANT — this table's GM==WM speed is a BENCHMARK CONVENTION, NOT PHYSICS.
# Treating it as physics produced a false premise ("velocity FWI is structurally
# blind to GM/WM") that mis-steered the project. Real tissue separates in BOTH
# channels:
#   * ATTENUATION — WM ~1.5x GM (Kang et al., Ultrasonics 2022).
#   * SOUND SPEED — ~17 m/s (~1%). Mitcham et al., Med. Phys. 2025
#     (DOI 10.1002/mp.18090) measure ex-vivo human WM (corpus callosum)
#     1639 +/- 3 vs GM (cerebellum) 1656 +/- 6 m/s. Note the tight error bars:
#     this is a 3-5 sigma separation, NOT "at the noise floor" as older
#     estimates (SD 10-14 m/s) suggested. Ex-vivo fixation elevates the absolute
#     values (~1650 vs in-vivo ~1560), so only the DIFFERENCE transfers.
#     Literature is not consistent on the SIGN; detection of a GM/WM boundary
#     does not depend on it.
# The real constraint on seeing GM/WM in a velocity map is RESOLUTION, not
# contrast: the cortical ribbon is 2.5-4 mm, so it needs >=300-500 kHz
# (lambda/2 = 2.6-1.6 mm). Below ~200 kHz it is unresolvable at any contrast.
# Contrast is OPT-IN so the default table stays ITRUSST-faithful.
GM_WM_ALPHA_RATIO = 1.5   # Kang et al. 2022
GM_WM_C_DELTA = -17.0     # WM - GM sound speed (m/s), Mitcham et al. 2025


def tissue_properties_contrasted(
    alpha_ratio: float = GM_WM_ALPHA_RATIO,
    wm_c_delta: float = 0.0,
    wm_rho_delta: float = 0.0,
) -> Dict[int, Tuple[float, float, float]]:
    """A copy of :data:`TISSUE_PROPERTIES` with a measured GM/WM contrast.

    White matter (label 3) gets ``alpha_ratio`` x grey matter's attenuation
    (Kang 2022) and, optionally, a sound-speed offset. Grey matter and all other
    tissues are untouched; the global table is NOT mutated.

    ``wm_c_delta`` defaults to 0 to preserve backwards compatibility, but for a
    PHYSICALLY REALISTIC phantom pass ``wm_c_delta=GM_WM_C_DELTA`` (-17 m/s,
    Mitcham 2025). Speed is *not* negligible — see the module note above; the
    binding constraint on resolving GM/WM is imaging frequency, not contrast.

    Args:
        alpha_ratio: WM/GM attenuation ratio (default 1.5, Kang 2022).
        wm_c_delta: WM sound-speed offset (m/s) added to GM's speed. Pass
            :data:`GM_WM_C_DELTA` for the measured value.
        wm_rho_delta: optional WM density offset (kg/m^3).

    Returns:
        A new ``{label: (c, rho, alpha)}`` dict.
    """
    props = dict(TISSUE_PROPERTIES)
    gm_c, gm_rho, gm_alpha = props[2]
    props[3] = (gm_c + wm_c_delta, gm_rho + wm_rho_delta, gm_alpha * alpha_ratio)
    return props


def _lookups_for(properties: Dict[int, Tuple[float, float, float]]):
    """Build (c, rho, alpha) JAX lookup arrays for an arbitrary property table."""
    max_label = max(properties.keys())
    c = np.zeros(max_label + 1, np.float32)
    rho = np.zeros(max_label + 1, np.float32)
    alpha = np.zeros(max_label + 1, np.float32)
    for lab, (cc, rr, aa) in properties.items():
        c[lab], rho[lab], alpha[lab] = cc, rr, aa
    return jnp.array(c), jnp.array(rho), jnp.array(alpha)


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
    properties: Dict[int, Tuple[float, float, float]] = None,
) -> Dict[str, jnp.ndarray]:
    """Map tissue labels to all acoustic properties.

    Args:
        labels: integer tissue-label array.
        properties: optional custom ``{label: (c, rho, alpha)}`` table, e.g.
            :func:`tissue_properties_contrasted` for GM/WM contrast (#47).
            Defaults to the canonical ITRUSST :data:`TISSUE_PROPERTIES`.

    Returns dict with keys: 'sound_speed', 'density', 'attenuation'.
    """
    if properties is None:
        c_jax, rho_jax, alpha_jax, max_label = _C_JAX, _RHO_JAX, _ALPHA_JAX, _MAX_LABEL
    else:
        c_jax, rho_jax, alpha_jax = _lookups_for(properties)
        max_label = max(properties.keys())
    safe = jnp.clip(labels, 0, max_label).astype(jnp.int32)
    return {
        "sound_speed": c_jax[safe],
        "density": rho_jax[safe],
        "attenuation": alpha_jax[safe],
    }


def remap_sci_labels(sci_labels: jnp.ndarray) -> jnp.ndarray:
    """Remap SCI Institute head model labels to BrainWeb convention."""
    lookup = jnp.array([SCI_TO_BRAINWEB.get(i, 0) for i in range(6)],
                       dtype=jnp.int32)
    return lookup[jnp.clip(sci_labels, 0, 5).astype(jnp.int32)]

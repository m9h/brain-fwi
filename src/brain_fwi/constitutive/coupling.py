"""Velocity→attenuation constitutive coupling (Phase 6.2, issue #45).

Phase-only FWI recovers sound speed well (traveltime is a strong constraint) but
attenuation weakly. Because c and α are properties of the *same tissue*, the
recovered velocity predicts α through the constitutive (c, α) relation — the
causal / Kramers-Kronig premise that both derive from one medium. Used as a
ramped prior in :func:`brain_fwi.inversion.fwi.run_fwi`, this lets the strong
velocity channel inform the weak attenuation channel.

Honest limit: grey and white matter share the same sound speed (1560 m/s), so
the c→α map is **degenerate** there — the coupling helps c-contrasted tissues
(skull, CSF, brain-vs-background) but cannot separate GM/WM, which stays
data-limited (needs higher frequency / multi-band, issue #48).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import jax.numpy as jnp

from brain_fwi.phantoms.properties import TISSUE_PROPERTIES


def speed_alpha_anchors(
    properties: Optional[Dict[int, Tuple[float, float, float]]] = None,
    labels: Optional[Sequence[int]] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sorted, unique-in-speed (c, α) anchor points for the c→α map.

    Tissues that share a sound speed (GM/WM/dura at 1560) are collapsed to a
    single anchor with their mean α — the degeneracy that makes velocity unable
    to distinguish them.

    Returns:
        ``(c_anchors, alpha_anchors)``, both 1-D and strictly increasing in c.
    """
    props = properties if properties is not None else TISSUE_PROPERTIES
    keys = list(props) if labels is None else list(labels)
    by_speed: Dict[float, list] = {}
    for lab in keys:
        c, _rho, a = props[lab]
        by_speed.setdefault(round(float(c), 3), []).append(float(a))
    cs = sorted(by_speed)
    alphas = [float(np.mean(by_speed[c])) for c in cs]
    return jnp.asarray(cs, jnp.float32), jnp.asarray(alphas, jnp.float32)


def alpha_from_speed(
    c_field: jnp.ndarray,
    c_anchors: jnp.ndarray,
    alpha_anchors: jnp.ndarray,
) -> jnp.ndarray:
    """Predict attenuation from sound speed by interpolating the (c, α) anchors.

    ``jnp.interp`` clamps outside the anchor range (flat extrapolation), which is
    the desired behaviour — an out-of-range recovered speed maps to the nearest
    tissue's α rather than diverging.
    """
    return jnp.interp(c_field, jnp.asarray(c_anchors), jnp.asarray(alpha_anchors))

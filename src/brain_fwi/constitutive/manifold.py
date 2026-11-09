"""Tissue-α manifold prior for attenuation inversion (Phase 6.2).

Phase-only FWI constrains attenuation weakly, so a free per-voxel α field
smears. The remedy is to tie α to the low-dimensional set of *physical tissue
attenuation values* — the CANN α(ω) manifold expressed in power-law-coefficient
form (α at 1 MHz, dB/cm/MHz, the same units :func:`build_medium` consumes).
Instead of inverting a noisy α image we invert toward "a few coefficients + a
spatial field".

The prior is a proximal snap toward the nearest archetype, applied to the α
field after each optimiser step (a projection onto the tissue manifold). Because
it snaps toward the *nearest* archetype, a sub-midpoint value is pulled toward
water — so it must be RAMPED IN LATE, after the data has grown α past the
archetype midpoint (see :func:`brain_fwi.inversion.fwi.run_fwi`).

The companion causal coupling (Kramers–Kronig) lives in :mod:`.kk`.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import jax.numpy as jnp

from brain_fwi.phantoms.properties import TISSUE_PROPERTIES

# BrainWeb label → archetype name, for the α levels that matter to brain FWI.
# (Values come from the single source of truth, TISSUE_PROPERTIES, so the
# manifold and the phantoms can never drift apart.)
_ARCHETYPE_LABELS: Dict[str, int] = {
    "water": 1,            # CSF ≈ water, α ≈ 0
    "brain": 2,            # grey/white matter (both 0.6 today — see issue #47)
    "cortical_bone": 7,    # skull cortical layer
    "trabecular_bone": 11, # skull diploë / marrow
}


def tissue_alpha_coefficients(
    names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Physical tissue attenuation coefficients (α at 1 MHz, dB/cm/MHz).

    Args:
        names: subset of archetype names; default = all brain-FWI archetypes.

    Returns:
        ``{name: alpha_db_cm_mhz}`` drawn from ``TISSUE_PROPERTIES``.
    """
    keys = list(_ARCHETYPE_LABELS) if names is None else list(names)
    out: Dict[str, float] = {}
    for k in keys:
        label = _ARCHETYPE_LABELS[k]
        out[k] = float(TISSUE_PROPERTIES[label][2])
    return out


def manifold_proximal(
    alpha: jnp.ndarray,
    archetypes: jnp.ndarray,
    beta: float = 0.5,
) -> jnp.ndarray:
    """Proximal snap of an α field toward the nearest tissue archetype.

    ``alpha ← alpha + beta·(nearest_archetype − alpha)``, elementwise. ``beta``
    in [0, 1] controls how hard the projection pulls (1 = hard snap, 0 = no-op).

    Args:
        alpha: α field (any shape), dB/cm/MHz.
        archetypes: 1-D array of archetype α values.
        beta: pull strength in [0, 1].

    Returns:
        α field of the same shape, moved toward the tissue manifold.
    """
    archetypes = jnp.asarray(archetypes)
    # distance to every archetype; pick the nearest.
    dist = jnp.abs(alpha[..., None] - archetypes)        # (..., K)
    nearest = archetypes[jnp.argmin(dist, axis=-1)]      # (...)
    return alpha + beta * (nearest - alpha)


def manifold_prior_grad(
    alpha: jnp.ndarray,
    archetypes: jnp.ndarray,
    temperature: float = 0.5,
) -> jnp.ndarray:
    """Differentiable soft-attraction gradient toward the α manifold.

    Gradient of ``½·softmin-distance²`` w.r.t. ``alpha`` — a smooth analogue of
    :func:`manifold_proximal` for callers that prefer to add the prior to the
    optimisation gradient rather than apply a proximal step. ``temperature``
    sets how softly archetypes are blended (→0 recovers the hard nearest).

    Returns ``alpha − soft_target`` where ``soft_target`` is the
    softmin-weighted archetype mean.
    """
    archetypes = jnp.asarray(archetypes)
    d2 = (alpha[..., None] - archetypes) ** 2
    w = jnp.exp(-d2 / (2.0 * temperature ** 2 + 1e-30))
    w = w / (jnp.sum(w, axis=-1, keepdims=True) + 1e-30)
    soft_target = jnp.sum(w * archetypes, axis=-1)
    return alpha - soft_target

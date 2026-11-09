"""Crossing-fibre model — resolving fibre crossings from the angular attenuation
profile (the acoustic analogue of DTI -> HARDI).

Two fibres each with a SOFT sin^2(theta-phi) attenuation profile sum to a single
2-theta sinusoid: 2 measured numbers (amplitude, phase) cannot recover 4 unknowns
(two directions + two amplitudes) — crossings are fundamentally unresolvable with
a pure-2theta (DTI-like) model. Crossings become resolvable only when the single-
fibre profile is SHARPER (sin^4, ...), carrying higher (4-theta) harmonics — the
acoustic analogue of high-order ODFs / spherical deconvolution.

Signature: for an ORTHOGONAL crossing the 2-theta terms CANCEL (phases pi apart)
while the 4-theta terms ADD, so the 4theta/2theta ratio is the crossing detector.

One behaviour per test, each written failing first.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest


def _angles(n=180):
    return jnp.linspace(0.0, np.pi, n, endpoint=False)


def test_fibre_profile_min_along_fibre_max_across():
    """Attenuation is minimal along the fibre (theta=phi), maximal across it."""
    from brain_fwi.inversion.crossing_fibres import fibre_profile
    phi = 0.4
    assert float(fibre_profile(jnp.array(phi), phi, 1.0, sharpness=2)) < 1e-6
    assert float(fibre_profile(jnp.array(phi + np.pi / 2), phi, 1.0, sharpness=2)) == pytest.approx(1.0, abs=1e-5)


def test_crossing_index_detects_sharp_crossing_only():
    """The 4theta/2theta index is small for a single fibre, LARGE for a sharp
    (sin^4) orthogonal crossing, and ~0 for a soft (sin^2) crossing (which has no
    4theta content — the fundamental limit)."""
    from brain_fwi.inversion.crossing_fibres import (
        two_fibre_profile, angular_harmonics, crossing_index)
    th = _angles()
    single = two_fibre_profile(th, 0.1, 1.0, 0.3, 0.0, 0.3, sharpness=2)   # a2=0 -> single
    cross_sharp = two_fibre_profile(th, 0.1, 1.0, 0.0, 1.0, np.pi / 2, sharpness=2)
    # soft (sin^2) crossing: pure 2-theta, no 4-theta -> unresolvable. (Use a
    # non-orthogonal angle; the orthogonal soft crossing degenerates to a constant
    # = fully isotropic, an even stronger statement of the limit.)
    cross_soft = two_fibre_profile(th, 0.1, 1.0, 0.0, 1.0, np.pi / 3, sharpness=1)
    idx_single = crossing_index(angular_harmonics(th, single))
    idx_sharp = crossing_index(angular_harmonics(th, cross_sharp))
    idx_soft = crossing_index(angular_harmonics(th, cross_soft))
    assert idx_sharp > 3 * idx_single, f"crossing not flagged: sharp {idx_sharp:.2f} vs single {idx_single:.2f}"
    assert idx_soft < 0.05, f"soft crossing should have ~no 4theta, got {idx_soft:.3f}"


@pytest.mark.slow
def test_fit_two_fibres_recovers_crossing_directions():
    """A sharp (sin^4) orthogonal crossing: the two-fibre fit recovers BOTH fibre
    directions (~within 10 deg), where a single-direction model would return only
    their resultant."""
    from brain_fwi.inversion.crossing_fibres import two_fibre_profile, fit_two_fibres
    th = _angles(240)
    phi1, phi2 = 0.2, 0.2 + np.pi / 2
    alpha = two_fibre_profile(th, 0.1, 1.0, phi1, 1.0, phi2, sharpness=2)
    res = fit_two_fibres(th, alpha, sharpness=2)
    got = sorted([res.phi1 % np.pi, res.phi2 % np.pi])
    exp = sorted([phi1 % np.pi, phi2 % np.pi])
    err = max(abs((g - e + np.pi / 2) % np.pi - np.pi / 2) for g, e in zip(got, exp))
    assert err < np.radians(10), f"crossing directions off by {np.degrees(err):.1f} deg: {got} vs {exp}"

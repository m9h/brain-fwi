"""Per-voxel two-fibre resolution on the attenuation ODF — explicit crossing
directions (the tomography's "acoustic HARDI" output)."""

from __future__ import annotations

import numpy as np
import pytest


def _crossing_phantom(N=32):
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    r = np.sqrt((xx - N / 2) ** 2 + (yy - N / 2) ** 2)
    brain = r <= N * 0.42
    cross = (r <= N * 0.16) & brain
    single = brain & ~cross
    amps = np.zeros((N, N, 2), np.float32); phis = np.zeros((N, N, 2), np.float32)
    amps[brain, 0] = 1.0; phis[..., 0] = 0.2
    amps[cross, 1] = 1.0; phis[cross, 1] = 0.2 + np.pi / 2
    return brain, cross, single, amps, phis


def test_resolve_odf_crossings_recovers_two_directions():
    """On ground-truth ODF fields: the crossing patch yields two distinct fibre
    directions (~0.2 and ~0.2+pi/2); the single-fibre surround yields one (its
    second amplitude ~0)."""
    from brain_fwi.inversion.crossing_fibres import sin4_odf_coeffs, resolve_odf_crossings
    N = 32
    brain, cross, single, amps, phis = _crossing_phantom(N)
    a0, c2, s2, c4, s4 = sin4_odf_coeffs(0.6 * brain, amps, phis)
    res = resolve_odf_crossings(a0, c2, s2, c4, s4, brain, sharpness=2, n_grid=30)

    # single-fibre region: second amplitude is negligible vs first
    a1s, a2s = res.a1[single], res.a2[single]
    assert np.median(a2s) < 0.25 * np.median(a1s + 1e-6), "single region wrongly split into two fibres"
    # crossing patch: two comparable amplitudes...
    a1c, a2c = res.a1[cross], res.a2[cross]
    assert np.median(a2c) > 0.4 * np.median(a1c), "crossing patch did not yield a real second fibre"
    # ...at ~orthogonal directions
    dth = np.abs((res.phi1[cross] - res.phi2[cross] + np.pi / 2) % np.pi - np.pi / 2)
    assert np.median(dth) > np.radians(60), f"crossing angle {np.degrees(np.median(dth)):.0f} deg not ~90"
    # and one of them matches the true fibre-0 direction (0.2)
    e1 = np.abs((res.phi1[cross] - 0.2 + np.pi / 2) % np.pi - np.pi / 2)
    e2 = np.abs((res.phi2[cross] - 0.2 + np.pi / 2) % np.pi - np.pi / 2)
    assert np.median(np.minimum(e1, e2)) < np.radians(12)


def test_resolve_returns_fields_shaped_like_grid():
    from brain_fwi.inversion.crossing_fibres import sin4_odf_coeffs, resolve_odf_crossings
    N = 24
    brain, cross, single, amps, phis = _crossing_phantom(N)
    a0, c2, s2, c4, s4 = sin4_odf_coeffs(0.6 * brain, amps, phis)
    res = resolve_odf_crossings(a0, c2, s2, c4, s4, brain, sharpness=2, n_grid=20)
    for f in (res.phi1, res.phi2, res.a1, res.a2):
        assert np.asarray(f).shape == (N, N)

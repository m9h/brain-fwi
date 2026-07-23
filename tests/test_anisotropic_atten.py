"""The path forward for GM/WM: multi-angle anisotropic attenuation tomography.

Sound speed is degenerate (GM=WM), spectral shape is not band-resolvable, and the
isotropic magnitude contrast is small. The one axis with real leverage is
ANISOTROPY: white matter is fibre-oriented so its attenuation depends on the
propagation angle, alpha(theta) = alpha_iso + alpha_aniso * sin^2(theta - phi);
gray matter is isotropic (alpha_aniso = 0). Transmission tomography crosses every
interior voxel with rays at many angles, so the per-voxel anisotropy is
recoverable — and it separates WM from GM even when their ISOTROPIC properties are
identical.

This is a straight-ray proof of principle (the leverage is the angular coverage,
which a full-wave solver inherits).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest


def test_anisotropic_alpha_min_along_fibre_max_across():
    """alpha is minimal along the fibre (theta=phi) and maximal across it."""
    from brain_fwi.inversion.anisotropic_atten import anisotropic_alpha
    phi = 0.3
    along = float(anisotropic_alpha(jnp.array(phi), 0.6, 0.4, phi))
    across = float(anisotropic_alpha(jnp.array(phi + np.pi / 2), 0.6, 0.4, phi))
    assert along == pytest.approx(0.6, abs=1e-6)
    assert across == pytest.approx(1.0, abs=1e-6)


def test_forward_uniform_isotropic_decay_equals_alpha_times_length():
    """A uniform isotropic field: every ray's decay = alpha_iso * ray length."""
    from brain_fwi.inversion.anisotropic_atten import make_ring_rays, forward_ray_decay
    N, dx = 32, 1.0e-3
    a_iso = jnp.full((N, N), 0.5)
    a_aniso = jnp.zeros((N, N))
    phi = jnp.zeros((N, N))
    rays = make_ring_rays(N, dx, n_trans=24, radius_frac=0.42)
    decay = np.asarray(forward_ray_decay(a_iso, a_aniso, phi, rays))
    lengths = np.asarray(rays.length)
    # decay / length should be ~constant = alpha_iso (interior rays; edge rays
    # clip outside the field so allow a tolerance).
    ratio = decay / (lengths + 1e-12)
    assert np.nanmedian(ratio) == pytest.approx(0.5, rel=0.15)


@pytest.mark.slow
def test_anisotropy_separates_wm_from_gm_with_identical_isotropic_alpha():
    """THE proof of the path: GM (isotropic) and WM (anisotropic) with the SAME
    isotropic alpha are separated by the recovered alpha_aniso map, from
    multi-angle ray data — the two-stage recipe (bulk alpha from the isotropic
    pass, then anisotropy from the residual angular signal).

    Sound speed cannot separate them (degenerate), the isotropic alpha here is
    identical by construction, yet the ANISOTROPY map lights up WM and stays ~0
    in GM. That is the leverage GM/WM discrimination needs.

    (The bulk alpha is supplied here — as the Phase-6 isotropic FWI provides it.
    The fully-BLIND joint bulk+anisotropy separation is a harder tomographic
    identifiability problem, documented in the module; naive gradient descent
    lands in a crosstalk basin. This test proves the angular leverage exists and
    is recoverable given a bulk estimate.)"""
    from brain_fwi.inversion.anisotropic_atten import (
        make_ring_rays, forward_ray_decay, invert_anisotropic)
    N, dx = 40, 1.0e-3
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    r = np.sqrt((xx - N / 2) ** 2 + (yy - N / 2) ** 2)
    brain = r <= N * 0.4
    wm = r <= N * 0.22                      # inner disc = white matter
    gm = brain & ~wm                        # outer ring = gray matter
    a_iso = np.where(brain, 0.6, 0.0).astype(np.float32)      # SAME for GM and WM
    a_aniso = np.where(wm, 0.5, 0.0).astype(np.float32)       # only WM is anisotropic
    phi = np.full((N, N), 0.6, np.float32)                    # known fibre direction (DTI)

    rays = make_ring_rays(N, dx, n_trans=64, radius_frac=0.46)
    obs = forward_ray_decay(jnp.asarray(a_iso), jnp.asarray(a_aniso), jnp.asarray(phi), rays)

    # Bulk alpha estimate (from the isotropic pass) supplied; recover anisotropy.
    rec = invert_anisotropic(obs, rays, jnp.asarray(phi), (N, N),
                             mask=jnp.asarray(brain.astype(np.float32)),
                             a_iso=jnp.asarray(a_iso), n_iters=2000)
    aa = np.asarray(rec.a_aniso)
    wm_aniso, gm_aniso = aa[wm].mean(), aa[gm].mean()
    assert wm_aniso > 3 * max(gm_aniso, 1e-6), (
        f"anisotropy did not separate WM({wm_aniso:.3f}) from GM({gm_aniso:.3f})")
    assert wm_aniso > 0.3, f"WM anisotropy under-recovered: {wm_aniso:.3f} (true 0.5)"


@pytest.mark.slow
def test_blind_joint_recovery_with_homogeneous_bulk_prior():
    """BLIND (no bulk supplied): a homogeneous-bulk structural prior — the Living
    Matter Lab way of taming an ill-posed inversion — makes the joint bulk +
    anisotropy recovery identifiable. The bulk cannot absorb sharp WM structure,
    so it lands in the anisotropy channel; WM/GM separate ~10x and the scalar
    bulk recovers ~truth."""
    from brain_fwi.inversion.anisotropic_atten import (
        make_ring_rays, forward_ray_decay, invert_anisotropic)
    N, dx = 40, 1.0e-3
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    r = np.sqrt((xx - N / 2) ** 2 + (yy - N / 2) ** 2)
    brain = r <= N * 0.4; wm = r <= N * 0.22; gm = brain & ~wm
    a_iso = np.where(brain, 0.6, 0.0).astype(np.float32)
    a_aniso = np.where(wm, 0.5, 0.0).astype(np.float32)
    phi = np.full((N, N), 0.6, np.float32)
    rays = make_ring_rays(N, dx, n_trans=64, radius_frac=0.46)
    obs = forward_ray_decay(jnp.asarray(a_iso), jnp.asarray(a_aniso), jnp.asarray(phi), rays)

    rec = invert_anisotropic(obs, rays, jnp.asarray(phi), (N, N),   # a_iso=None -> blind
                             mask=jnp.asarray(brain.astype(np.float32)), n_iters=3000)
    aa = np.asarray(rec.a_aniso); ai = np.asarray(rec.a_iso)
    assert aa[wm].mean() > 3 * max(aa[gm].mean(), 1e-6), (
        f"blind: anisotropy WM({aa[wm].mean():.3f}) vs GM({aa[gm].mean():.3f})")
    assert abs(ai[brain].mean() - 0.6) < 0.1, f"blind bulk off: {ai[brain].mean():.3f} (true 0.6)"

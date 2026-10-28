"""Per-voxel attenuation-ODF tomography — resolving fibre crossings end-to-end.

The single-phi (2-theta) tomography averages crossings to a resultant direction
(the DiSCo hub failure). Inverting the full per-voxel angular ODF (2-theta AND
4-theta harmonic fields) carries the crossing information: where two sharp fibres
cross, the recovered 4-theta content lights up (crossing_index), where a single
fibre lies it stays low.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest


def test_odf_forward_uniform_bulk_only():
    """a0-only ODF (no harmonics): every ray decays a0 * length."""
    from brain_fwi.inversion.anisotropic_atten import make_ring_rays, forward_ray_decay_odf
    N, dx = 32, 1e-3
    z = jnp.zeros((N, N)); a0 = jnp.full((N, N), 0.5)
    rays = make_ring_rays(N, dx, n_trans=24, radius_frac=0.42)
    dec = np.asarray(forward_ray_decay_odf(a0, z, z, z, z, rays))
    ratio = dec / (np.asarray(rays.length) + 1e-12)
    assert np.nanmedian(ratio) == pytest.approx(0.5, rel=0.15)


@pytest.mark.slow
def test_odf_tomography_lights_up_crossing_region():
    """A central patch of two orthogonal sharp (sin^4) fibres vs a single-fibre
    surround: the recovered 4-theta/2-theta crossing map is much larger in the
    crossing patch than in the single-fibre region."""
    from brain_fwi.inversion.anisotropic_atten import (
        make_ring_rays, forward_ray_decay_odf, invert_odf)
    from brain_fwi.inversion.crossing_fibres import sin4_odf_coeffs, crossing_index_map
    N, dx = 40, 1.0e-3
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    r = np.sqrt((xx - N / 2) ** 2 + (yy - N / 2) ** 2)
    brain = r <= N * 0.4
    cross = (r <= N * 0.13) & brain            # central crossing patch
    single = brain & ~cross

    amps = np.zeros((N, N, 2), np.float32); phis = np.zeros((N, N, 2), np.float32)
    amps[brain, 0] = 1.0; phis[..., 0] = 0.2                 # fibre 0 everywhere in brain
    amps[cross, 1] = 1.0; phis[cross, 1] = 0.2 + np.pi / 2   # orthogonal fibre 1 in the patch
    a0, c2, s2, c4, s4 = sin4_odf_coeffs(0.6 * brain, amps, phis)

    rays = make_ring_rays(N, dx, n_trans=72, radius_frac=0.46)
    obs = forward_ray_decay_odf(*[jnp.asarray(f.astype(np.float32)) for f in (a0, c2, s2, c4, s4)], rays)
    rec = invert_odf(obs, rays, (N, N), mask=jnp.asarray(brain.astype(np.float32)), n_iters=8000)
    ci = crossing_index_map(np.asarray(rec.c2), np.asarray(rec.s2),
                            np.asarray(rec.c4), np.asarray(rec.s4))
    ci_cross, ci_single = ci[cross].mean(), ci[single].mean()
    # The 4-theta harmonic is higher-order/weaker, so the recovered crossing map
    # is elevated (~2x) but not the true (near-infinite, 2-theta-cancels) contrast.
    assert ci_cross > 1.6 * ci_single, (
        f"crossing not resolved: patch {ci_cross:.2f} vs single {ci_single:.2f}")

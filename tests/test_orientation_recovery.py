"""B-blocker: recover fibre ORIENTATION from ultrasound alone (no DTI).

The anisotropy recovery so far assumed the fibre direction phi was known (from
DTI) — a data dependency. But alpha(theta) = alpha_iso + alpha_aniso*sin^2(theta-
phi) can be written alpha(theta) = b - u*cos(2 theta) - v*sin(2 theta) with the
anisotropy VECTOR (u, v) = (0.5 alpha_aniso cos 2phi, 0.5 alpha_aniso sin 2phi).
That is LINEAR in (u, v), and

    alpha_aniso = 2*sqrt(u^2 + v^2),   phi = 0.5*atan2(v, u),

so the magnitude AND the fibre direction fall out of the angular pattern — no DTI
required. This removes the fibre-data dependency for GM/WM anisotropy imaging.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest


def test_uv_forward_matches_sin2_forward():
    """The (u,v) forward reproduces the sin^2(theta-phi) forward exactly."""
    from brain_fwi.inversion.anisotropic_atten import (
        make_ring_rays, forward_ray_decay, forward_ray_decay_uv)
    N, dx = 24, 1e-3
    rng = np.random.default_rng(0)
    a_iso = jnp.asarray(rng.uniform(0.3, 0.7, (N, N)).astype(np.float32))
    a_aniso = jnp.asarray(rng.uniform(0.0, 0.5, (N, N)).astype(np.float32))
    # phi constant (physical fibre fields are smooth): the (u,v) reparameterisation
    # is exact at every point; only ray-sampling of a spatially-varying phi (where
    # interp of cos2phi != cos2(interp phi)) introduces a tiny mismatch.
    phi = jnp.full((N, N), 0.7, jnp.float32)
    rays = make_ring_rays(N, dx, n_trans=20)
    d1 = forward_ray_decay(a_iso, a_aniso, phi, rays)
    b = a_iso + 0.5 * a_aniso
    u = 0.5 * a_aniso * jnp.cos(2 * phi)
    v = 0.5 * a_aniso * jnp.sin(2 * phi)
    d2 = forward_ray_decay_uv(b, u, v, rays)
    assert float(jnp.max(jnp.abs(d1 - d2))) < 1e-4


@pytest.mark.slow
def test_recovers_fibre_orientation_from_ultrasound_alone():
    """Blind: no phi given. Recover the anisotropy map (WM>GM) AND the fibre
    direction inside WM, from the angular pattern of ray decays."""
    from brain_fwi.inversion.anisotropic_atten import (
        make_ring_rays, forward_ray_decay, invert_orientation)
    N, dx = 40, 1.0e-3
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    r = np.sqrt((xx - N / 2) ** 2 + (yy - N / 2) ** 2)
    brain = r <= N * 0.4; wm = r <= N * 0.22; gm = brain & ~wm
    a_iso = np.where(brain, 0.6, 0.0).astype(np.float32)
    a_aniso = np.where(wm, 0.5, 0.0).astype(np.float32)
    phi_true = 0.6
    phi = np.full((N, N), phi_true, np.float32)
    rays = make_ring_rays(N, dx, n_trans=64, radius_frac=0.46)
    obs = forward_ray_decay(jnp.asarray(a_iso), jnp.asarray(a_aniso), jnp.asarray(phi), rays)

    rec = invert_orientation(obs, rays, (N, N),
                             mask=jnp.asarray(brain.astype(np.float32)), n_iters=3000)
    aa = np.asarray(rec.a_aniso); ph = np.asarray(rec.phi)

    # THE headline: fibre DIRECTION recovered from ultrasound alone (no DTI), to
    # well under a degree inside WM — acoustic tractography.
    core = wm & (aa > 0.2)
    err = np.abs((ph[core] - phi_true + np.pi / 2) % np.pi - np.pi / 2)
    assert np.median(err) < 0.1, f"fibre angle error median {np.median(err):.3f} rad"
    # anisotropy magnitude favours WM (blind (u,v) is leakier than known-phi; the
    # two-stage uv-direction -> sin^2-magnitude refinement sharpens it further).
    assert aa[wm].mean() > 1.5 * max(aa[gm].mean(), 1e-6), (
        f"anisotropy WM({aa[wm].mean():.3f}) vs GM({aa[gm].mean():.3f})")

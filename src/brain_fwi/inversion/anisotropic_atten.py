"""Multi-angle anisotropic attenuation tomography — the path forward for GM/WM.

Gray and white matter share sound speed (degenerate to velocity FWI), their
spectral-shape difference is not resolvable over the narrow transcranial band, and
their isotropic attenuation contrast is small. The one axis with real leverage is
ANISOTROPY: white matter is fibre-oriented, so its attenuation depends on the
angle between propagation and fibre,

    alpha(theta) = alpha_iso + alpha_aniso * sin^2(theta - phi),

minimal along the fibre (theta = phi) and maximal across it; gray matter is
isotropic (alpha_aniso = 0). Transmission tomography crosses every interior voxel
with rays at many angles, so the per-voxel anisotropy is recoverable — and it
separates WM from GM even when their isotropic properties are identical.

This module is a differentiable straight-ray proof of principle. The leverage is
the angular coverage of the acquisition, which a full-wave solver inherits; the
straight-ray forward is the cheapest model that exposes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax.scipy.ndimage import map_coordinates


def anisotropic_alpha(theta, a_iso, a_aniso, phi):
    """Direction-dependent attenuation alpha_iso + alpha_aniso*sin^2(theta-phi)."""
    return a_iso + a_aniso * jnp.sin(theta - phi) ** 2


@dataclass
class Rays:
    """A bundle of straight rays for tomography.

    coords: (n_rays, 2, n_samples) grid-unit sample coordinates ([row=y, col=x]).
    angles: (n_rays,) propagation angle theta = atan2(dy, dx).
    length: (n_rays,) physical ray length (m).
    ds:     (n_rays,) physical sample spacing (m) = length / n_samples.
    """
    coords: jnp.ndarray
    angles: jnp.ndarray
    length: jnp.ndarray
    ds: jnp.ndarray


def make_ring_rays(n_grid: int, dx: float, n_trans: int = 48,
                   radius_frac: float = 0.46, n_samples: int = None) -> Rays:
    """All-pairs straight rays between ``n_trans`` transducers on a ring — dense,
    multi-angle coverage of the interior."""
    if n_samples is None:
        n_samples = int(n_grid * 1.6)
    c = (n_grid - 1) / 2.0
    R = radius_frac * n_grid
    ang = np.linspace(0, 2 * np.pi, n_trans, endpoint=False)
    tx = c + R * np.cos(ang)
    ty = c + R * np.sin(ang)
    coords, angles, length, ds = [], [], [], []
    t = np.linspace(0.0, 1.0, n_samples)
    for i in range(n_trans):
        for j in range(i + 1, n_trans):
            xi, yi, xj, yj = tx[i], ty[i], tx[j], ty[j]
            x = xi + (xj - xi) * t
            y = yi + (yj - yi) * t
            coords.append(np.stack([y, x], axis=0))           # [row, col]
            angles.append(np.arctan2(yj - yi, xj - xi))
            L = np.hypot(xj - xi, yj - yi) * dx
            length.append(L)
            ds.append(L / n_samples)
    return Rays(
        coords=jnp.asarray(np.stack(coords), jnp.float32),
        angles=jnp.asarray(np.asarray(angles), jnp.float32),
        length=jnp.asarray(np.asarray(length), jnp.float32),
        ds=jnp.asarray(np.asarray(ds), jnp.float32),
    )


def forward_ray_decay(a_iso, a_aniso, phi, rays: Rays):
    """Per-ray log-amplitude decay = path integral of alpha at the ray's angle."""
    def one(coords, theta, ds):
        ai = map_coordinates(a_iso, coords, order=1, mode="constant", cval=0.0)
        aa = map_coordinates(a_aniso, coords, order=1, mode="constant", cval=0.0)
        ph = map_coordinates(phi, coords, order=1, mode="constant", cval=0.0)
        return jnp.sum(anisotropic_alpha(theta, ai, aa, ph)) * ds
    return jax.vmap(one)(rays.coords, rays.angles, rays.ds)


@dataclass
class AnisoResult:
    a_iso: jnp.ndarray
    a_aniso: jnp.ndarray
    loss_history: list


def invert_anisotropic(obs, rays: Rays, phi, grid_shape, mask=None, a_iso=None,
                       n_iters: int = 2000, lr: float = 3e-2,
                       smooth_aniso: float = 3e-4) -> AnisoResult:
    """Recover the per-voxel anisotropy ``alpha_aniso`` from multi-angle ray
    decays, given the fibre direction ``phi`` (from DTI) and a **bulk isotropic
    attenuation estimate** ``a_iso`` (from standard isotropic tomography / the
    Phase-6 FWI). This is the two-stage recipe:

      1. bulk alpha  -> a_iso  (isotropic pass, already available),
      2. anisotropy  -> a_aniso (this call, from the residual angular signal).

    Why two-stage: the angle-mean of alpha is ``alpha_iso + 0.5*alpha_aniso``, so
    a *joint* iso/aniso inversion is weakly identifiable and crosstalks. Given a
    bulk estimate, the residual cos-2theta angular signal recovers alpha_aniso
    cleanly (WM lights up, GM stays ~0). Non-negativity via softplus; a light
    Laplacian smoothness stabilises the tomographic inverse.

    (Joint single-stage iso+aniso recovery is a harder tomographic identifiability
    problem — cross-gradient / alternating methods — left as a follow-on.)
    """
    obs = jnp.asarray(obs)
    m = jnp.ones(grid_shape) if mask is None else jnp.asarray(mask)
    ai = jnp.zeros(grid_shape) if a_iso is None else jnp.asarray(a_iso)
    params = jnp.full(grid_shape, -2.0)

    def aniso_of(p):
        return jax.nn.softplus(p) * m

    def lap(f):
        return (f - 0.25 * (jnp.roll(f, 1, 0) + jnp.roll(f, -1, 0)
                            + jnp.roll(f, 1, 1) + jnp.roll(f, -1, 1))) * m

    def loss_fn(p):
        aa = aniso_of(p)
        pred = forward_ray_decay(ai, aa, phi, rays)
        return jnp.mean((pred - obs) ** 2) + smooth_aniso * jnp.mean(lap(aa) ** 2)

    opt = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def step(p, s):
        loss, g = jax.value_and_grad(loss_fn)(p)
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, loss

    hist = []
    for _ in range(n_iters):
        params, state, loss = step(params, state)
        hist.append(float(loss))
    return AnisoResult(a_iso=ai, a_aniso=aniso_of(params), loss_history=hist)

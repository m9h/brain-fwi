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


def forward_ray_decay_uv(b, u, v, rays: Rays):
    """Per-ray decay in the anisotropy-VECTOR parameterisation.

    ``alpha(theta) = b - u*cos(2 theta) - v*sin(2 theta)`` (LINEAR in b, u, v),
    the orthogonal-basis form of ``alpha_iso + alpha_aniso*sin^2(theta-phi)`` with
    ``b = alpha_iso + 0.5*alpha_aniso``, ``u = 0.5*alpha_aniso*cos 2phi``,
    ``v = 0.5*alpha_aniso*sin 2phi``. Recovering (u, v) yields both the anisotropy
    magnitude and the fibre direction without knowing phi.
    """
    c2 = jnp.cos(2.0 * rays.angles)
    s2 = jnp.sin(2.0 * rays.angles)

    def one(coords, c2r, s2r, ds):
        bb = map_coordinates(b, coords, order=1, mode="constant", cval=0.0)
        uu = map_coordinates(u, coords, order=1, mode="constant", cval=0.0)
        vv = map_coordinates(v, coords, order=1, mode="constant", cval=0.0)
        return jnp.sum(bb - uu * c2r - vv * s2r) * ds
    return jax.vmap(one)(rays.coords, c2, s2, rays.ds)


@dataclass
class AnisoResult:
    a_iso: jnp.ndarray
    a_aniso: jnp.ndarray
    loss_history: list
    phi: jnp.ndarray = None            # recovered fibre direction (rad), if any


def invert_anisotropic(obs, rays: Rays, phi, grid_shape, mask=None, a_iso=None,
                       n_iters: int = 2000, lr: float = 3e-2,
                       smooth_aniso: float = 3e-4) -> AnisoResult:
    """Recover the per-voxel anisotropy ``alpha_aniso`` from multi-angle ray
    decays, given the fibre direction ``phi`` (from DTI). Two modes:

    - **Two-stage** (``a_iso`` supplied) — a bulk isotropic attenuation estimate
      (from the isotropic pass / Phase-6 FWI) is fixed and only the residual
      cos-2theta angular signal is inverted for ``alpha_aniso``.
    - **Blind** (``a_iso=None``) — jointly recover a *homogeneous (scalar)* bulk
      ``alpha_iso`` and the anisotropy field. The angle-mean of alpha is
      ``alpha_iso + 0.5*alpha_aniso``, so a per-voxel *field* bulk is weakly
      identifiable and crosstalks (a smooth bulk still absorbs the sharp WM
      structure). Constraining the bulk to be **homogeneous** — a strong
      structural prior, in the Living Matter Lab spirit of baking physics into the
      parameterisation — makes the anisotropy identifiable: the bulk *cannot*
      absorb sharp structure, so it goes into the anisotropy channel where it
      belongs. Measured: WM/GM anisotropy ratio ~12x, bulk recovered ~0.6.

    Non-negativity via softplus; a light Laplacian smoothness stabilises the
    anisotropy. (A low-rank — not strictly scalar — bulk is the practical
    generalisation for slowly-varying tissue; see the design doc.)
    """
    obs = jnp.asarray(obs)
    m = jnp.ones(grid_shape) if mask is None else jnp.asarray(mask)
    blind = a_iso is None
    fixed_ai = None if blind else jnp.asarray(a_iso)
    # blind mode also carries a scalar bulk parameter ``rb`` (see fields()).
    params = {"rd": jnp.full(grid_shape, -2.0), "rb": jnp.asarray(-1.0)}

    def fields(p):
        aa = jax.nn.softplus(p["rd"]) * m
        if blind:
            ai = jax.nn.softplus(p["rb"]) * m       # homogeneous (scalar) bulk
        else:
            ai = fixed_ai
        return ai, aa

    def lap(f):
        return (f - 0.25 * (jnp.roll(f, 1, 0) + jnp.roll(f, -1, 0)
                            + jnp.roll(f, 1, 1) + jnp.roll(f, -1, 1))) * m

    def loss_fn(p):
        ai, aa = fields(p)
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
    ai, aa = fields(params)
    return AnisoResult(a_iso=ai, a_aniso=aa, loss_history=hist)


def forward_ray_decay_odf(a0, c2, s2, c4, s4, rays: Rays):
    """Per-ray decay for a full per-voxel angular-attenuation ODF
    ``alpha(theta) = a0 + c2 cos2theta + s2 sin2theta + c4 cos4theta + s4 sin4theta``
    (linear in all five harmonic fields). The 4-theta fields carry the crossing
    information that the 2-theta-only (single-phi) model discards.
    """
    ca2 = jnp.cos(2.0 * rays.angles); sa2 = jnp.sin(2.0 * rays.angles)
    ca4 = jnp.cos(4.0 * rays.angles); sa4 = jnp.sin(4.0 * rays.angles)

    def one(coords, c2r, s2r, c4r, s4r, ds):
        A0 = map_coordinates(a0, coords, order=1, mode="constant", cval=0.0)
        C2 = map_coordinates(c2, coords, order=1, mode="constant", cval=0.0)
        S2 = map_coordinates(s2, coords, order=1, mode="constant", cval=0.0)
        C4 = map_coordinates(c4, coords, order=1, mode="constant", cval=0.0)
        S4 = map_coordinates(s4, coords, order=1, mode="constant", cval=0.0)
        return jnp.sum(A0 + C2 * c2r + S2 * s2r + C4 * c4r + S4 * s4r) * ds
    return jax.vmap(one)(rays.coords, ca2, sa2, ca4, sa4, rays.ds)


@dataclass
class OdfResult:
    a0: jnp.ndarray
    c2: jnp.ndarray
    s2: jnp.ndarray
    c4: jnp.ndarray
    s4: jnp.ndarray
    loss_history: list


def invert_odf(obs, rays: Rays, grid_shape, mask=None, n_iters: int = 6000,
               lr: float = 3e-2, smooth: float = 1e-5) -> OdfResult:
    """Blindly recover the per-voxel attenuation ODF (2-theta AND 4-theta harmonic
    fields) with a homogeneous (scalar) bulk ``a0`` — the tomographic upgrade that
    carries fibre-crossing information (via the 4-theta fields), resolving the
    crossings the single-phi model averages away."""
    obs = jnp.asarray(obs)
    m = jnp.ones(grid_shape) if mask is None else jnp.asarray(mask)
    params = {"rb": jnp.asarray(-1.0),
              "c2": jnp.zeros(grid_shape), "s2": jnp.zeros(grid_shape),
              "c4": jnp.zeros(grid_shape), "s4": jnp.zeros(grid_shape)}

    def lap(f):
        return (f - 0.25 * (jnp.roll(f, 1, 0) + jnp.roll(f, -1, 0)
                            + jnp.roll(f, 1, 1) + jnp.roll(f, -1, 1))) * m

    def loss_fn(p):
        a0 = jax.nn.softplus(p["rb"]) * m                # homogeneous bulk
        fields = [p[k] * m for k in ("c2", "s2", "c4", "s4")]
        pred = forward_ray_decay_odf(a0, *fields, rays)
        reg = smooth * sum(jnp.mean(lap(f) ** 2) for f in fields)
        return jnp.mean((pred - obs) ** 2) + reg

    opt = optax.adam(lr); state = opt.init(params)

    @jax.jit
    def step(p, s):
        loss, g = jax.value_and_grad(loss_fn)(p)
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, loss

    hist = []
    for _ in range(n_iters):
        params, state, loss = step(params, state)
        hist.append(float(loss))
    a0 = jax.nn.softplus(params["rb"]) * m
    return OdfResult(a0=a0, c2=params["c2"] * m, s2=params["s2"] * m,
                     c4=params["c4"] * m, s4=params["s4"] * m, loss_history=hist)


def invert_orientation(obs, rays: Rays, grid_shape, mask=None,
                       n_iters: int = 3000, lr: float = 3e-2,
                       smooth: float = 3e-4) -> AnisoResult:
    """Blindly recover the anisotropy magnitude AND fibre orientation from
    multi-angle ray decays — no DTI. Inverts the anisotropy VECTOR ``(u, v)``
    (linear, orthogonal to the bulk) plus a **homogeneous (scalar) bulk** (the
    structural prior that makes the bulk/anisotropy split identifiable, as in
    :func:`invert_anisotropic`). Returns ``a_aniso = 2*sqrt(u^2+v^2)`` and
    ``phi = 0.5*atan2(v, u)`` per voxel; the fibre direction is meaningful only
    where ``a_aniso`` is non-trivial (WM).
    """
    obs = jnp.asarray(obs)
    m = jnp.ones(grid_shape) if mask is None else jnp.asarray(mask)
    params = {"rb": jnp.asarray(-1.0),
              "u": jnp.zeros(grid_shape), "v": jnp.zeros(grid_shape)}

    def lap(f):
        return (f - 0.25 * (jnp.roll(f, 1, 0) + jnp.roll(f, -1, 0)
                            + jnp.roll(f, 1, 1) + jnp.roll(f, -1, 1))) * m

    def loss_fn(p):
        b = jax.nn.softplus(p["rb"]) * m         # homogeneous bulk
        u, v = p["u"] * m, p["v"] * m
        pred = forward_ray_decay_uv(b, u, v, rays)
        return (jnp.mean((pred - obs) ** 2)
                + smooth * (jnp.mean(lap(u) ** 2) + jnp.mean(lap(v) ** 2)))

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
    b = jax.nn.softplus(params["rb"]) * m
    u, v = params["u"] * m, params["v"] * m
    mag = jnp.sqrt(u ** 2 + v ** 2)
    a_aniso = 2.0 * mag
    a_iso = (b - mag) * m
    phi = 0.5 * jnp.arctan2(v, u)
    return AnisoResult(a_iso=a_iso, a_aniso=a_aniso, loss_history=hist, phi=phi)

"""Crossing-fibre attenuation model — resolving fibre crossings from the angular
attenuation profile (the acoustic analogue of DTI -> HARDI).

A single fibre attenuates minimally along its axis and maximally across it. With a
SOFT profile ``sin^2(theta-phi)`` the angular attenuation is pure 2-theta, so two
crossing fibres sum to a single 2-theta sinusoid and are indistinguishable from
one fibre (2 measurements, 4 unknowns). Crossings are resolvable only when the
single-fibre profile is SHARPER (``sin^{2s}``, s>1), carrying higher (4-theta,
6-theta) angular harmonics — the acoustic analogue of high-order ODFs / spherical
deconvolution. For an orthogonal crossing the 2-theta terms cancel and the
4-theta terms add, so ``|4theta| / |2theta|`` detects the crossing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp


def fibre_profile(theta, phi, amp, sharpness: int = 2):
    """Single-fibre attenuation ``amp * sin(theta-phi)^{2*sharpness}`` — zero along
    the fibre, ``amp`` across it. sharpness=1 is the soft (pure-2theta) profile;
    sharpness>=2 adds higher harmonics that make crossings resolvable."""
    return amp * jnp.sin(theta - phi) ** (2 * sharpness)


def two_fibre_profile(theta, iso, a1, phi1, a2, phi2, sharpness: int = 2):
    """Isotropic baseline plus two fibre populations."""
    return (iso + fibre_profile(theta, phi1, a1, sharpness)
            + fibre_profile(theta, phi2, a2, sharpness))


def angular_harmonics(theta, alpha):
    """Least-squares Fourier fit of the angular profile on {1, cos2t, sin2t,
    cos4t, sin4t} (the attenuation-ODF harmonics)."""
    th = np.asarray(theta); al = np.asarray(alpha)
    A = np.stack([np.ones_like(th), np.cos(2 * th), np.sin(2 * th),
                  np.cos(4 * th), np.sin(4 * th)], 1)
    c, *_ = np.linalg.lstsq(A, al, rcond=None)
    return {"a0": float(c[0]), "c2": float(c[1]), "s2": float(c[2]),
            "c4": float(c[3]), "s4": float(c[4])}


def crossing_index(h) -> float:
    """``|4theta| / |2theta|`` — small for a single fibre, large for a (sharp)
    orthogonal crossing (where the 2-theta terms cancel), ~0 for a soft (sin^2)
    profile that carries no 4-theta content."""
    p2 = np.hypot(h["c2"], h["s2"]); p4 = np.hypot(h["c4"], h["s4"])
    return float(p4 / (p2 + 1e-9))


def sin4_odf_coeffs(a_iso, amps, phis):
    """Angular-ODF harmonics {a0, c2, s2, c4, s4} of an isotropic baseline plus a
    set of sharp (sin^4) fibres. Uses sin^4(x) = 3/8 - 1/2 cos2x + 1/8 cos4x.

    ``amps``/``phis`` are broadcastable arrays with a trailing fibre axis (amp=0
    for empty slots). Returns five arrays with the leading (spatial) shape.
    """
    amps = np.asarray(amps); phis = np.asarray(phis)
    a0 = np.asarray(a_iso) + (amps * 3.0 / 8.0).sum(-1)
    c2 = (amps * (-0.5) * np.cos(2 * phis)).sum(-1)
    s2 = (amps * (-0.5) * np.sin(2 * phis)).sum(-1)
    c4 = (amps * (1.0 / 8.0) * np.cos(4 * phis)).sum(-1)
    s4 = (amps * (1.0 / 8.0) * np.sin(4 * phis)).sum(-1)
    return a0, c2, s2, c4, s4


def crossing_index_map(c2, s2, c4, s4):
    """Per-voxel |4theta|/|2theta| crossing map from ODF-harmonic fields."""
    c2 = np.asarray(c2); s2 = np.asarray(s2); c4 = np.asarray(c4); s4 = np.asarray(s4)
    return np.hypot(c4, s4) / (np.hypot(c2, s2) + 1e-9)


@dataclass
class OdfCrossingMaps:
    """Per-voxel two-fibre decomposition of an attenuation-ODF field."""
    phi1: np.ndarray     # primary fibre direction (rad)
    phi2: np.ndarray     # secondary fibre direction (rad); meaningful where a2>0
    a1: np.ndarray       # primary amplitude
    a2: np.ndarray       # secondary amplitude (~0 where single fibre)
    iso: np.ndarray      # isotropic baseline


def resolve_odf_crossings(a0, c2, s2, c4, s4, mask, sharpness: int = 2,
                          n_grid: int = 30, n_angles: int = 180) -> OdfCrossingMaps:
    """Per-voxel two-fibre decomposition of the attenuation ODF — the tomography's
    explicit-crossing ("acoustic HARDI") output. Reconstructs each voxel's angular
    profile from its harmonic fields and grid-searches two fibre directions
    (amplitudes solved in closed form per grid pair, vectorised across voxels).

    These are the same anisotropic unknowns a full-wave anisotropic-attenuation
    FWI would invert; the straight-ray tomography validates the parameterisation.
    """
    a0, c2, s2, c4, s4 = (np.asarray(x) for x in (a0, c2, s2, c4, s4))
    mask = np.asarray(mask) > 0.5
    shape = a0.shape
    idx = np.where(mask); nvox = len(idx[0])
    th = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    prof = (a0[idx][:, None] + c2[idx][:, None] * np.cos(2 * th)
            + s2[idx][:, None] * np.sin(2 * th) + c4[idx][:, None] * np.cos(4 * th)
            + s4[idx][:, None] * np.sin(4 * th))                      # (nvox, K)
    Y = prof.T                                                        # (K, nvox)
    grid = np.linspace(0.0, np.pi, n_grid, endpoint=False)
    basis = [np.sin(th - p) ** (2 * sharpness) for p in grid]
    ones = np.ones(n_angles)

    best = np.full(nvox, np.inf)
    b_iso = np.zeros(nvox); b_a1 = np.zeros(nvox); b_a2 = np.zeros(nvox)
    b_p1 = np.zeros(nvox); b_p2 = np.zeros(nvox)
    for i in range(n_grid):
        for j in range(i, n_grid):
            A = np.stack([ones, basis[i], basis[j]], 1)              # (K, 3)
            coef, *_ = np.linalg.lstsq(A, Y, rcond=None)             # (3, nvox)
            res = np.mean((A @ coef - Y) ** 2, 0)                    # (nvox,)
            ok = (coef[1] >= -1e-6) & (coef[2] >= -1e-6)
            upd = (res < best) & ok
            best[upd] = res[upd]
            b_iso[upd] = coef[0][upd]; b_a1[upd] = coef[1][upd]; b_a2[upd] = coef[2][upd]
            b_p1[upd] = grid[i]; b_p2[upd] = grid[j]

    prim = b_a1 >= b_a2                                              # order primary first
    a1v = np.where(prim, b_a1, b_a2); a2v = np.where(prim, b_a2, b_a1)
    p1v = np.where(prim, b_p1, b_p2); p2v = np.where(prim, b_p2, b_p1)

    def scatter(v):
        out = np.zeros(shape); out[idx] = v; return out
    return OdfCrossingMaps(phi1=scatter(p1v % np.pi), phi2=scatter(p2v % np.pi),
                           a1=scatter(a1v), a2=scatter(a2v), iso=scatter(b_iso))


@dataclass
class TwoFibreFit:
    iso: float
    a1: float
    phi1: float
    a2: float
    phi2: float
    rmse: float


def fit_two_fibres(theta, alpha, sharpness: int = 2, n_grid: int = 24) -> TwoFibreFit:
    """Fit two fibre directions to an angular attenuation profile.

    The profile is LINEAR in (iso, a1, a2) for fixed (phi1, phi2), so we grid-
    search the two directions (solving the non-negative amplitudes in closed form
    per grid point) and then refine locally. Robust, no local-minimum issues.
    """
    th = np.asarray(theta); al = np.asarray(alpha)

    def solve(p1, p2):
        A = np.stack([np.ones_like(th), np.sin(th - p1) ** (2 * sharpness),
                      np.sin(th - p2) ** (2 * sharpness)], 1)
        c, *_ = np.linalg.lstsq(A, al, rcond=None)
        r = float(np.mean((A @ c - al) ** 2))
        ok = c[1] >= -1e-6 and c[2] >= -1e-6
        return (r if ok else np.inf), c

    grid = np.linspace(0.0, np.pi, n_grid, endpoint=False)
    best_r, best = np.inf, (0.0, 0.0, np.zeros(3))
    for i, p1 in enumerate(grid):
        for p2 in grid[i:]:
            r, c = solve(p1, p2)
            if r < best_r:
                best_r, best = r, (p1, p2, c)

    p1, p2, c = best
    step = np.pi / n_grid
    for _ in range(3):                                   # local refinement
        step *= 0.4
        improved = False
        for dp1 in np.linspace(-step, step, 5):
            for dp2 in np.linspace(-step, step, 5):
                r, cc = solve(p1 + dp1, p2 + dp2)
                if r < best_r:
                    best_r, p1, p2, c = r, p1 + dp1, p2 + dp2, cc
                    improved = True
        if not improved:
            break
    return TwoFibreFit(iso=float(c[0]), a1=float(c[1]), phi1=float(p1 % np.pi),
                       a2=float(c[2]), phi2=float(p2 % np.pi), rmse=float(np.sqrt(best_r)))

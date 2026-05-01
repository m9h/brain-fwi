"""Calibration diagnostics for syn-vs-exp trace comparisons."""

from __future__ import annotations

import numpy as np


def _norm_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def best_lag_correlation(
    a: np.ndarray, b: np.ndarray, max_lag: int,
) -> tuple[int, float]:
    """Lag k ∈ [-max_lag, +max_lag] that maximises normalised correlation
    between ``a`` and ``b`` after non-circular shift.

    Sign convention: positive k means ``b`` is delayed relative to
    ``a`` by k samples (we slice ``a`` from index k).

    Returns:
        (k, corr) — lag in samples and the normalised correlation at
        that lag.
    """
    n = len(a)
    best_k = 0
    best_c = -2.0
    for k in range(-max_lag, max_lag + 1):
        # Convention: +k means b lags a by k samples, i.e. b[k+i] ≡ a[i].
        if k > 0:
            a_s, b_s = a[: n - k], b[k:]
        elif k < 0:
            a_s, b_s = a[-k:], b[: n + k]
        else:
            a_s, b_s = a, b
        c = _norm_corr(a_s, b_s)
        if c > best_c:
            best_c, best_k = c, k
    return best_k, best_c


def perp_distance_xz(
    src: np.ndarray, rcv: np.ndarray, centre: np.ndarray,
) -> float:
    """Perpendicular distance from ``centre`` to the line ``src -> rcv``,
    projected onto the xz plane (y is ignored).

    Used to stratify ICL dual-probe traces by how much phantom they
    traverse: small distance ⇒ ray passes near the imaging-target
    centre; large distance ⇒ tangential.
    """
    s = np.asarray([src[0], src[2]], dtype=np.float64)
    r = np.asarray([rcv[0], rcv[2]], dtype=np.float64)
    c = np.asarray([centre[0], centre[2]], dtype=np.float64)
    line = r - s
    L = float(np.linalg.norm(line))
    if L < 1e-12:
        return float(np.linalg.norm(c - s))
    cross = float(line[0] * (c[1] - s[1]) - line[1] * (c[0] - s[0]))
    return abs(cross) / L

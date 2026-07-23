"""Automated constitutive discovery for acoustic attenuation alpha(omega).

Mirrors the Living Matter Lab CANN method (Linka & Kuhl, CMAME 403:115731, 2023;
McCulloch, St. Pierre, Linka & Kuhl, IJNME 125:e7481, 2024 for the L0 sparse
regression; Linka, St. Pierre & Kuhl, Acta Biomater. 160:134, 2023 for the brain
GM/WM result), adapted from hyperelasticity to frequency-domain dissipation:

  - a small OVER-COMPLETE library of building blocks, each **non-negative**
    (dissipation >= 0) and **DC-vanishing** (alpha(0)=0),
  - the model is **LINEAR in interpretable outer weights** (as in the
    principal-stretch CANN), so each restricted fit is a convex non-negative
    least squares with a unique optimum — which makes L0 discovery tractable,
  - **L0 subset selection** (their preferred penalty: counts terms, does not
    bias the retained magnitudes) via greedy forward selection.

The one constraint their hyperelastic framework never needed is **causality**
(Kramers-Kronig linking alpha(omega) to dispersion c(omega)); see :mod:`.kk`.
Building blocks here (power laws + Debye/relaxation) have analytic KK partners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import jax.numpy as jnp


def alpha_basis_library(
    omega: jnp.ndarray,
    exponents: Sequence[float] = (1.0, 1.1, 1.3, 1.5, 2.0),
    taus: Optional[Sequence[float]] = None,
    omega_scale: Optional[float] = None,
) -> Tuple[jnp.ndarray, List[str]]:
    """Over-complete non-negative, DC-vanishing basis for alpha(omega).

    Columns:
      - power laws ``|omega/omega_scale|^y`` for each ``y`` in ``exponents``
        (viscous/tissue regime; classical absorption models are sparse corners),
      - Debye/relaxation losses ``omega*tau / (1 + (omega*tau)^2)`` for each
        ``tau`` (the frequency-domain image of the iCANN Prony spectrum;
        analytic Kramers-Kronig partner).

    Each column is normalised to unit max over ``omega`` (Kuhl-style mode
    normalisation → better-conditioned discovery). All columns vanish at
    ``omega=0``.

    Returns:
        ``(Phi, names)`` with ``Phi`` shape ``(len(omega), n_terms)``.
    """
    omega = jnp.asarray(omega)
    a = jnp.abs(omega)
    if omega_scale is None:
        omega_scale = float(jnp.max(a)) if float(jnp.max(a)) > 0 else 1.0
    if taus is None:
        # relaxation times spanning the band (omega*tau = 1 near band lo/mid/hi)
        lo = float(jnp.min(a[a > 0])) if bool(jnp.any(a > 0)) else 1.0
        hi = float(jnp.max(a)) if float(jnp.max(a)) > 0 else 1.0
        mid = np.sqrt(lo * hi)
        taus = tuple(1.0 / v for v in (lo, mid, hi))

    cols: List[jnp.ndarray] = []
    names: List[str] = []
    x = a / omega_scale
    for y in exponents:
        cols.append(x ** y)
        names.append(f"pow_{y:g}")
    for tau in taus:
        wt = a * tau
        cols.append(wt / (1.0 + wt ** 2))
        names.append(f"debye_wt{omega_scale * tau:.2g}")

    Phi = jnp.stack(cols, axis=-1)
    # normalise each column to unit max over the band (guard all-zero columns)
    colmax = jnp.max(jnp.abs(Phi), axis=0, keepdims=True)
    Phi = Phi / jnp.where(colmax > 1e-30, colmax, 1.0)
    return Phi, names


def alpha_basis_library_anisotropic(
    omega: jnp.ndarray,
    theta: jnp.ndarray,
    exponents: Sequence[float] = (1.0, 1.1, 1.3, 1.5, 2.0),
    taus: Optional[Sequence[float]] = None,
    omega_scale: Optional[float] = None,
) -> Tuple[jnp.ndarray, List[str]]:
    """Direction-dependent attenuation library alpha(omega, theta).

    White matter is fibre-oriented, so its attenuation depends on the angle
    ``theta`` between propagation and fibre direction; gray matter is isotropic.
    This is the CANN-native GM/WM *shape* discriminator — the structural-tensor
    (I4/I5) analogue Kuhl did not activate in the isotropic brain study.

    Each frequency building block (:func:`alpha_basis_library`) is multiplied by
    each **non-negative** angular factor — isotropic ``1`` and anisotropic
    ``sin^2(theta)`` (extra attenuation across fibres) — so non-negativity and
    DC-vanishing are preserved and ``alpha >= 0`` for any non-negative weights.
    The design matrix is over the flattened ``(omega, theta)`` grid in
    ``indexing="ij"`` order (frequency-major).

    Returns:
        ``(Phi, names)`` with ``Phi`` shape ``(len(omega)*len(theta), n_terms)``
        and names like ``"pow_1.3xsin2"``.
    """
    theta = jnp.asarray(theta)
    Phi_f, fnames = alpha_basis_library(omega, exponents, taus, omega_scale)  # (nf, k)
    angular = (("iso", jnp.ones_like(theta)), ("sin2", jnp.sin(theta) ** 2))

    cols: List[jnp.ndarray] = []
    names: List[str] = []
    for j, fn in enumerate(fnames):
        for gname, gvals in angular:
            col = Phi_f[:, j][:, None] * gvals[None, :]      # (nf, na)
            cols.append(col.reshape(-1))
            names.append(f"{fn}x{gname}")
    return jnp.stack(cols, axis=-1), names


def _nnls(A: np.ndarray, b: np.ndarray, iters: int = 800) -> np.ndarray:
    """Non-negative least squares via projected gradient (small dense systems)."""
    A = np.asarray(A, float); b = np.asarray(b, float)
    if A.shape[1] == 0:
        return np.zeros(0)
    AtA = A.T @ A; Atb = A.T @ b
    L = float(np.linalg.norm(AtA, 2)) + 1e-12
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        w = np.maximum(0.0, w - (AtA @ w - Atb) / L)
    return w


def discover_tissue_alpha_law(
    band_center_freqs_hz: Sequence[float],
    alpha_at_band: Sequence[float],
    exponents: Sequence[float] = (1.0, 1.1, 1.3, 1.5, 2.0),
    taus: Optional[Sequence[float]] = None,
    l0_penalty: float = 1e-3,
) -> "DiscoveryResult":
    """Discover a tissue's alpha(omega) law from per-band FWI attenuation.

    The FWI -> CANN bridge: multi-band FWI recovers attenuation at several centre
    frequencies (``alpha_at_band[k]`` = recovered attenuation, dB/cm/MHz, for band
    ``k`` at ``band_center_freqs_hz[k]``); this assembles them into alpha(omega)
    samples and runs the L0 constitutive discovery (Kuhl's "measure at many
    conditions, then discover" workflow). No CANN evaluation in the forward is
    required — the frequency dependence comes from the *band-wise* estimates.

    Returns a :class:`DiscoveryResult`; ``sum(weights)`` is a convenient scalar
    attenuation-magnitude proxy for ranking tissues (e.g. WM vs GM).
    """
    f = np.asarray(band_center_freqs_hz, float)
    a = np.asarray(alpha_at_band, float)
    omega = jnp.asarray(2.0 * np.pi * f)
    Phi, names = alpha_basis_library(omega, exponents=exponents, taus=taus)
    return discover_alpha_law(jnp.asarray(a), Phi, names, l0_penalty)


@dataclass
class DiscoveryResult:
    """Outcome of :func:`discover_alpha_law`."""
    weights: np.ndarray          # non-negative weight per library term (0 if unused)
    active_terms: List[str]      # names of the selected building blocks
    n_terms: int
    rel_rmse: float              # ||fit - target|| / ||target|| over the band


def discover_alpha_law(
    alpha_true: jnp.ndarray,
    Phi: jnp.ndarray,
    names: Sequence[str],
    l0_penalty: float = 1e-3,
) -> DiscoveryResult:
    """L0 greedy forward selection of a minimal non-negative alpha(omega) law.

    Adds, one at a time, the building block that most reduces the non-negative
    least-squares misfit, stopping when the next term's improvement no longer
    exceeds ``l0_penalty * ||alpha_true||^2`` (the L0 cost of one extra term).
    This is the tractable, magnitude-unbiased discovery of Kuhl's L0 sparse
    regression, exploiting that the model is linear in the non-negative weights.

    Returns the fitted weights on the selected support (zeros elsewhere).
    """
    b = np.asarray(alpha_true, float)
    Phi = np.asarray(Phi, float)
    names = list(names)
    ntot = float(b @ b) + 1e-30

    support: List[int] = []
    remaining = list(range(Phi.shape[1]))
    best_misfit = ntot                # empty model predicts 0
    best_w = np.zeros(0)

    while remaining:
        cand_j, cand_m, cand_w = None, None, None
        for j in remaining:
            supp = support + [j]
            w = _nnls(Phi[:, supp], b)
            m = float(np.sum((Phi[:, supp] @ w - b) ** 2))
            if cand_m is None or m < cand_m:
                cand_j, cand_m, cand_w = j, m, w
        if best_misfit - cand_m > l0_penalty * ntot:
            support.append(cand_j); remaining.remove(cand_j)
            best_misfit, best_w = cand_m, cand_w
        else:
            break

    weights = np.zeros(Phi.shape[1])
    for idx, j in enumerate(support):
        weights[j] = best_w[idx]
    return DiscoveryResult(
        weights=weights,
        active_terms=[names[j] for j in support],
        n_terms=len(support),
        rel_rmse=float(np.sqrt(best_misfit / ntot)),
    )

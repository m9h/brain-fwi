"""Kuhl-style automated constitutive discovery for acoustic attenuation alpha(omega).

Mirrors the Living Matter Lab CANN method (Linka & Kuhl CMAME 2023; McCulloch et
al. IJNME 2024 for L0; Linka/StPierre/Kuhl Acta Biomat 2023 for the brain GM/WM
result): a small OVER-COMPLETE library of non-negative, DC-vanishing building
blocks, LINEAR in interpretable outer weights, sparsified by L0 to a minimal
interpretable law.

The headline test reproduces their brain finding in our domain: grey and white
matter select the SAME functional alpha(omega) law and differ only in MAGNITUDE.

One behaviour per test, each written failing first.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest


def _band(n=64):
    # FWI band 50-300 kHz as angular frequency.
    return jnp.linspace(2 * np.pi * 50e3, 2 * np.pi * 300e3, n)


def test_library_columns_are_dc_vanishing_and_shaped():
    """Every building block vanishes at omega=0 (no static attenuation) and the
    design matrix is (n_omega, n_terms)."""
    from brain_fwi.constitutive import alpha_basis_library
    omega = _band()
    Phi, names = alpha_basis_library(omega)
    assert Phi.shape[0] == omega.shape[0]
    assert Phi.shape[1] == len(names) >= 4
    Phi0, _ = alpha_basis_library(jnp.array([0.0]))
    assert float(jnp.max(jnp.abs(Phi0))) < 1e-6, "library not DC-vanishing at omega=0"


def test_discovery_recovers_single_power_law_sparsely():
    """A pure power-law alpha(omega) ∝ |f|^1.3 is discovered as ~one term."""
    from brain_fwi.constitutive import alpha_basis_library, discover_alpha_law
    omega = _band()
    f = omega / (2 * np.pi)
    alpha_true = 0.7 * (f / 1e6) ** 1.3
    Phi, names = alpha_basis_library(omega, exponents=(1.0, 1.3, 1.5, 2.0))
    res = discover_alpha_law(alpha_true, Phi, names, l0_penalty=1e-3)
    assert res.rel_rmse < 0.05, f"fit too poor: {res.rel_rmse:.3f}"
    assert res.n_terms <= 2, f"not sparse: {res.n_terms} terms {res.active_terms}"
    # the y=1.3 power-law term must be among the selected
    assert any("1.3" in t for t in res.active_terms), res.active_terms


def test_discovery_weights_are_nonnegative():
    """Dissipation ≥ 0: discovered weights are non-negative (softplus/NNLS)."""
    from brain_fwi.constitutive import alpha_basis_library, discover_alpha_law
    omega = _band()
    f = omega / (2 * np.pi)
    alpha_true = 0.5 * (f / 1e6) ** 1.1 + 0.3 * (f / 1e6) ** 2.0
    Phi, names = alpha_basis_library(omega)
    res = discover_alpha_law(alpha_true, Phi, names, l0_penalty=1e-4)
    assert np.all(np.asarray(res.weights) >= -1e-9), res.weights


def test_gm_wm_same_shape_different_magnitude():
    """Kuhl's brain result in our domain: GM and WM select the SAME building
    block(s); only the magnitude differs (~1.5x, Kang 2022 / our table)."""
    from brain_fwi.constitutive import alpha_basis_library, discover_alpha_law
    omega = _band()
    f = omega / (2 * np.pi)
    gm = 0.6 * (f / 1e6) ** 1.3
    wm = 0.9 * (f / 1e6) ** 1.3
    Phi, names = alpha_basis_library(omega, exponents=(1.0, 1.3, 1.5, 2.0))
    rg = discover_alpha_law(gm, Phi, names, l0_penalty=1e-3)
    rw = discover_alpha_law(wm, Phi, names, l0_penalty=1e-3)
    # Same functional form: identical selected support.
    assert set(rg.active_terms) == set(rw.active_terms), (
        f"GM {rg.active_terms} != WM {rw.active_terms}")
    # Different magnitude: total attenuation ratio ~1.5.
    ratio = float(np.sum(rw.weights)) / float(np.sum(rg.weights) + 1e-12)
    assert 1.3 < ratio < 1.7, f"WM/GM magnitude ratio {ratio:.2f} not ~1.5"

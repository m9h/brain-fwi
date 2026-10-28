"""Anisotropy is the CANN-native GM/WM SHAPE discriminator.

Kuhl's isotropic brain CANN finds GM and WM share a functional law and differ
only in magnitude. The one axis where they differ in FORM is anisotropy: white
matter is fibre-oriented, so its attenuation is direction-dependent
(alpha(omega, theta)), while gray matter is isotropic. This is the structural-
tensor (I4/I5) extension Kuhl did not activate — and it is what multi-angle
transmission FWI could resolve.

Here we demonstrate it at the constitutive-discovery level: given multi-angle
alpha samples, the CANN discovery selects a direction-dependent term for WM but
NOT for GM — a genuine shape difference, not a magnitude one.

One behaviour per test, each written failing first.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest


def _grid(nf=24, na=8):
    omega = jnp.linspace(2 * np.pi * 50e3, 2 * np.pi * 300e3, nf)
    theta = jnp.linspace(0.0, np.pi / 2, na)   # 0 = along fibre, pi/2 = across
    return omega, theta


def test_anisotropic_library_is_nonneg_and_names_angular_terms():
    """The direction-dependent library keeps non-negative, DC-vanishing columns
    and labels isotropic vs anisotropic (sin^2 theta) angular factors."""
    from brain_fwi.constitutive import alpha_basis_library_anisotropic
    omega, theta = _grid()
    Phi, names = alpha_basis_library_anisotropic(omega, theta)
    assert Phi.shape[0] == omega.shape[0] * theta.shape[0]
    assert Phi.shape[1] == len(names)
    assert np.all(np.asarray(Phi) >= -1e-9), "angular basis must stay non-negative"
    assert any("iso" in n for n in names) and any("sin2" in n for n in names)


def test_gm_isotropic_selects_no_anisotropy_term():
    """Gray matter alpha is direction-independent ⇒ discovery selects only
    isotropic terms."""
    from brain_fwi.constitutive import (
        alpha_basis_library_anisotropic, discover_alpha_law)
    omega, theta = _grid()
    f = np.asarray(omega) / (2 * np.pi)
    W, T = np.meshgrid(f, np.asarray(theta), indexing="ij")
    gm = (0.6 * (W / 1e6) ** 1.3).ravel()                 # no theta dependence
    Phi, names = alpha_basis_library_anisotropic(omega, theta, exponents=(1.0, 1.3, 2.0))
    res = discover_alpha_law(gm, Phi, names, l0_penalty=1e-3)
    assert not any("sin2" in t for t in res.active_terms), (
        f"GM wrongly picked an anisotropy term: {res.active_terms}")


def test_wm_anisotropic_selects_a_direction_dependent_term():
    """White matter attenuation depends on angle to fibre ⇒ discovery selects a
    sin^2(theta) (anisotropic) term that GM does not — the SHAPE discriminator."""
    from brain_fwi.constitutive import (
        alpha_basis_library_anisotropic, discover_alpha_law)
    omega, theta = _grid()
    f = np.asarray(omega) / (2 * np.pi)
    W, T = np.meshgrid(f, np.asarray(theta), indexing="ij")
    # anisotropic: baseline + extra attenuation across fibres (sin^2 theta)
    wm = (0.9 * (W / 1e6) ** 1.3 + 0.6 * (W / 1e6) ** 1.3 * np.sin(T) ** 2).ravel()
    Phi, names = alpha_basis_library_anisotropic(omega, theta, exponents=(1.0, 1.3, 2.0))
    res = discover_alpha_law(wm, Phi, names, l0_penalty=1e-3)
    assert any("sin2" in t for t in res.active_terms), (
        f"WM failed to discover its anisotropy term: {res.active_terms}")
    assert res.rel_rmse < 0.1, f"WM fit poor: {res.rel_rmse:.3f}"

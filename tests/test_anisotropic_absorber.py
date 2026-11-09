"""Anisotropic-attenuation absorber (prototype for the j-Wave fork).

Directional acoustic loss as a per-voxel symmetric TENSOR D(x): the loss operator
``sum_ij D_ij d_i d_j (field)`` is LOCAL (spectral derivatives contracted with a
local tensor), so it handles spatially-varying anisotropy, is differentiable
(the FWI unknowns are the tensor fields), and parallels the DTI diffusion tensor.

For a plane wave at angle theta the loss coefficient is
``k^T D k / |k|^2 = a_par*cos^2(theta-phi) + a_perp*sin^2(theta-phi)`` -- exactly
the sin^2(theta-phi) anisotropy the ray-tomography sandbox assumed. So this
absorber IS the full-wave implementation of the sandbox model.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest


def test_tensor_eigenstructure():
    """The attenuation tensor has the fibre direction as an eigenvector with
    eigenvalue a_par (along), and the perpendicular with a_perp (across)."""
    from brain_fwi.simulation.anisotropic_absorber import fibre_attenuation_tensor
    phi, a_par, a_perp = 0.6, 0.2, 0.9
    Dxx, Dxy, Dyy = (float(x) for x in fibre_attenuation_tensor(phi, a_par, a_perp))
    D = np.array([[Dxx, Dxy], [Dxy, Dyy]])
    v = np.array([np.cos(phi), np.sin(phi)])          # fibre direction
    w = np.array([-np.sin(phi), np.cos(phi)])         # perpendicular
    assert np.allclose(D @ v, a_par * v, atol=1e-6)
    assert np.allclose(D @ w, a_perp * w, atol=1e-6)


# integer (on-grid) wavevectors so the plane wave is exactly periodic -> the
# spectral derivative is exact (a non-integer k leaks and biases the estimate).
@pytest.mark.parametrize("mx,my", [(3, 0), (3, 2), (2, 3), (0, 3), (-2, 3)])
def test_loss_coefficient_matches_sin2_anisotropy(mx, my):
    """A plane wave at angle theta=atan2(my,mx): the measured loss coefficient
    equals a_par cos^2(theta-phi) + a_perp sin^2(theta-phi) -- the sandbox sin^2
    model, i.e. this tensor absorber IS the sandbox anisotropy, full-wave."""
    from brain_fwi.simulation.anisotropic_absorber import (
        fibre_attenuation_tensor, anisotropic_loss)
    N, dx = 48, 1.0e-3
    phi, a_par, a_perp = 0.6, 0.2, 0.9
    Dxx, Dxy, Dyy = fibre_attenuation_tensor(phi, a_par, a_perp)
    Dxx, Dxy, Dyy = (jnp.full((N, N), float(v)) for v in (Dxx, Dxy, Dyy))
    kx = 2 * np.pi * mx / (N * dx); ky = 2 * np.pi * my / (N * dx)
    theta = np.arctan2(my, mx)
    xx, yy = np.meshgrid(np.arange(N) * dx, np.arange(N) * dx, indexing="ij")
    p = jnp.asarray(np.cos(kx * xx + ky * yy).astype(np.float32))
    loss = np.asarray(anisotropic_loss(p, Dxx, Dxy, Dyy, dx))
    kmag2 = kx ** 2 + ky ** 2
    coeff = -np.sum(loss * np.asarray(p)) / np.sum(kmag2 * np.asarray(p) ** 2)
    expect = a_par * np.cos(theta - phi) ** 2 + a_perp * np.sin(theta - phi) ** 2
    assert coeff == pytest.approx(expect, rel=0.03), f"theta={theta:.2f}: {coeff:.3f} vs {expect:.3f}"


def test_loss_is_differentiable_wrt_tensor():
    """The loss is differentiable w.r.t. the tensor fields (the FWI unknowns)."""
    import jax
    from brain_fwi.simulation.anisotropic_absorber import anisotropic_loss
    N, dx = 24, 1e-3
    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.normal(size=(N, N)).astype(np.float32))
    def scalar(Dxx):
        return jnp.sum(anisotropic_loss(p, Dxx, jnp.zeros((N, N)), jnp.zeros((N, N)), dx) ** 2)
    g = jax.grad(scalar)(jnp.full((N, N), 0.3))
    assert np.all(np.isfinite(np.asarray(g))) and np.any(np.asarray(g) != 0)

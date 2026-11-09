"""Anisotropic-attenuation absorber (prototype for the j-Wave fork).

Directional acoustic loss as a per-voxel symmetric attenuation TENSOR ``D(x)``,
applied via ``sum_ij D_ij d_i d_j (field)``. This is:

  - **local** — spectral second derivatives contracted with a *local* tensor
    field, so spatially-varying anisotropy is handled directly (unlike the
    isotropic fractional-Laplacian absorber, whose global exponent cannot vary
    per voxel);
  - **differentiable** — the tensor components are the FWI unknowns;
  - the **full-wave implementation of the ray-tomography sandbox model**: for a
    plane wave at angle theta the loss coefficient is
    ``k^T D k / |k|^2 = a_par cos^2(theta-phi) + a_perp sin^2(theta-phi)`` — the
    same ``sin^2(theta-phi)`` anisotropy validated there;
  - the **acoustic analogue of the DTI diffusion tensor** (same rank-2 symmetric
    structure; principal eigenvector = fibre direction), which is why diffusion
    FA/orientation is the physical ground truth.

This module is the 2-D prototype; the fork build reuses the same tensor form in
the Treeby-Cox pressure update (replacing the isotropic |k|^p loss).
"""

from __future__ import annotations

import jax.numpy as jnp


def fibre_attenuation_tensor(phi, a_par, a_perp):
    """Symmetric 2x2 attenuation tensor with the fibre direction (angle ``phi``)
    as an eigenvector of eigenvalue ``a_par`` (attenuation along the fibre) and
    the perpendicular of eigenvalue ``a_perp`` (across). Returns ``(Dxx, Dxy,
    Dyy)`` (scalars or fields).

    Physical WM: attenuation is larger across fibres, so ``a_perp > a_par``.
    """
    c = jnp.cos(phi); s = jnp.sin(phi)
    Dxx = a_par * c * c + a_perp * s * s
    Dyy = a_par * s * s + a_perp * c * c
    Dxy = (a_par - a_perp) * c * s
    return Dxx, Dxy, Dyy


def anisotropic_loss(p, Dxx, Dxy, Dyy, dx):
    """Directional loss operator ``sum_ij D_ij d_i d_j p`` (2-D), with second
    derivatives taken spectrally (pseudospectral, as in j-Wave) and contracted
    with the local tensor field. For a plane wave ``cos(k.x)`` this returns
    ``-(k^T D k) p``, so the loss is minimal along the low-eigenvalue axis and
    maximal along the high-eigenvalue axis — directional attenuation.
    """
    nx, ny = p.shape
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=dx)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    P = jnp.fft.fft2(p)
    pxx = jnp.real(jnp.fft.ifft2(-(KX ** 2) * P))
    pyy = jnp.real(jnp.fft.ifft2(-(KY ** 2) * P))
    pxy = jnp.real(jnp.fft.ifft2(-(KX * KY) * P))
    return Dxx * pxx + 2.0 * Dxy * pxy + Dyy * pyy

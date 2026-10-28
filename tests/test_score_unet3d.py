"""Spec for the 3D conv U-Net score net + 3D Birnbaum volume dataset builder.

Companion to test_diffusion_prior.py (which covers the SDE/DSM/sampler core).
Here we pin the 3D score net's I/O contract — flat (S^3,) in/out, finite, and
composability with the diffusion sampler — and the 3D cerebrum dataset builder.
"""
from __future__ import annotations

import os

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest


def test_unet3d_score_returns_flat_cube_shape():
    """UNet3DScore(theta_t: (S^3,), t) -> (S^3,), finite."""
    from brain_fwi.inference.score_unet3d import UNet3DScore

    S = 8
    net = UNet3DScore(S, C=4, key=jr.PRNGKey(0))
    theta_t = jnp.ones((S ** 3,))
    out = net(theta_t, jnp.array(0.3))
    assert out.shape == (S ** 3,), f"got {out.shape}, expected {(S ** 3,)}"
    assert jnp.all(jnp.isfinite(out))


def test_unet3d_score_composes_with_ddim_sample():
    """The net works as a ``score_fn`` for the diffusion sampler (correct shape)."""
    from brain_fwi.inference.diffusion import VPSDE, ddim_sample
    from brain_fwi.inference.score_unet3d import UNet3DScore

    S = 8
    net = UNet3DScore(S, C=4, key=jr.PRNGKey(0))
    sde = VPSDE()
    samp = ddim_sample(lambda th, t: net(th, t), sde, dim=S ** 3,
                       n_samples=2, n_steps=4, key=jr.PRNGKey(1))
    assert samp.shape == (2, S ** 3)
    assert jnp.all(jnp.isfinite(samp))


def test_unet3d_score_is_time_dependent():
    """Output varies with t (the time embedding is wired in, not dead weight)."""
    from brain_fwi.inference.score_unet3d import UNet3DScore

    S = 8
    net = UNet3DScore(S, C=4, key=jr.PRNGKey(0))
    x = jr.normal(jr.PRNGKey(2), (S ** 3,))
    o_lo = net(x, jnp.array(0.05))
    o_hi = net(x, jnp.array(0.9))
    assert not jnp.allclose(o_lo, o_hi)


@pytest.mark.skipif(
    not os.path.isdir("/data/datasets/birnbaum/Data/Anonymized_Subjects"),
    reason="Birnbaum dataset not on disk",
)
def test_build_volume_dataset_shapes_and_flip_augment():
    """build_volume_dataset returns (N, S^3) float32 with L-R flip doubling N."""
    from brain_fwi.phantoms import birnbaum

    files = birnbaum.label_files()[:2]
    S = 32
    data = birnbaum.build_volume_dataset(files, S=S, flip_augment=True)
    assert data.ndim == 2 and data.shape[1] == S ** 3
    assert data.shape[0] % 2 == 0  # flip augment pairs
    assert data.dtype == np.float32
    # velocities live in a sane acoustic band (water .. lesion)
    assert data.min() >= birnbaum.C_WATER - 1
    assert data.max() <= birnbaum.C_LESION + 1

"""2D Fourier Neural Operator for Phase 4 Evidence 9.2.1.

Wraps ``pdequinox.arch.ClassicFNO`` for the toy 2D experiment:
``(H, W, 1) sound speed → (N_t,) sensor trace``.

Used by ``scripts/toy_2d_fno.py``. Backed by PDEQuinox for robust
spectral handling and consistent initialisation across 2D/3D.
"""

from __future__ import annotations

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from .uno import UNONet


class CToTraceFNO(eqx.Module):
    """``(H, W, 1) sound-speed`` → ``(N_t,) sensor trace``.

    UNO backbone captures multi-scale features; a global-average-pool + MLP
    head collapses to the trace.
    """

    backbone: UNONet
    head: eqx.nn.MLP
    grid_h: int = eqx.field(static=True)
    grid_w: int = eqx.field(static=True)
    n_timesteps: int = eqx.field(static=True)

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        n_timesteps: int,
        width: int,
        modes: int,
        depth: int,
        key: jax.Array,
    ):
        fno_key, head_key = jr.split(key)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_timesteps = n_timesteps

        self.backbone = UNONet(
            num_spatial_dims=2,
            in_channels=1,
            out_channels=width,
            hidden_channels=width,
            num_modes=modes,
            depth=depth,
            key=fno_key,
        )
        self.head = eqx.nn.MLP(
            in_size=width,
            out_size=n_timesteps,
            width_size=max(width, 32),
            depth=2,
            key=head_key,
        )

    def __call__(self, c_field: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            c_field: ``(H, W, 1)`` or ``(H, W)`` sound speed.

        Returns:
            ``(n_timesteps,)`` trace.
        """
        if c_field.ndim == 2:
            x = c_field[None, ...]  # (1, H, W)
        else:
            x = jnp.moveaxis(c_field, -1, 0)  # (1, H, W)

        features = self.backbone(x)                # (width, H, W)
        pooled = jnp.mean(features, axis=(1, 2))  # (width,)
        return self.head(pooled)

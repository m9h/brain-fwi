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


class CToTraceFNO2DGather(eqx.Module):
    """2D FNO with a spatial-gather readout at a fixed receiver position.

    Replaces the global-average-pool head of :class:`CToTraceFNO` with a
    direct lookup of the UNO's latent feature vector at the receiver
    voxel. Per the design doc (`docs/design/phase4_fno_surrogate.md`
    §10 step 4) and the Phase-4 readiness evidence (toy 2D FNO with
    pool readout hit p50 ≈ 8% rel-L2 vs the <1% gate), spatial-gather
    is the architectural upgrade most likely to close the trace-fidelity
    gap because the wave equation's response at a sensor depends on the
    sensor's *location*, not on a global summary of the velocity field.

    The readout is still a small MLP — the gather just selects which
    feature vector the MLP sees. For a fixed-helmet surrogate (Phase 4
    V1 non-goal: variable geometry), the receiver position is part of
    the model state.
    """

    backbone: UNONet
    head: eqx.nn.MLP
    grid_h: int = eqx.field(static=True)
    grid_w: int = eqx.field(static=True)
    n_timesteps: int = eqx.field(static=True)
    recv_pos: Tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        n_timesteps: int,
        recv_pos: Tuple[int, int],
        width: int,
        modes: int,
        depth: int,
        key: jax.Array,
    ):
        fno_key, head_key = jr.split(key)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_timesteps = n_timesteps
        self.recv_pos = (int(recv_pos[0]), int(recv_pos[1]))

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
        """``(H, W) or (H, W, 1)`` → ``(n_timesteps,)``."""
        if c_field.ndim == 2:
            x = c_field[None, ...]              # (1, H, W)
        else:
            x = jnp.moveaxis(c_field, -1, 0)    # (1, H, W)
        features = self.backbone(x)              # (width, H, W)
        ry, rx = self.recv_pos
        gathered = features[:, ry, rx]           # (width,)
        return self.head(gathered)


class CToTraceFNO2DPoolGather(eqx.Module):
    """2D FNO with ``concat(global-pool, gather-at-receiver)`` readout.

    First A/B (`scripts/bench_fno_gather_vs_pool.py`) showed gather alone
    underperforms pool on a Gaussian-bump toy — the single-voxel feature
    throws away noise-averaging without adding much information (the
    FNO's spectral conv already mixes globally). This variant keeps both:
    pool for robust global summary, gather for receiver-localised
    feature when it helps. The head input doubles to ``2 * width``.
    """

    backbone: UNONet
    head: eqx.nn.MLP
    grid_h: int = eqx.field(static=True)
    grid_w: int = eqx.field(static=True)
    n_timesteps: int = eqx.field(static=True)
    recv_pos: Tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        n_timesteps: int,
        recv_pos: Tuple[int, int],
        width: int,
        modes: int,
        depth: int,
        key: jax.Array,
    ):
        fno_key, head_key = jr.split(key)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_timesteps = n_timesteps
        self.recv_pos = (int(recv_pos[0]), int(recv_pos[1]))
        self.backbone = UNONet(
            num_spatial_dims=2, in_channels=1, out_channels=width,
            hidden_channels=width, num_modes=modes, depth=depth, key=fno_key,
        )
        self.head = eqx.nn.MLP(
            in_size=2 * width, out_size=n_timesteps,
            width_size=max(2 * width, 32), depth=2, key=head_key,
        )

    def __call__(self, c_field: jax.Array) -> jax.Array:
        if c_field.ndim == 2:
            x = c_field[None, ...]
        else:
            x = jnp.moveaxis(c_field, -1, 0)
        features = self.backbone(x)
        pooled = jnp.mean(features, axis=(1, 2))
        ry, rx = self.recv_pos
        gathered = features[:, ry, rx]
        return self.head(jnp.concatenate([pooled, gathered]))

"""3D FNO surrogate scaffold — ``(D, H, W) velocity + source-position``
→ ``(N_t, N_recv) helmet traces``.

Implements §10 step 2 of ``docs/design/phase4_fno_surrogate.md`` at MVP
scope: a thin wrapper around ``pdequinox.arch.ClassicFNO`` plus a
global-pool → MLP readout that produces a per-shot trace tensor.

This is the *architecture* step. The real surrogate training loop
lives in :mod:`brain_fwi.surrogate.train` (to be written — §10 step 3)
and consumes Phase-0 shards via :class:`~brain_fwi.data.ShardedReader`.

Design choices:

- **Input encoding**: two channels. Channel 0 is the normalised sound
  speed ``(c - c_min) / (c_max - c_min)``. Channel 1 is a Gaussian spike
  at the source voxel, standing in for a delta function. A spike channel
  (rather than concatenating ``src_pos`` as a scalar) lets the FNO
  condition on source location spatially, which matches how the wave
  equation depends on the forcing location.
- **Readout head**: global-average-pool over spatial axes, then a small
  MLP mapping the feature vector to ``N_t × N_recv`` entries. This is
  the MVP — a spatial-gather head that reads the FNO's latent field at
  the sensor coordinates is the natural V2 upgrade once we measure
  trace fidelity (§7.2 gate).
- **Backed by pdequinox**: the bespoke ``FNO2D`` in ``fno2d.py`` was
  written for the 2D toy; at 3D the weight init, FFT handling and
  boundary modes become tricky enough that building on a maintained
  library makes sense. ``pdequinox.arch.ClassicFNO`` takes
  ``(channels, D, H, W)`` input and preserves spatial shape with a
  configurable channel count.

Usage sketch::

    key = jr.PRNGKey(0)
    model = CToTraceFNO3D(
        grid_shape=(96, 96, 96),
        n_timesteps=74,
        n_receivers=128,
        hidden_channels=32, num_modes=12, num_blocks=4,
        key=key,
    )
    c_norm = (velocity - c_min) / (c_max - c_min)  # (D, H, W)
    traces = model(c_norm, src_pos_grid=(48, 48, 48))  # (74, 128)
"""

from __future__ import annotations

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from pdequinox.arch import ClassicFNO


def _source_spike(
    grid_shape: Tuple[int, int, int],
    src_pos_grid: Tuple[int, int, int],
    sigma_voxels: float = 1.0,
) -> jnp.ndarray:
    """Gaussian blob at ``src_pos_grid`` on a ``grid_shape`` volume.

    A true delta has no differentiable location gradient and creates a
    high-frequency feature the FNO would need many modes to resolve; a
    width-1 Gaussian keeps the spike smooth while still localising the
    source to a few voxels.
    """
    axes = [jnp.arange(n, dtype=jnp.float32) for n in grid_shape]
    dx, dy, dz = src_pos_grid
    xg, yg, zg = jnp.meshgrid(*axes, indexing="ij")
    r2 = (xg - dx) ** 2 + (yg - dy) ** 2 + (zg - dz) ** 2
    return jnp.exp(-0.5 * r2 / (sigma_voxels ** 2))


class CToTraceFNO3D(eqx.Module):
    """``(D, H, W) sound speed`` → ``(N_t, N_recv) helmet traces``.

    MVP scaffold per Phase 4 design §10 step 2. Fixed helmet geometry
    is baked into the output shape; swap the readout for a multi-head
    design if a geometry-conditional surrogate lands later.
    """

    fno: ClassicFNO
    head: eqx.nn.MLP
    grid_shape: Tuple[int, int, int] = eqx.field(static=True)
    n_timesteps: int = eqx.field(static=True)
    n_receivers: int = eqx.field(static=True)
    hidden_channels: int = eqx.field(static=True)

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        n_timesteps: int,
        n_receivers: int,
        *,
        hidden_channels: int = 32,
        num_modes: int = 12,
        num_blocks: int = 4,
        key: jax.Array,
    ):
        fno_key, head_key = jr.split(key)
        self.grid_shape = tuple(int(x) for x in grid_shape)
        self.n_timesteps = int(n_timesteps)
        self.n_receivers = int(n_receivers)
        self.hidden_channels = int(hidden_channels)

        self.fno = ClassicFNO(
            num_spatial_dims=3,
            in_channels=2,            # velocity + source-spike
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_modes=num_modes,
            num_blocks=num_blocks,
            key=fno_key,
        )
        self.head = eqx.nn.MLP(
            in_size=hidden_channels,
            out_size=n_timesteps * n_receivers,
            width_size=max(hidden_channels * 2, 64),
            depth=2,
            key=head_key,
        )

    def __call__(
        self,
        c_norm: jnp.ndarray,
        src_pos_grid: Tuple[int, int, int],
    ) -> jnp.ndarray:
        """Forward pass for a single shot.

        Args:
            c_norm: ``(D, H, W)`` normalised sound speed in roughly ``[0, 1]``.
            src_pos_grid: ``(ix, iy, iz)`` voxel coords of the source.

        Returns:
            ``(n_timesteps, n_receivers)`` trace tensor.
        """
        spike = _source_spike(self.grid_shape, src_pos_grid)
        x = jnp.stack([c_norm, spike], axis=0)    # (2, D, H, W)
        features = self.fno(x)                    # (hidden_channels, D, H, W)
        pooled = jnp.mean(features, axis=(1, 2, 3))  # (hidden_channels,)
        flat = self.head(pooled)                  # (n_t * n_recv,)
        return flat.reshape(self.n_timesteps, self.n_receivers)

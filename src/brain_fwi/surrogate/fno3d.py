"""3D FNO surrogate — ``(D, H, W) velocity + source-position``
→ ``(N_t, N_recv) helmet traces``.

Implements §10 step 2 of ``docs/design/phase4_fno_surrogate.md``.

Design choices:

- **Input encoding**: two channels. Channel 0 is the normalised sound
  speed ``(c - c_min) / (c_max - c_min)``. Channel 1 is a Gaussian spike
  at the source voxel, standing in for a delta function. A spike channel
  (rather than concatenating ``src_pos`` as a scalar) lets the FNO
  condition on source location spatially, which matches how the wave
  equation depends on the forcing location.
- **Readout head — point-sample at receiver coordinates**. The earlier
  global-average-pool readout discarded all spatial structure: the
  entire ``(hidden_channels, D, H, W)`` feature volume collapsed to
  ``hidden_channels`` numbers, after which the head MLP had to fan
  those out to ``N_t × N_recv`` outputs via a fixed per-receiver
  weighting. Empirically this collapsed to predict-zero everywhere
  on the v2a dataset (loss locked at ~1.30 across 4 ablations of
  ``lambda_spec`` / ``output_scale`` / ``c_min`` / ``c_max`` /
  learning-rate). The point-sample head reads ``features[:, rx, ry, rz]``
  at each receiver voxel and runs an MLP per receiver to produce its
  trace — preserving the spatial information the FNO body computed.
- **Backed by UNONet** for robust spectral handling.

Usage::

    receivers = ((48, 0, 48), (48, 95, 48), ...)  # voxel coords
    model = CToTraceFNO3D(
        grid_shape=(96, 96, 96),
        n_timesteps=1100,
        n_receivers=128,
        hidden_channels=32, num_modes=12, depth=2,
        receiver_positions=receivers,
        key=jr.PRNGKey(0),
    )
    c_norm = (velocity - c_min) / (c_max - c_min)  # (D, H, W)
    traces = model(c_norm, src_pos_grid=(48, 48, 48))  # (1100, 128)
"""

from __future__ import annotations

from typing import Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from .uno import UNONet


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

    Spatial-gather readout: the trained backbone produces a
    ``(hidden_channels, D, H, W)`` feature volume; for each receiver
    voxel ``(rx, ry, rz)`` the model reads ``features[:, rx, ry, rz]``
    and runs a shared per-receiver MLP to produce that receiver's
    ``n_timesteps``-long trace. Receiver coordinates are baked in as
    static (non-trainable) ints so they don't drift during training.
    """

    backbone: UNONet
    head: eqx.nn.MLP
    grid_shape: Tuple[int, int, int] = eqx.field(static=True)
    n_timesteps: int = eqx.field(static=True)
    n_receivers: int = eqx.field(static=True)
    hidden_channels: int = eqx.field(static=True)
    # Pin output_scale as static (non-trainable). When trainable it
    # collapses to zero — minimising loss by simply zeroing the
    # prediction — and the FNO body then has no gradient to learn
    # anything. Confirmed empirically.
    output_scale: float = eqx.field(static=True)
    # Receiver voxel coords as a flat tuple-of-ints triple. Tuple form
    # keeps it static (no float leaves) and JIT-compatible.
    receiver_positions: Tuple[Tuple[int, int, int], ...] = eqx.field(static=True)

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        n_timesteps: int,
        n_receivers: int,
        *,
        receiver_positions: Sequence[Tuple[int, int, int]],
        hidden_channels: int = 32,
        num_modes: int = 12,
        depth: int = 2,
        output_scale: float = 1.0,
        key: jax.Array,
    ):
        if len(receiver_positions) != n_receivers:
            raise ValueError(
                f"receiver_positions has {len(receiver_positions)} entries "
                f"but n_receivers={n_receivers}"
            )
        nx, ny, nz = grid_shape
        for i, (rx, ry, rz) in enumerate(receiver_positions):
            if not (0 <= rx < nx and 0 <= ry < ny and 0 <= rz < nz):
                raise ValueError(
                    f"receiver {i} at {(rx, ry, rz)} outside grid {grid_shape}"
                )

        fno_key, head_key = jr.split(key)
        self.grid_shape = tuple(int(x) for x in grid_shape)
        self.n_timesteps = int(n_timesteps)
        self.n_receivers = int(n_receivers)
        self.hidden_channels = int(hidden_channels)
        self.output_scale = float(output_scale)
        self.receiver_positions = tuple(
            (int(p[0]), int(p[1]), int(p[2])) for p in receiver_positions
        )

        self.backbone = UNONet(
            num_spatial_dims=3,
            in_channels=2,            # velocity + source-spike
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_modes=num_modes,
            depth=depth,
            key=fno_key,
        )
        # Per-receiver MLP: features at receiver voxel (hidden,)
        # → trace samples (n_timesteps,). Shared weights across
        # receivers — so receivers are distinguished only by where
        # they sit in the feature volume, which is what we want.
        self.head = eqx.nn.MLP(
            in_size=hidden_channels,
            out_size=n_timesteps,
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
        x = jnp.stack([c_norm, spike], axis=0)             # (2, D, H, W)
        features = self.backbone(x)                        # (hidden, D, H, W)

        # Point-sample features at each receiver voxel.
        # rx, ry, rz are static int tuples — JIT-friendly indexing.
        rx = jnp.array([p[0] for p in self.receiver_positions], dtype=jnp.int32)
        ry = jnp.array([p[1] for p in self.receiver_positions], dtype=jnp.int32)
        rz = jnp.array([p[2] for p in self.receiver_positions], dtype=jnp.int32)
        # features shape (hidden, D, H, W). Index along (D, H, W) with
        # advanced indexing → result shape (hidden, n_recv). Transpose
        # to (n_recv, hidden) for the per-receiver MLP.
        per_recv_feats = features[:, rx, ry, rz].T          # (n_recv, hidden)

        # Apply the shared head per receiver. vmap over the receiver axis.
        per_recv_traces = jax.vmap(self.head)(per_recv_feats)  # (n_recv, n_t)

        scaled = self.output_scale * per_recv_traces.T      # (n_t, n_recv)
        return scaled

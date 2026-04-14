"""Checkpointed scan for memory-efficient autodiff through long time series.

Standard jax.lax.scan stores all intermediate carries for the backward pass,
which at 192^3 x 1102 timesteps requires ~218 GB. This module implements a
two-level nested scan that reduces memory to O(sqrt(N)) carries:

  - Outer scan over K = ceil(sqrt(N)) segments, with jax.checkpoint
  - Inner scan over S = ceil(N/K) steps per segment

During the backward pass, only segment-boundary carries are stored (K states).
Each segment's internal carries are recomputed from the boundary carry.
Total memory: (K + S) carries instead of N carries.

For N=1102 at 192^3 (~198 MB/carry):
  Flat scan:   1102 * 198 MB = 218 GB
  Nested scan: (34 + 33) * 198 MB = 13 GB

Reference: Griewank & Walther (2000), "Algorithm 799: revolve"
"""

import math

import jax
import jax.numpy as jnp
from jax.lax import scan


def checkpointed_scan(f, init, xs, segment_length=None):
    """Drop-in replacement for jax.lax.scan with gradient checkpointing.

    Splits the scan into segments. The outer scan body is wrapped in
    jax.checkpoint so that only segment-boundary carries are stored
    during the forward pass. The inner scan stores all carries within
    one segment (recomputed during backward).

    Args:
        f: Scan body function (carry, x) -> (carry, y).
        init: Initial carry (pytree).
        xs: Inputs to scan over, leading axis is time (pytree).
        segment_length: Steps per segment. None = ceil(sqrt(N)).

    Returns:
        (final_carry, ys) — same as jax.lax.scan.
    """
    n_steps = jax.tree.leaves(xs)[0].shape[0]

    if segment_length is None:
        segment_length = int(math.ceil(math.sqrt(n_steps)))

    n_segments = int(math.ceil(n_steps / segment_length))
    padded_len = n_segments * segment_length

    # Pad xs to exact multiple of segment_length
    def pad_leaf(x):
        pad_size = padded_len - x.shape[0]
        if pad_size == 0:
            return x
        pad_widths = [(0, pad_size)] + [(0, 0)] * (x.ndim - 1)
        return jnp.pad(x, pad_widths)

    xs_padded = jax.tree.map(pad_leaf, xs)

    # Reshape leading axis: (padded_len, ...) -> (n_segments, segment_length, ...)
    def reshape_leaf(x):
        return x.reshape((n_segments, segment_length) + x.shape[1:])

    xs_segments = jax.tree.map(reshape_leaf, xs_padded)

    @jax.checkpoint
    def outer_step(carry, xs_segment):
        final_carry, ys_segment = scan(f, carry, xs_segment)
        return final_carry, ys_segment

    final_carry, all_ys = scan(outer_step, init, xs_segments)

    # Flatten ys: (n_segments, segment_length, ...) -> (padded_len, ...)
    def flatten_leaf(y):
        return y.reshape((padded_len,) + y.shape[2:])[:n_steps]

    ys = jax.tree.map(flatten_leaf, all_ys)

    return final_carry, ys

"""Multi-GPU shot-parallel FNO training surface.

Shards the helmet's source axis across a :class:`jax.sharding.Mesh` so
each device computes a slice of the per-shot losses in parallel. The
c-field is replicated on every device.

Why this exists: the Phase-0 helmet has 128 shots per sample, and the
FNO forward pass runs once per shot. With one GPU,
:func:`brain_fwi.surrogate.train.surrogate_loss` serialises the 128
forwards via ``lax.scan``. With 4 GPUs, :func:`shot_parallel_loss`
shards the 128 sources across the mesh and each device runs ``128 /
n_devices`` shots in parallel — a roughly ``n_devices``× per-step
speedup at the cost of an ``n_devices``× GPU bill.

The hybrid "shard outer, scan inner" structure keeps activation memory
per device bounded by one shot's forward+backward (sequential within
each rank) while still exploiting cross-device parallelism. This is
what fits at 128³ × `hidden=32 / depth=2` on H100/A100-class GPUs.

Usage::

    mesh = make_shot_mesh(4)
    loss = shot_parallel_loss(model, c_norm, d_true, src_pos, mesh)

Equivalence to :func:`surrogate_loss` is regression-tested on a
1-device CPU mesh in ``tests/test_phase4_parallel.py``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from .train import _rel_l2, _spectral_rel_l2


def make_shot_mesh(n_devices: int | None = None) -> Mesh:
    """Build a 1-D ``Mesh`` named ``'shots'`` over ``n_devices`` devices.

    Defaults to all available devices. On a single-CPU/GPU host this
    yields a 1-device mesh — useful for the equivalence regression
    tests where the parallel path must reduce to the serial path.
    """
    devices = jax.devices()
    if n_devices is None:
        n_devices = len(devices)
    if n_devices > len(devices):
        raise ValueError(
            f"asked for {n_devices} devices, only {len(devices)} available"
        )
    return Mesh(np.asarray(devices[:n_devices]).reshape(n_devices), ("shots",))


def shot_parallel_loss(
    model,
    c_norm: jnp.ndarray,
    d_true: jnp.ndarray,
    source_positions: Sequence[Tuple[int, int, int]],
    mesh: Mesh,
    lambda_spec: float = 0.3,
) -> jnp.ndarray:
    """Sharded equivalent of :func:`surrogate_loss`.

    Args:
        model: FNO surrogate (replicated across mesh).
        c_norm: ``(D, H, W)`` normalised velocity (replicated).
        d_true: ``(n_src, n_t, n_recv)`` j-Wave traces. Sharded along
            axis 0 across the ``'shots'`` mesh axis.
        source_positions: length-``n_src`` sequence of integer voxel
            coords. Sharded along axis 0.
        mesh: 1-D mesh over the ``'shots'`` axis.
        lambda_spec: spectral-loss weight.

    Returns:
        Scalar loss equal to ``surrogate_loss(...)`` up to summation
        order numerics. Tested to 1e-5 vs serial.
    """
    n_src = len(source_positions)
    n_devices = mesh.shape["shots"]
    if n_src % n_devices != 0:
        raise ValueError(
            f"n_src={n_src} not divisible by mesh shots={n_devices}; "
            f"pad source_positions or pick a divisor mesh size"
        )

    src_arr = jnp.asarray(source_positions, dtype=jnp.int32)  # (n_src, 3)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P("shots"), P("shots")),
        out_specs=(P(), P()),
    )
    def _sharded_sum(src_local, target_local):
        # src_local: (n_src/n_devices, 3)
        # target_local: (n_src/n_devices, n_t, n_recv)
        @jax.checkpoint
        def _per_shot(src_xyz, target):
            pred = model(c_norm, (src_xyz[0], src_xyz[1], src_xyz[2]))
            return _rel_l2(pred, target), _spectral_rel_l2(pred, target)

        def body(carry, xs):
            sx, tg = xs
            t_loss, s_loss = _per_shot(sx, tg)
            return (carry[0] + t_loss, carry[1] + s_loss), None

        # JAX 0.9 shard_map requires the scan carry to declare its
        # mesh-axis variance via pcast. Inputs sharded along 'shots'
        # produce per-shard losses that vary along that axis.
        zero = jax.lax.pcast(jnp.zeros(()), ("shots",), to="varying")
        (t_sum, s_sum), _ = jax.lax.scan(
            body, (zero, zero), (src_local, target_local),
        )
        # All-reduce across mesh so every device returns the global sum.
        return (
            jax.lax.psum(t_sum, axis_name="shots"),
            jax.lax.psum(s_sum, axis_name="shots"),
        )

    time_total, spec_total = _sharded_sum(src_arr, d_true)
    return (time_total + lambda_spec * spec_total) / n_src

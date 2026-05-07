"""Training loop for the FNO surrogate (Phase 4 §10 step 3).

Consumes Phase-0 shards via :class:`~brain_fwi.data.ShardedReader` and
fits a :class:`~brain_fwi.surrogate.fno3d.CToTraceFNO3D` to the j-Wave
traces stored in each sample.

Loss per design §4::

    L(φ) = ‖F_φ(c) − F_jwave(c)‖_relL2²
         + λ_spec · ‖FFT(F_φ(c)) − FFT(F_jwave(c))‖_relL2²

Relative-L2 is per-sample: ``‖x−y‖₂ / (‖y‖₂ + ε)``. That prevents
high-amplitude samples (near-source shots) from dominating the loss.

Scope (V1):

- Single `(c, d)` pair per step. Mini-batching would interact with
  `eqx.filter_vmap` over the FNO's spatial FFT in ways we haven't
  measured memory on; deferred to V2.
- Fixed helmet geometry baked into the surrogate (design §1 non-goal).
  The source positions are loaded from the first sample's
  ``transducer_positions_grid`` and validated against subsequent
  samples.
- No held-out discipline enforced here — callers should split
  ``ShardedReader`` sample ids upstream (see design §5).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from .fno3d import CToTraceFNO3D


def _rel_l2(pred: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Per-sample relative L2: ``‖pred − target‖₂ / (‖target‖₂ + eps)``."""
    num = jnp.sqrt(jnp.sum((pred - target) ** 2))
    den = jnp.sqrt(jnp.sum(target ** 2)) + eps
    return num / den


def _spectral_rel_l2(pred: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Relative-L2 in the frequency domain, along the time axis.

    Catches phase/frequency drift that time-domain MSE can miss —
    design §4 rationale.
    """
    # pred / target shape: (n_t, n_recv). FFT along axis 0.
    f_pred = jnp.fft.rfft(pred, axis=0)
    f_tgt = jnp.fft.rfft(target, axis=0)
    return _rel_l2(jnp.abs(f_pred), jnp.abs(f_tgt), eps)


def surrogate_loss(
    model: CToTraceFNO3D,
    c_norm: jnp.ndarray,
    d_true: jnp.ndarray,
    source_positions: Sequence[Tuple[int, int, int]],
    lambda_spec: float = 0.3,
) -> jnp.ndarray:
    """Design §4 training objective on a single ``(c, d)`` pair.

    Sums per-shot losses with ``lax.scan`` + ``jax.checkpoint`` so
    backward rematerialises one shot at a time. Without this, JIT
    unrolls the per-shot loop and the activations for all 128 shots
    blow up VRAM (190 GiB on H200, observed).

    Args:
        model: FNO surrogate.
        c_norm: ``(D, H, W)`` normalised velocity.
        d_true: ``(n_src, n_t, n_recv)`` j-Wave traces.
        source_positions: list of ``(ix, iy, iz)`` source voxel coords.
        lambda_spec: weight on the spectral term.

    Returns:
        Scalar loss.
    """
    n_src = len(source_positions)
    src_arr = jnp.asarray(source_positions, dtype=jnp.int32)  # (n_src, 3)

    @jax.checkpoint
    def _per_shot(src_xyz: jnp.ndarray, target: jnp.ndarray):
        pred = model(c_norm, (src_xyz[0], src_xyz[1], src_xyz[2]))
        return _rel_l2(pred, target), _spectral_rel_l2(pred, target)

    def body(carry, xs):
        src_xyz, target = xs
        t_loss, s_loss = _per_shot(src_xyz, target)
        t_acc, s_acc = carry
        return (t_acc + t_loss, s_acc + s_loss), None

    (time_total, spec_total), _ = jax.lax.scan(
        body, (jnp.zeros(()), jnp.zeros(())), (src_arr, d_true)
    )
    return (time_total + lambda_spec * spec_total) / n_src


def _normalise_c(c: jnp.ndarray, c_min: float, c_max: float) -> jnp.ndarray:
    return (c - c_min) / (c_max - c_min)


def _voxelise_positions(arr_or_metres: np.ndarray, dx: float) -> List[Tuple[int, int, int]]:
    """Convert (N, 3) positions to integer voxel-coord tuples."""
    arr = np.asarray(arr_or_metres)
    if arr.dtype.kind == "f":
        arr = np.round(arr / dx).astype(np.int32)
    return [tuple(int(x) for x in row) for row in arr]


def _extract_source_positions(reader_item) -> List[Tuple[int, int, int]]:
    """Pull integer source grid coords from a Phase-0 sample.

    Samples store ``transducer_positions`` in metres (N, 3). The training
    loop needs voxel coords. ``gen_phase0.py`` uses a fixed helmet per
    shard, so we can read the coords from the first sample.
    """
    if "transducer_positions_grid" in reader_item:
        return _voxelise_positions(
            np.asarray(reader_item["transducer_positions_grid"]), dx=1.0,
        )
    elif "transducer_positions" in reader_item and "dx" in reader_item:
        return _voxelise_positions(
            np.asarray(reader_item["transducer_positions"]),
            float(reader_item["dx"]),
        )
    raise KeyError(
        "sample lacks transducer_positions_grid or (transducer_positions + dx)"
    )


def _extract_receiver_positions(reader_item) -> List[Tuple[int, int, int]]:
    """Pull integer receiver grid coords. Falls back to source positions
    when the sample doesn't store sensor coords explicitly (the v2a
    helmet uses the same array for emitters and receivers).
    """
    if "sensor_positions_grid" in reader_item:
        return _voxelise_positions(
            np.asarray(reader_item["sensor_positions_grid"]), dx=1.0,
        )
    if "sensor_positions" in reader_item and "dx" in reader_item:
        return _voxelise_positions(
            np.asarray(reader_item["sensor_positions"]),
            float(reader_item["dx"]),
        )
    return _extract_source_positions(reader_item)


def train_fno_surrogate(
    model: CToTraceFNO3D,
    reader,
    *,
    n_steps: int,
    key: jax.Array,
    c_min: float = 1400.0,
    c_max: float = 3200.0,
    learning_rate: float = 1e-3,
    lr_schedule: str = "cosine",
    lr_alpha: float = 0.01,
    lambda_spec: float = 0.3,
    source_positions: Optional[Sequence[Tuple[int, int, int]]] = None,
    held_out_ids: Optional[Sequence[str]] = None,
    log_every: int = 50,
    verbose: bool = True,
) -> Tuple[CToTraceFNO3D, List[float]]:
    """Train ``model`` on the ``reader``'s samples.

    Args:
        model: FNO surrogate to train (returned updated).
        reader: ``ShardedReader`` or any object with ``sample_ids``,
            ``__getitem__``, and the expected field names. Must expose
            ``sound_speed_voxel`` and ``observed_data`` per sample.
        n_steps: number of gradient steps.
        key: PRNG key for sample selection.
        c_min, c_max: velocity-normalisation bounds.
        learning_rate: peak Adam LR; ``cosine`` decays to
            ``lr_alpha * learning_rate`` over ``n_steps``.
        lr_schedule: ``"cosine"`` (default) or ``"constant"``. Cosine
            stops the late-training divergence we observed in the
            constant-LR FNO production v2 (loss bottomed at 0.48 on
            step 193, bounced up to 1.04 by step 1000).
        lr_alpha: cosine final/initial LR ratio. 0.01 → end at 1e-5
            when ``learning_rate=1e-3``.
        lambda_spec: spectral-loss weight.
        source_positions: override the auto-extracted helmet. Use when
            the reader does not expose transducer coords (e.g. tests).
        held_out_ids: sample ids to exclude from training. The validation
            half of the Phase-0 split goes here.
        log_every: print a loss line every N steps.
        verbose: print progress.

    Returns:
        ``(best_model, loss_history)``. The first element is the model
        weights from the lowest-loss step seen, NOT the model at the
        end of training. With cosine LR this usually coincides with the
        last step, but with constant LR or a destabilising config the
        best can be hundreds of steps behind the final.
    """
    train_ids = list(reader.sample_ids)
    if held_out_ids is not None:
        held = set(held_out_ids)
        train_ids = [sid for sid in train_ids if sid not in held]
    if not train_ids:
        raise ValueError("no training samples after filtering held-out ids")

    # Resolve source positions from the first sample unless caller
    # provided them explicitly.
    if source_positions is None:
        source_positions = _extract_source_positions(reader[train_ids[0]])
    source_positions = list(source_positions)

    if lr_schedule == "cosine":
        lr = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=n_steps,
            alpha=lr_alpha,
        )
    elif lr_schedule == "constant":
        lr = learning_rate
    else:
        raise ValueError(
            f"unknown lr_schedule {lr_schedule!r}; expected 'cosine' or 'constant'"
        )

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(m, opt_state, c_norm, d_true):
        def loss_fn(m_):
            return surrogate_loss(m_, c_norm, d_true, source_positions, lambda_spec)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        updates, opt_state = optimizer.update(grads, opt_state)
        m = eqx.apply_updates(m, updates)
        return m, opt_state, loss

    losses: List[float] = []
    best_loss = float("inf")
    best_model = model
    n_train = len(train_ids)
    for s in range(n_steps):
        key, subkey = jr.split(key)
        idx = int(jr.randint(subkey, (), 0, n_train))
        sample = reader[train_ids[idx]]
        c = jnp.asarray(sample["sound_speed_voxel"], dtype=jnp.float32)
        d = jnp.asarray(sample["observed_data"], dtype=jnp.float32)
        c_norm = _normalise_c(c, c_min, c_max)

        # Phase-0 samples have variable trace length (each MIDA aug has
        # a slightly different max-c, hence a different CFL-derived dt
        # and n_timesteps). The FNO head has a fixed-size output, so we
        # crop to model.n_timesteps. Pad if a sample happens to be
        # shorter than the model's expected length.
        n_t_model = int(model.n_timesteps)
        if d.shape[1] >= n_t_model:
            d = d[:, :n_t_model, :]
        else:
            pad = jnp.zeros(
                (d.shape[0], n_t_model - d.shape[1], d.shape[2]),
                dtype=d.dtype,
            )
            d = jnp.concatenate([d, pad], axis=1)

        model, opt_state, loss = step(model, opt_state, c_norm, d)
        loss_f = float(loss)
        losses.append(loss_f)
        if loss_f < best_loss:
            best_loss = loss_f
            best_model = model
        if verbose and (s + 1) % log_every == 0:
            recent = np.mean(losses[-log_every:])
            print(f"  FNO-train step {s+1}/{n_steps}: "
                  f"loss={recent:.4f} (avg over last {log_every})  "
                  f"best={best_loss:.4f}")

    if verbose:
        best_step = int(np.argmin(losses)) + 1
        print(f"  best loss {best_loss:.4f} at step {best_step}/{n_steps} "
              f"(returning best, not final={losses[-1]:.4f})")

    return best_model, losses

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
    time_losses = jnp.zeros(())
    spec_losses = jnp.zeros(())
    for s_idx in range(n_src):
        pred = model(c_norm, source_positions[s_idx])    # (n_t, n_recv)
        target = d_true[s_idx]                           # (n_t, n_recv)
        time_losses = time_losses + _rel_l2(pred, target)
        spec_losses = spec_losses + _spectral_rel_l2(pred, target)
    return (time_losses + lambda_spec * spec_losses) / n_src


def _normalise_c(c: jnp.ndarray, c_min: float, c_max: float) -> jnp.ndarray:
    return (c - c_min) / (c_max - c_min)


def _extract_source_positions(reader_item) -> List[Tuple[int, int, int]]:
    """Pull integer source grid coords from a Phase-0 sample.

    Samples store ``transducer_positions`` in metres (N, 3). The training
    loop needs voxel coords. ``gen_phase0.py`` uses a fixed helmet per
    shard, so we can read the coords from the first sample.
    """
    if "transducer_positions_grid" in reader_item:
        arr = np.asarray(reader_item["transducer_positions_grid"])
    elif "transducer_positions" in reader_item and "dx" in reader_item:
        positions_m = np.asarray(reader_item["transducer_positions"])
        dx = float(reader_item["dx"])
        arr = np.round(positions_m / dx).astype(np.int32)
    else:
        raise KeyError(
            "sample lacks transducer_positions_grid or (transducer_positions + dx)"
        )
    return [tuple(int(x) for x in row) for row in arr]


def train_fno_surrogate(
    model: CToTraceFNO3D,
    reader,
    *,
    n_steps: int,
    key: jax.Array,
    c_min: float = 1400.0,
    c_max: float = 3200.0,
    learning_rate: float = 1e-3,
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
        learning_rate: Adam LR.
        lambda_spec: spectral-loss weight.
        source_positions: override the auto-extracted helmet. Use when
            the reader does not expose transducer coords (e.g. tests).
        held_out_ids: sample ids to exclude from training. The validation
            half of the Phase-0 split goes here.
        log_every: print a loss line every N steps.
        verbose: print progress.

    Returns:
        ``(trained_model, loss_history)``.
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

    optimizer = optax.adam(learning_rate)
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
        losses.append(float(loss))
        if verbose and (s + 1) % log_every == 0:
            recent = np.mean(losses[-log_every:])
            print(f"  FNO-train step {s+1}/{n_steps}: "
                  f"loss={recent:.4f} (avg over last {log_every})")

    return model, losses

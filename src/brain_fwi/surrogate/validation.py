"""Validation harness for the FNO surrogate (Phase 4 §10 step 4).

Implements the Phase 4 §7.2 trace-fidelity gate and §7.3 gradient-
accuracy gate. Callers hand in a trained surrogate + a held-out set of
``(c, d_true)`` pairs; the functions return metric dicts that downstream
tooling can assert against.

Gates (per the design doc):

§7.2 Trace-level accuracy
    - Median relative-L2 per trace < **1%**
    - 95th-percentile relative-L2 per trace < **5%**
    - Spectral-ratio within ±10% across bands.

§7.3 Gradient accuracy (only if §6B is a target)
    - Cosine similarity of ∂F_φ/∂c vs ∂F_jwave/∂c > **0.95**
      averaged over held-out ``c`` samples.

Both gates are reusable by `scripts/modal_train_fno.py` and any
downstream validation script that needs the same contracts.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .fno3d import CToTraceFNO3D


def _per_trace_rel_l2(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Rel-L2 per (shot, receiver) pair.

    ``pred`` and ``target`` shape: ``(n_src, n_t, n_recv)``. Reduction is
    over the time axis only so each trace gets its own score.
    """
    num = jnp.sqrt(jnp.sum((pred - target) ** 2, axis=1))       # (n_src, n_recv)
    den = jnp.sqrt(jnp.sum(target ** 2, axis=1)) + eps
    return num / den


def _predict_all_shots(
    model: CToTraceFNO3D,
    c_norm: jnp.ndarray,
    source_positions: Sequence[Tuple[int, int, int]],
) -> jnp.ndarray:
    """Stack per-shot outputs into ``(n_src, n_t, n_recv)``."""
    return jnp.stack(
        [model(c_norm, src) for src in source_positions],
        axis=0,
    )


def trace_fidelity(
    model: CToTraceFNO3D,
    samples: Sequence[Dict],
    source_positions: Sequence[Tuple[int, int, int]],
    *,
    c_min: float = 1400.0,
    c_max: float = 3200.0,
) -> Dict[str, float]:
    """§7.2 trace-fidelity metrics on a held-out batch.

    Args:
        model: trained surrogate.
        samples: list of dicts with ``sound_speed_voxel`` and
            ``observed_data`` fields.
        source_positions: helmet voxel coords.
        c_min, c_max: velocity bounds for normalisation.

    Returns:
        Dict with ``median_rel_l2``, ``p95_rel_l2``, ``mean_rel_l2``,
        ``spectral_ratio`` (mean |FFT(pred)| / mean |FFT(target)| per
        sample, averaged). Also ``gate_pass_7_2`` (bool): median<1% AND
        p95<5% AND spectral-ratio within ±10%.
    """
    all_rel_l2 = []
    spectral_ratios = []
    for s in samples:
        c = jnp.asarray(s["sound_speed_voxel"], dtype=jnp.float32)
        d = jnp.asarray(s["observed_data"], dtype=jnp.float32)
        c_norm = (c - c_min) / (c_max - c_min)

        pred = _predict_all_shots(model, c_norm, source_positions)
        rel = _per_trace_rel_l2(pred, d)
        all_rel_l2.append(np.asarray(rel).ravel())

        # Spectral-ratio: per-sample mean magnitude ratio across the
        # time axis, then average over shots/receivers.
        f_pred = jnp.fft.rfft(pred, axis=1)
        f_tgt = jnp.fft.rfft(d, axis=1)
        ratio = jnp.mean(jnp.abs(f_pred)) / (jnp.mean(jnp.abs(f_tgt)) + 1e-12)
        spectral_ratios.append(float(ratio))

    flat = np.concatenate(all_rel_l2)
    median = float(np.median(flat))
    p95 = float(np.percentile(flat, 95))
    mean = float(np.mean(flat))
    mean_spec = float(np.mean(spectral_ratios))
    gate_pass = (median < 0.01) and (p95 < 0.05) and (0.9 <= mean_spec <= 1.1)

    return {
        "n_samples": len(samples),
        "n_traces": int(flat.size),
        "median_rel_l2": median,
        "p95_rel_l2": p95,
        "mean_rel_l2": mean,
        "spectral_ratio": mean_spec,
        "gate_pass_7_2": bool(gate_pass),
    }


def _flatten_grad(g) -> np.ndarray:
    return np.asarray(g).ravel()


def gradient_accuracy(
    surrogate_forward: Callable[[jnp.ndarray], jnp.ndarray],
    jwave_forward: Callable[[jnp.ndarray], jnp.ndarray],
    c_samples: Sequence[jnp.ndarray],
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray] = None,
) -> Dict[str, float]:
    """§7.3 gradient-accuracy gate.

    For each ``c`` sample, compute ``∂L(F_φ(c))/∂c`` and
    ``∂L(F_jwave(c))/∂c`` where ``L`` is a scalar reduction (default
    ``sum(y ** 2)``). Report cosine similarity per sample and in
    aggregate.

    Args:
        surrogate_forward: ``c → y`` callable (takes a velocity field,
            returns whatever ``loss_fn`` reduces to a scalar).
        jwave_forward: ground-truth ``c → y`` callable.
        c_samples: list/sequence of ``c`` arrays.
        loss_fn: scalar reduction. Defaults to ``sum(y ** 2)``.

    Returns:
        Dict with ``mean_cosine``, ``min_cosine``, ``median_cosine``,
        ``per_sample_cosine`` list, ``gate_pass_7_3`` (bool:
        mean > 0.95).
    """
    if loss_fn is None:
        def loss_fn(y: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(y ** 2)

    def _grad(fwd, c):
        return jax.grad(lambda cc: loss_fn(fwd(cc)))(c)

    sims: list = []
    for c in c_samples:
        g_phi = _flatten_grad(_grad(surrogate_forward, c))
        g_jw = _flatten_grad(_grad(jwave_forward, c))
        denom = (np.linalg.norm(g_phi) * np.linalg.norm(g_jw)) + 1e-12
        sims.append(float(np.dot(g_phi, g_jw) / denom))

    arr = np.asarray(sims)
    mean_cos = float(arr.mean())
    return {
        "n_samples": len(c_samples),
        "per_sample_cosine": sims,
        "mean_cosine": mean_cos,
        "median_cosine": float(np.median(arr)),
        "min_cosine": float(arr.min()),
        "gate_pass_7_3": bool(mean_cos > 0.95),
    }


def format_gate_report(trace_metrics: Dict, grad_metrics: Dict = None) -> str:
    """Human-readable summary for logs / PR comments."""
    lines = ["=== Phase 4 §7.2 trace-fidelity gate ==="]
    lines.append(f"  n_samples       = {trace_metrics['n_samples']}")
    lines.append(f"  n_traces        = {trace_metrics['n_traces']}")
    lines.append(f"  median rel-L2   = {trace_metrics['median_rel_l2']:.4%}  "
                 f"(gate: <1%)")
    lines.append(f"  p95    rel-L2   = {trace_metrics['p95_rel_l2']:.4%}  "
                 f"(gate: <5%)")
    lines.append(f"  mean   rel-L2   = {trace_metrics['mean_rel_l2']:.4%}")
    lines.append(f"  spectral ratio  = {trace_metrics['spectral_ratio']:.4f}  "
                 f"(gate: 0.9-1.1)")
    verdict = "PASS" if trace_metrics['gate_pass_7_2'] else "FAIL"
    lines.append(f"  §7.2 verdict    = {verdict}")

    if grad_metrics is not None:
        lines.append("")
        lines.append("=== Phase 4 §7.3 gradient-accuracy gate ===")
        lines.append(f"  n_samples       = {grad_metrics['n_samples']}")
        lines.append(f"  mean cosine     = {grad_metrics['mean_cosine']:.4f}  "
                     f"(gate: >0.95)")
        lines.append(f"  median cosine   = {grad_metrics['median_cosine']:.4f}")
        lines.append(f"  min    cosine   = {grad_metrics['min_cosine']:.4f}")
        verdict = "PASS" if grad_metrics['gate_pass_7_3'] else "FAIL"
        lines.append(f"  §7.3 verdict    = {verdict}")

    return "\n".join(lines)

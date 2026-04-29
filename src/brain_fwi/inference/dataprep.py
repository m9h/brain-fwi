"""Sample → (theta, d) conversion for learning-based FWI inference.

Phase-0 samples (produced by ``scripts/gen_phase0.py``) store SIREN
weights as an opaque byte blob plus a small architecture descriptor.
These helpers reverse that encoding so training code can consume
(theta, d) pairs as plain arrays.

Two entry points:

  - :func:`siren_from_sample` — reconstruct the full :class:`SIREN`
    module. Use for decoding theta back into a velocity field at
    inference time (``siren.to_velocity(...)`` via :class:`SIRENField`).
  - :func:`theta_from_sample` — flatten SIREN's inexact-array leaves
    into a 1-D vector. Use as the posterior target for NPE / the
    denoising target for diffusion priors.

Both accept ``siren_arch`` as either a dict (in-memory samples) or a
JSON string (samples freshly loaded from HDF5 via ``ShardedReader``,
where dict-valued attrs are stored as JSON strings by the writer).
"""

from __future__ import annotations

import io
import json
from typing import Any, List, Literal, Mapping, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..inversion.param_field import SIREN


_REQUIRED_FIELDS = ("siren_weights_bytes", "siren_arch")


def _parse_arch(raw: Union[Mapping[str, Any], str, bytes]) -> dict:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def siren_from_sample(sample: Mapping[str, Any]) -> SIREN:
    """Reconstruct the :class:`SIREN` module stored in ``sample``."""
    for f in _REQUIRED_FIELDS:
        if f not in sample:
            raise KeyError(f"sample missing required field {f!r}")

    arch = _parse_arch(sample["siren_arch"])
    weights_bytes = np.asarray(sample["siren_weights_bytes"], dtype=np.uint8)

    template = SIREN(
        in_dim=int(arch["in_dim"]),
        hidden_dim=int(arch["hidden_dim"]),
        n_hidden=int(arch["n_hidden"]),
        out_dim=int(arch["out_dim"]),
        omega_0=float(arch["omega_0"]),
        key=jax.random.PRNGKey(0),  # overwritten by deserialise
    )
    buf = io.BytesIO(weights_bytes.tobytes())
    return eqx.tree_deserialise_leaves(buf, template)


def theta_from_sample(sample: Mapping[str, Any]) -> np.ndarray:
    """Flatten SIREN inexact-array leaves into a 1-D theta vector.

    The returned vector is the concatenation of ravelled weight and bias
    tensors, layer by layer, in the order Equinox's pytree traversal
    yields them. Deterministic for a given architecture, so samples
    from the same dataset share a consistent theta-coordinate system.
    """
    siren = siren_from_sample(sample)
    leaves = jax.tree.leaves(eqx.filter(siren, eqx.is_inexact_array))
    return np.concatenate([np.asarray(l).ravel() for l in leaves])


# ---------------------------------------------------------------------------
# Summary statistics on observed_data
# ---------------------------------------------------------------------------

SummaryMethod = Literal["max_abs"]


def summary_d_from_sample(
    sample: Mapping[str, Any],
    method: SummaryMethod = "max_abs",
) -> np.ndarray:
    """Reduce ``sample['observed_data']`` to a flat NPE-input feature vector.

    Phase-0 ``observed_data`` has shape ``(N_src, N_t, N_recv)`` — too
    large for a normalizing flow to consume directly. This returns a
    summary whose shape is independent of ``N_t``, so different datasets
    produce d-vectors in a fixed coordinate system.

    Methods:
        ``"max_abs"`` — peak absolute pressure per (source, receiver)
            pair, flattened to shape ``(N_src * N_recv,)``. Simplest
            useful summary; preserves source/receiver structure without
            the time-axis explosion. Insufficient for full FWI-quality
            inference but good enough for Phase 2 plumbing validation.
    """
    if "observed_data" not in sample:
        raise KeyError("sample missing required field 'observed_data'")

    obs = np.asarray(sample["observed_data"])
    if method == "max_abs":
        return np.max(np.abs(obs), axis=1).ravel().astype(np.float32)
    raise ValueError(f"unknown summary method {method!r}")


# ---------------------------------------------------------------------------
# Matrix building
# ---------------------------------------------------------------------------

def build_theta_d_matrix(
    reader,
    d_method: SummaryMethod = "max_abs",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Iterate a ``ShardedReader`` and stack ``(theta, d)`` rows.

    Args:
        reader: Iterable yielding sample dicts with ``sample_id``,
            ``siren_weights_bytes``, ``siren_arch``, and
            ``observed_data`` fields (typically a
            :class:`brain_fwi.data.ShardedReader`).
        d_method: Summary statistic for the observation reduction.

    Returns:
        Tuple ``(theta, d, sample_ids)``:
          - ``theta`` is ``(N, theta_dim)`` float32
          - ``d`` is ``(N, d_dim)`` float32
          - ``sample_ids`` is a list of length ``N`` preserving reader order.
    """
    theta_rows: List[np.ndarray] = []
    d_rows: List[np.ndarray] = []
    ids: List[str] = []

    for sample in reader:
        theta_rows.append(theta_from_sample(sample).astype(np.float32))
        d_rows.append(summary_d_from_sample(sample, method=d_method))
        ids.append(str(sample["sample_id"]))

    if not theta_rows:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            [],
        )

    return (
        np.stack(theta_rows, axis=0),
        np.stack(d_rows, axis=0),
        ids,
    )


def load_theta_matrix(reader, sample_ids=None) -> jax.Array:
    """Stack θ vectors from a Phase-0 reader into ``(N, D)``.

    Phase 3 step-1 helper (``docs/design/phase3_diffusion_prior.md``
    §9). Bridge from a ``ShardedReader`` to the matrix consumed by
    ``brain_fwi.inference.diffusion.train_score_matching``.

    Args:
        reader: anything with sample dicts. If ``sample_ids`` is given,
            ``reader[sid]`` must work; otherwise the reader is iterated.
        sample_ids: optional subset of sample ids to load (list of str
            or list of ints).

    Returns:
        ``(N, D)`` float32 ``jnp.ndarray``.

    Raises:
        ValueError if the loaded samples have heterogeneous θ
        dimensions (e.g. mixed SIREN architectures — the
        ``hidden=128`` vs ``hidden=64`` collision the Phase-0b
        handoff doc warned about).
    """
    if sample_ids is not None:
        rows = [theta_from_sample(reader[sid]) for sid in sample_ids]
    else:
        rows = [theta_from_sample(s) for s in reader]
    if not rows:
        raise ValueError("no samples to load")

    dims = {r.shape[0] for r in rows}
    if len(dims) > 1:
        raise ValueError(
            f"heterogeneous θ dimensions in dataset: {sorted(dims)}. "
            "All samples must use the same SIREN architecture."
        )
    return jnp.stack(
        [jnp.asarray(r, dtype=jnp.float32) for r in rows], axis=0,
    )

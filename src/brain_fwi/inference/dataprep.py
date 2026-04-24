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
from typing import Any, Mapping, Union

import equinox as eqx
import jax
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

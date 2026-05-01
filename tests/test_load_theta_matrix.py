"""TDD spec for the Phase 3 §9 step 1 data loader.

``load_theta_matrix`` is the bridge from a Phase-0 ``ShardedReader``
to the ``(N, D)`` θ matrix consumed by ``train_score_matching``.
"""

from __future__ import annotations

import io
import json
from typing import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from brain_fwi.inversion.param_field import SIREN

ARCH_A = {"in_dim": 3, "hidden_dim": 16, "n_hidden": 2, "out_dim": 1, "omega_0": 30.0}
ARCH_B = {"in_dim": 3, "hidden_dim": 32, "n_hidden": 2, "out_dim": 1, "omega_0": 30.0}


def _sample_from_siren(model: SIREN, arch: Mapping, sample_id: str) -> dict:
    buf = io.BytesIO()
    eqx.tree_serialise_leaves(buf, model)
    return {
        "sample_id": sample_id,
        "siren_weights_bytes": np.frombuffer(buf.getvalue(), dtype=np.uint8),
        "siren_arch": dict(arch),
    }


class _MockReader:
    def __init__(self, samples: dict):
        self._samples = samples
        self.sample_ids = list(samples.keys())

    def __getitem__(self, sid: str):
        return self._samples[sid]

    def __iter__(self):
        return iter(self._samples.values())


def test_load_theta_matrix_stacks_samples_with_same_arch():
    """N samples with identical SIREN arch → ``(N, D)`` float32 matrix."""
    from brain_fwi.inference.dataprep import load_theta_matrix

    samples = {}
    for i in range(5):
        siren = SIREN(
            in_dim=ARCH_A["in_dim"], hidden_dim=ARCH_A["hidden_dim"],
            n_hidden=ARCH_A["n_hidden"], out_dim=ARCH_A["out_dim"],
            omega_0=ARCH_A["omega_0"], key=jr.PRNGKey(i),
        )
        samples[f"s{i}"] = _sample_from_siren(siren, ARCH_A, f"s{i}")
    reader = _MockReader(samples)

    theta = load_theta_matrix(reader)
    assert theta.shape[0] == 5
    assert theta.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(theta))


def test_load_theta_matrix_rejects_heterogeneous_arch():
    """Mixed SIREN architectures (different θ-dim) ⇒ ValueError.

    Phase-0b handoff explicitly warned about this collision: the
    existing 100-sample dataset is at hidden=128 (~50k θ-dim) and
    the new run is hidden=64 (~12.8k). Phase 2/3 cannot consume a
    heterogeneous mix; the loader must catch it loudly.
    """
    from brain_fwi.inference.dataprep import load_theta_matrix

    siren_a = SIREN(
        in_dim=ARCH_A["in_dim"], hidden_dim=ARCH_A["hidden_dim"],
        n_hidden=ARCH_A["n_hidden"], out_dim=ARCH_A["out_dim"],
        omega_0=ARCH_A["omega_0"], key=jr.PRNGKey(0),
    )
    siren_b = SIREN(
        in_dim=ARCH_B["in_dim"], hidden_dim=ARCH_B["hidden_dim"],
        n_hidden=ARCH_B["n_hidden"], out_dim=ARCH_B["out_dim"],
        omega_0=ARCH_B["omega_0"], key=jr.PRNGKey(1),
    )
    samples = {
        "a": _sample_from_siren(siren_a, ARCH_A, "a"),
        "b": _sample_from_siren(siren_b, ARCH_B, "b"),
    }
    reader = _MockReader(samples)

    with pytest.raises(ValueError, match="heterogeneous"):
        load_theta_matrix(reader)


def test_load_theta_matrix_subset_via_sample_ids():
    """``sample_ids`` selects a subset and preserves order."""
    from brain_fwi.inference.dataprep import load_theta_matrix

    samples = {}
    for i in range(4):
        siren = SIREN(
            in_dim=ARCH_A["in_dim"], hidden_dim=ARCH_A["hidden_dim"],
            n_hidden=ARCH_A["n_hidden"], out_dim=ARCH_A["out_dim"],
            omega_0=ARCH_A["omega_0"], key=jr.PRNGKey(i),
        )
        samples[f"s{i}"] = _sample_from_siren(siren, ARCH_A, f"s{i}")
    reader = _MockReader(samples)

    full = load_theta_matrix(reader)
    sub = load_theta_matrix(reader, sample_ids=["s2", "s0"])
    assert sub.shape == (2, full.shape[1])
    # order matches the supplied sample_ids: s2 first, then s0
    assert jnp.allclose(sub[0], full[2])
    assert jnp.allclose(sub[1], full[0])


def test_load_theta_matrix_empty_reader_raises():
    from brain_fwi.inference.dataprep import load_theta_matrix

    reader = _MockReader({})
    with pytest.raises(ValueError, match="no samples"):
        load_theta_matrix(reader)

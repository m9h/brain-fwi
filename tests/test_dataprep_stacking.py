"""Tests for Phase 0 → Phase 2 data-prep stacking.

Covers:
  - ``summary_d_from_sample`` produces a fixed-size d-vector from each
    sample's raw ``observed_data``. Must be deterministic and have
    shape independent of time-axis length.
  - ``build_theta_d_matrix`` iterates a ShardedReader and stacks
    matching ``(theta, d)`` rows. Used as the training-data entry
    point for ``train_npe``.
"""

from pathlib import Path

import io
import numpy as np
import pytest

import equinox as eqx
import jax.random as jr

from brain_fwi.data import ShardedReader, ShardedWriter
from brain_fwi.inference.dataprep import (
    build_theta_d_matrix,
    summary_d_from_sample,
)
from brain_fwi.inversion.param_field import SIREN


# ---------------------------------------------------------------------------
# Fixture factory
# ---------------------------------------------------------------------------

def _siren_bytes(siren: SIREN) -> np.ndarray:
    buf = io.BytesIO()
    eqx.tree_serialise_leaves(buf, siren)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8).copy()


def _mk_sample(sample_id: str, n_src=4, n_t=50, n_recv=8, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    arch = {"in_dim": 3, "hidden_dim": 16, "n_hidden": 1, "out_dim": 1,
            "omega_0": 30.0}
    siren = SIREN(in_dim=3, hidden_dim=16, n_hidden=1, out_dim=1,
                  omega_0=30.0, key=jr.PRNGKey(seed))
    return {
        "sample_id": sample_id,
        "observed_data": rng.standard_normal((n_src, n_t, n_recv)).astype(np.float32),
        "siren_weights_bytes": _siren_bytes(siren),
        "siren_arch": arch,
    }


@pytest.fixture
def small_phase0_dataset(tmp_path: Path) -> Path:
    with ShardedWriter(tmp_path, shard_size=10) as w:
        for i in range(6):
            w.write(_mk_sample(f"s{i:02d}", seed=i))
    return tmp_path


# ---------------------------------------------------------------------------
# summary_d_from_sample
# ---------------------------------------------------------------------------

class TestSummaryD:
    def test_max_abs_shape(self):
        s = _mk_sample("x", n_src=4, n_t=50, n_recv=8)
        d = summary_d_from_sample(s, method="max_abs")
        assert d.shape == (4 * 8,)

    def test_max_abs_values(self):
        """max_abs is the peak absolute amplitude per (source, receiver)
        pair, flattened."""
        sample = _mk_sample("x", n_src=2, n_t=10, n_recv=3)
        obs = sample["observed_data"]
        d = summary_d_from_sample(sample, method="max_abs")
        expected = np.max(np.abs(obs), axis=1).ravel()
        np.testing.assert_allclose(d, expected, rtol=1e-6)

    def test_invariant_to_timestep_count(self):
        """Whatever the observed_data time dim, summary output has the
        same shape (determined only by n_src × n_recv)."""
        s1 = _mk_sample("x", n_src=4, n_t=30, n_recv=8)
        s2 = _mk_sample("x", n_src=4, n_t=200, n_recv=8)
        assert summary_d_from_sample(s1).shape == summary_d_from_sample(s2).shape

    def test_unknown_method_raises(self):
        s = _mk_sample("x")
        with pytest.raises(ValueError, match="method"):
            summary_d_from_sample(s, method="no_such_method")

    def test_missing_observed_data_raises(self):
        with pytest.raises(KeyError, match="observed_data"):
            summary_d_from_sample({"sample_id": "x"})


# ---------------------------------------------------------------------------
# build_theta_d_matrix
# ---------------------------------------------------------------------------

class TestBuildThetaD:
    def test_shape(self, small_phase0_dataset: Path):
        reader = ShardedReader(small_phase0_dataset)
        theta, d, ids = build_theta_d_matrix(reader, d_method="max_abs")

        # 6 samples, theta-dim determined by SIREN architecture, d-dim by n_src*n_recv
        assert theta.shape[0] == 6
        assert d.shape[0] == 6
        assert theta.ndim == 2
        assert d.ndim == 2
        assert len(ids) == 6

    def test_ids_match_manifest_order(self, small_phase0_dataset: Path):
        reader = ShardedReader(small_phase0_dataset)
        _, _, ids = build_theta_d_matrix(reader)
        assert ids == reader.sample_ids

    def test_different_seeds_give_different_theta(self, small_phase0_dataset: Path):
        reader = ShardedReader(small_phase0_dataset)
        theta, _, _ = build_theta_d_matrix(reader)
        # Samples s00..s05 were built with different seeds → SIREN weights differ
        assert not np.allclose(theta[0], theta[1])

    def test_empty_dataset_returns_empty_matrices(self, tmp_path: Path):
        ShardedWriter(tmp_path).close()
        reader = ShardedReader(tmp_path)
        theta, d, ids = build_theta_d_matrix(reader)
        assert theta.shape[0] == 0
        assert d.shape[0] == 0
        assert ids == []

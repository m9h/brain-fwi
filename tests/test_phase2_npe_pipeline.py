"""Phase-2 NPE training-pipeline contract tests against the v2a dataset.

TDD red phase: these tests run end-to-end Phase-2 plumbing on real
production data (or skip cleanly if it's not on disk). Failures here are
Phase-0 → Phase-2 data-contract mismatches, not bugs in flowjax itself.

Pulled into a separate file from the sample-level contract tests because
these touch the conditional flow + multi-sample stacking, not just the
shard schema. Skip the whole module when flowjax is missing — flowjax has
no Mac wheel for some Python versions, and we want the harness to stay
green even on a developer machine that lacks it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ShardedReader is importable on every host; flow / NPE imports are heavier
# and may not have wheels everywhere.
flowjax = pytest.importorskip("flowjax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402

from brain_fwi.data import ShardedReader  # noqa: E402
from brain_fwi.inference.dataprep import build_theta_d_matrix  # noqa: E402
from brain_fwi.inference.flow import ConditionalFlow, train_npe  # noqa: E402


_DATASET_CANDIDATES = (
    Path("/tmp/phase0_v2a_prod/merged"),
    Path("/tmp/phase0_smoke4/merged"),
)


def _find_dataset() -> Path | None:
    for p in _DATASET_CANDIDATES:
        if (p / "manifest.json").exists() and (p / "shards").exists():
            return p
    return None


@pytest.fixture(scope="module")
def reader() -> ShardedReader:
    root = _find_dataset()
    if root is None:
        pytest.skip("no Phase-0 dataset on disk")
    return ShardedReader(root)


# ---------------------------------------------------------------------------
# Stage 1: build_theta_d_matrix on a real reader
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def theta_d(reader: ShardedReader):
    """Build (theta, d) on first 8 samples — keeps the test < 30s on CPU."""
    ids = list(reader.sample_ids)[:8]
    subset = [reader[sid] for sid in ids]
    return build_theta_d_matrix(iter(subset))


class TestBuildThetaD:
    """The Phase-0 → Phase-2 stacking step. Surfaces shape / NaN issues."""

    def test_returns_three_outputs(self, theta_d):
        assert len(theta_d) == 3, "build_theta_d_matrix must return (theta, d, ids)"

    def test_theta_is_2d(self, theta_d):
        theta, _, _ = theta_d
        assert theta.ndim == 2, f"theta rank {theta.ndim} != 2"

    def test_d_is_2d(self, theta_d):
        _, d, _ = theta_d
        assert d.ndim == 2, f"d rank {d.ndim} != 2"

    def test_n_samples_matches_ids(self, theta_d):
        theta, d, ids = theta_d
        assert theta.shape[0] == len(ids) == d.shape[0]

    def test_theta_finite(self, theta_d):
        theta, _, _ = theta_d
        assert np.all(np.isfinite(theta)), "theta has NaN or inf — SIREN serialization issue"

    def test_d_finite(self, theta_d):
        _, d, _ = theta_d
        assert np.all(np.isfinite(d)), (
            "d has NaN or inf — observed_data summary inherited the air-cavity NaN bug"
        )

    def test_d_nonzero(self, theta_d):
        _, d, _ = theta_d
        assert float(np.max(np.abs(d))) > 0.0, "d is identically zero"

    def test_theta_dim_reasonable(self, theta_d):
        """Production SIREN at hidden=128, layers=3 has ~50k weights.

        If theta_dim is wildly off, downstream NPE training will OOM or
        produce a useless flow.
        """
        theta, _, _ = theta_d
        assert 1000 < theta.shape[1] < 200_000, (
            f"theta_dim {theta.shape[1]} outside expected SIREN-weight range"
        )

    def test_d_dim_reasonable(self, theta_d):
        """Production helmet has 128 sources × 128 receivers = 16384 entries
        from max_abs summary (per src × recv, time-axis collapsed)."""
        _, d, _ = theta_d
        assert d.shape[1] == 128 * 128, (
            f"d_dim {d.shape[1]} != 16384 — helmet geometry drift?"
        )

    def test_ids_are_unique(self, theta_d):
        _, _, ids = theta_d
        assert len(set(ids)) == len(ids), "duplicate sample ids in theta/d"


# ---------------------------------------------------------------------------
# Stage 2: ConditionalFlow construction + 1 training step
# ---------------------------------------------------------------------------


class TestConditionalFlow:
    """The flow itself must accept Phase-0-shaped (theta, d)."""

    def test_flow_constructs_with_phase0_dims(self, theta_d):
        theta, d, _ = theta_d
        flow = ConditionalFlow(
            theta_dim=theta.shape[1],
            d_dim=d.shape[1],
            key=jr.PRNGKey(0),
            n_transforms=2,
            nn_width=16,
            nn_depth=2,
        )
        assert flow.theta_dim == theta.shape[1]
        assert flow.d_dim == d.shape[1]

    def test_log_prob_returns_finite_scalar(self, theta_d):
        theta, d, _ = theta_d
        flow = ConditionalFlow(
            theta_dim=theta.shape[1], d_dim=d.shape[1],
            key=jr.PRNGKey(0),
            n_transforms=2, nn_width=16, nn_depth=2,
        )
        lp = float(flow.log_prob(theta[0], d[0]))
        assert np.isfinite(lp), f"flow log_prob is not finite: {lp}"


# ---------------------------------------------------------------------------
# Stage 3: train_npe — single-step smoke
# ---------------------------------------------------------------------------


class TestTrainNpeOneStep:
    """A single training step must execute without NaN/inf in loss or params.

    With v2a's 8-sample slice, training won't converge in 5 steps — this
    is a smoke check that the gradient pipeline is wired up correctly.
    """

    def test_one_step_loss_finite(self, theta_d):
        theta, d, _ = theta_d
        flow = ConditionalFlow(
            theta_dim=theta.shape[1], d_dim=d.shape[1],
            key=jr.PRNGKey(0),
            n_transforms=2, nn_width=16, nn_depth=2,
        )
        # train_npe accepts batch_size; force batch=full to keep test
        # determinism + speed.
        trained, losses = train_npe(
            flow, jnp.asarray(theta), jnp.asarray(d),
            key=jr.PRNGKey(1),
            n_steps=1, learning_rate=1e-3,
            batch_size=min(4, theta.shape[0]),
        )
        assert len(losses) == 1
        assert np.isfinite(losses[0]), f"first-step loss not finite: {losses[0]}"

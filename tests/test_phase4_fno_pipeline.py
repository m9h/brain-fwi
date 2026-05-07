"""Phase-4 FNO surrogate-training pipeline contract tests against v2a.

TDD red phase. Surfaces the issues a real Phase-0-fed FNO trainer must
deal with:

* `CToTraceFNO3D` bakes ``n_timesteps`` into its output shape, but
  Phase-0 produces a different ``n_timesteps`` per sample (depends on
  jittered c — CFL-derived dt → variable t_end / dt steps). A fixed-N_t
  model can't consume a variable-N_t dataset without a stacking strategy.

* The Phase-0 sample dict and the trainer-expected reader interface must
  agree on field names, dtypes, and grid shape.

These tests do NOT actually train (no GPU) — they construct the model,
exercise one forward pass, and validate the data contract for stacking.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from brain_fwi.data import ShardedReader
from brain_fwi.surrogate.fno3d import CToTraceFNO3D


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


@pytest.fixture(scope="module")
def first_sample(reader):
    return reader[next(iter(reader.sample_ids))]


# ---------------------------------------------------------------------------
# Stage 1: dataset-shape contract for FNO
# ---------------------------------------------------------------------------


class TestPhase0HasFnoFields:
    """Trainer needs sound_speed_voxel + observed_data per sample."""

    def test_sound_speed_voxel_3d_float(self, first_sample):
        c = np.asarray(first_sample["sound_speed_voxel"]).astype(np.float32)
        assert c.ndim == 3, f"sound_speed_voxel rank {c.ndim} != 3"
        assert c.dtype == np.float32

    def test_observed_data_is_3d(self, first_sample):
        obs = np.asarray(first_sample["observed_data"])
        assert obs.ndim == 3, "observed_data must be (N_src, N_t, N_recv)"


# ---------------------------------------------------------------------------
# Stage 2: variable n_timesteps across samples — the FNO blocker
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def n_timesteps_distribution(reader):
    """Collect n_timesteps from every sample to expose the variable-N_t issue."""
    counts = []
    for sid in list(reader.sample_ids)[:32]:  # capped to keep test fast
        s = reader[sid]
        counts.append(int(np.asarray(s["observed_data"]).shape[1]))
    return counts


class TestObservedDataTimeAxisConsistency:
    """The FNO output shape is fixed at construction time. Phase-0 samples
    have variable n_timesteps (CFL-derived dt depends on jittered c_max),
    so the trainer in `surrogate.train.train_fno_surrogate` must crop or
    pad to ``model.n_timesteps`` per step. The contract here is that the
    spread is bounded enough that truncating-to-min loses an acceptable
    fraction of each trace.
    """

    def test_n_timesteps_spread_is_bounded(self, n_timesteps_distribution):
        counts = np.asarray(n_timesteps_distribution)
        spread_pct = (counts.max() - counts.min()) / counts.min() * 100
        assert spread_pct < 30.0, (
            f"n_timesteps spread = {spread_pct:.1f}% (min={counts.min()}, "
            f"max={counts.max()}); truncation to min would discard >30% "
            f"of each trace, hurting FNO training quality. Either bound "
            f"the c-jitter at generation time or set model.n_timesteps "
            f"closer to the median (~{int(np.median(counts))})."
        )

    def test_min_n_timesteps_safe_for_model(self, n_timesteps_distribution):
        """A model built with n_timesteps = min(dataset) won't pad, only crop.

        Padding silently teaches the FNO that the late-time signal is zero
        for short samples, which is wrong (it's just unobserved). Cropping
        is the safer default — but only if model.n_timesteps <= min(d.n_t).
        """
        assert min(n_timesteps_distribution) > 100, (
            "min n_timesteps suspiciously low; build_time_axis may have "
            "produced a degenerate trace"
        )


# ---------------------------------------------------------------------------
# Stage 3: CToTraceFNO3D constructs at v2a's grid + can do a forward pass
# ---------------------------------------------------------------------------


class TestFnoModelOnV2a:
    """The model must construct + run on the actual v2a grid, even if
    we have to commit to a single N_t for the architecture."""

    @pytest.fixture(scope="class")
    def model(self, first_sample):
        c = np.asarray(first_sample["sound_speed_voxel"])
        obs = np.asarray(first_sample["observed_data"])
        return CToTraceFNO3D(
            grid_shape=tuple(c.shape),
            n_timesteps=int(obs.shape[1]),
            n_receivers=int(obs.shape[2]),
            hidden_channels=8,   # tiny — local CPU smoke
            num_modes=4,
            depth=1,
            key=jr.PRNGKey(0),
        )

    def test_grid_shape_matches_v2a(self, model, first_sample):
        c = np.asarray(first_sample["sound_speed_voxel"])
        assert model.grid_shape == tuple(c.shape)

    def test_n_receivers_matches_helmet(self, model, first_sample):
        obs = np.asarray(first_sample["observed_data"])
        assert model.n_receivers == int(obs.shape[2])

    def test_forward_pass_runs(self, model, first_sample):
        """Single forward pass on tiny model — verify shape + finiteness."""
        c = jnp.asarray(first_sample["sound_speed_voxel"], dtype=jnp.float32)
        # Normalise c the way training does: [c_min, c_max] -> [0, 1]
        c_norm = jnp.clip((c - 1400.0) / (3200.0 - 1400.0), 0.0, 1.0)
        # Pick the first transducer position as the source for this shot.
        # The model takes one source at a time; the trainer scans across
        # all positions and sums losses.
        n = c.shape[0]
        src_pos_grid = (n // 2, n // 2, n // 4)  # arbitrary interior point
        out = model(c_norm, src_pos_grid=src_pos_grid)
        assert out.shape == (model.n_timesteps, model.n_receivers), (
            f"FNO output shape {out.shape} != (n_t, n_recv) = "
            f"({model.n_timesteps}, {model.n_receivers})"
        )
        assert jnp.all(jnp.isfinite(out)), "FNO produced NaN/inf on a finite c"


# ---------------------------------------------------------------------------
# Stage 4: train_fno_surrogate one-step smoke
# ---------------------------------------------------------------------------


class TestTrainFnoOneStep:
    """A single train step against the v2a reader must execute without crash.

    Will fail at the second sample if N_t varies — exactly the contract
    test we want for the production trainer.
    """

    def test_one_step_does_not_crash(self, reader, first_sample, n_timesteps_distribution):
        # Lazily — train_fno_surrogate is heavy, but a single step is cheap
        # on a tiny model.
        from brain_fwi.surrogate.train import train_fno_surrogate

        c = np.asarray(first_sample["sound_speed_voxel"])
        obs = np.asarray(first_sample["observed_data"])
        # Use min n_t across the sampled dataset so the trainer's truncation
        # path (not pad) kicks in — pad would silently teach FNO that late
        # time-steps are zero for short samples.
        model_n_t = min(n_timesteps_distribution)
        model = CToTraceFNO3D(
            grid_shape=tuple(c.shape),
            n_timesteps=model_n_t,
            n_receivers=int(obs.shape[2]),
            hidden_channels=8,
            num_modes=4,
            depth=1,
            key=jr.PRNGKey(0),
        )
        # Use only the first 2 sample IDs so the test runs in <30s on CPU.
        first_two = list(reader.sample_ids)[:2]

        class _Slice:
            sample_ids = first_two
            def __getitem__(self, sid):
                return reader[sid]
            def __iter__(self):
                return (reader[sid] for sid in first_two)

        try:
            # Don't pass source_positions — let the trainer auto-extract via
            # _extract_source_positions(first_sample). Production callers
            # (scripts/train_fno_on_phase0.py) do the same.
            trained, losses = train_fno_surrogate(
                model, _Slice(),
                n_steps=1,
                key=jr.PRNGKey(1),
                log_every=1, verbose=False,
            )
        except (ValueError, TypeError, IndexError) as e:
            pytest.fail(
                f"train_fno_surrogate failed on its first step against v2a: "
                f"{type(e).__name__}: {e}\n"
                f"This usually means a sample-shape contract violation — "
                f"variable n_timesteps, missing field, or transducer-position "
                f"format mismatch."
            )
        assert len(losses) == 1
        assert np.isfinite(losses[0]), f"first-step loss not finite: {losses[0]}"

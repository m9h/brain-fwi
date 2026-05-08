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

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from brain_fwi.data import ShardedReader
from brain_fwi.surrogate.fno3d import CToTraceFNO3D


def _voxelise_helmet(positions_m: np.ndarray, dx: float, grid_shape: tuple) -> tuple:
    """Convert (N, 3) helmet positions in metres to clamped voxel-grid tuples.

    Mirrors ``brain_fwi.surrogate.train._extract_source_positions``.
    """
    voxels = np.round(positions_m / dx).astype(int)
    nx, ny, nz = grid_shape
    voxels[:, 0] = np.clip(voxels[:, 0], 0, nx - 1)
    voxels[:, 1] = np.clip(voxels[:, 1], 0, ny - 1)
    voxels[:, 2] = np.clip(voxels[:, 2], 0, nz - 1)
    return tuple((int(v[0]), int(v[1]), int(v[2])) for v in voxels)


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
        sensor_pos = _voxelise_helmet(
            np.asarray(first_sample["sensor_positions"]),
            float(first_sample["dx"]),
            tuple(c.shape),
        )
        return CToTraceFNO3D(
            grid_shape=tuple(c.shape),
            n_timesteps=int(obs.shape[1]),
            n_receivers=int(obs.shape[2]),
            hidden_channels=8,   # tiny — local CPU smoke
            num_modes=4,
            depth=1,
            receiver_positions=sensor_pos,
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
        sensor_pos = _voxelise_helmet(
            np.asarray(first_sample["sensor_positions"]),
            float(first_sample["dx"]),
            tuple(c.shape),
        )
        model = CToTraceFNO3D(
            grid_shape=tuple(c.shape),
            n_timesteps=model_n_t,
            n_receivers=int(obs.shape[2]),
            hidden_channels=8,
            num_modes=4,
            depth=1,
            receiver_positions=sensor_pos,
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


# ---------------------------------------------------------------------------
# Stage 5: Point-sample head locality (the v1 global-pool architecture
# could not break out of a predict-zero plateau because all spatial info was
# discarded before the head; the v2 head looks up features at each receiver's
# voxel, preserving spatial structure).
# ---------------------------------------------------------------------------


class TestPointSampleHead:
    """Catches regressions to a global-pool readout. Without point-sampling
    the FNO collapses to a constant prediction (loss ~ 1.30 plateau) because
    receivers can't be distinguished by anything except a fixed per-receiver
    output weight in the head's last linear layer."""

    GRID = (16, 16, 16)
    N_T = 32
    HIDDEN = 8

    def _build(self, receiver_positions, *, key=0):
        return CToTraceFNO3D(
            grid_shape=self.GRID,
            n_timesteps=self.N_T,
            n_receivers=len(receiver_positions),
            hidden_channels=self.HIDDEN,
            num_modes=4,
            depth=1,
            receiver_positions=receiver_positions,
            key=jr.PRNGKey(key),
        )

    def test_constructor_requires_receiver_positions(self):
        """Without receiver positions the model can't point-sample, so the
        new constructor must reject calls that omit the arg."""
        with pytest.raises(TypeError):
            CToTraceFNO3D(
                grid_shape=self.GRID,
                n_timesteps=self.N_T,
                n_receivers=2,
                hidden_channels=self.HIDDEN,
                num_modes=4,
                depth=1,
                key=jr.PRNGKey(0),
            )  # type: ignore[call-arg]

    def test_receiver_positions_baked_in_as_static(self):
        """The receiver coordinates must be static so they don't show up
        as differentiable leaves and don't drift during training."""
        model = self._build(((4, 4, 4), (12, 12, 12)))
        leaves = jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
        # Voxel coords are integers — the JAX inexact-array filter should
        # exclude them entirely. Any leaked float receiver leaf would be
        # a regression.
        for leaf in leaves:
            assert leaf.dtype.kind in ("f", "c"), leaf
            # Sanity: make sure no leaf is suspiciously shaped like a
            # (n_recv, 3) coordinate array.
            assert leaf.shape != (2, 3), (
                f"receiver_positions leaked as a trainable leaf: {leaf.shape}"
            )

    def test_local_perturbation_affects_local_receiver_more(self):
        """Local change in the c-field near receiver 0 should affect
        rec 0's trace MORE than rec 1's. Under the old global-pool head
        any input change propagates to all receivers' traces equally
        (they share the same pooled feature vector), so the inequality
        below holds only for a spatial-gather head."""
        rec_pos = ((4, 4, 4), (12, 12, 12))
        model = self._build(rec_pos)
        c0 = jnp.full(self.GRID, 0.4, dtype=jnp.float32)
        # Bump c-field around receiver 0 by half its range.
        c1 = c0.at[3:6, 3:6, 3:6].set(0.9)
        out0 = model(c0, src_pos_grid=(0, 0, 0))
        out1 = model(c1, src_pos_grid=(0, 0, 0))

        change_at_0 = float(jnp.mean(jnp.abs(out0[:, 0] - out1[:, 0])))
        change_at_1 = float(jnp.mean(jnp.abs(out0[:, 1] - out1[:, 1])))
        assert change_at_0 > 5.0 * change_at_1, (
            f"local c-perturbation near receiver 0 should affect that "
            f"receiver's trace much more than a far receiver's; got "
            f"change_at_0={change_at_0:.4e} vs change_at_1={change_at_1:.4e}. "
            f"This usually means the head is globally pooling features."
        )

    def test_two_receivers_at_same_voxel_produce_same_trace(self):
        """A direct property of point-sampling: two receivers placed at
        the same voxel must read the same feature vector and thus produce
        the same trace. The global-pool head produces *different* traces
        for these because the head's last linear layer assigns different
        output weights per receiver index."""
        rec_pos = ((4, 4, 4), (4, 4, 4))
        model = self._build(rec_pos)
        c = jnp.full(self.GRID, 0.5, dtype=jnp.float32)
        # Use a c-field that has spatial variance so the FNO body
        # produces non-trivial features.
        c = c.at[1:4, 1:4, 1:4].set(0.9)
        out = model(c, src_pos_grid=(0, 0, 0))
        np.testing.assert_allclose(
            np.asarray(out[:, 0]),
            np.asarray(out[:, 1]),
            atol=1e-5,
            err_msg="two receivers at the same voxel produced different traces "
                    "— the head is not point-sampling identically per receiver",
        )

    def test_output_rank_scales_with_n_receivers(self):
        """The old global-pool head had output rank ≤ hidden_channels per
        shot regardless of n_receivers. The new head should have rank up
        to n_receivers (each receiver column is independent)."""
        rec_pos = tuple((i, 8, 8) for i in range(2, 14, 2))  # 6 receivers
        model = self._build(rec_pos)
        c = jnp.linspace(0.0, 1.0, int(np.prod(self.GRID))).reshape(self.GRID)
        out = model(c, src_pos_grid=(0, 0, 0))
        # Compute matrix rank of the (n_t, n_recv) trace tensor — a
        # globally-pooled head with the same head weights would yield
        # rank 1 (every column is a linear scaling of the same source
        # vector). Point-sampling should yield rank > 1 (with non-trivial
        # spatial variation in c).
        rank = int(np.linalg.matrix_rank(np.asarray(out), tol=1e-6))
        assert rank > 1, (
            f"trace tensor has rank {rank} ≤ 1; output is just a global "
            f"signal scaled per receiver — head is not point-sampling"
        )


# ---------------------------------------------------------------------------
# Stage 6: Trainer must return the BEST-loss model and support cosine LR.
# After the FNO production v2 run we observed: loss bottomed at step 193
# (0.48), then bounced upward to 1.04 at step 1000 — and the saved model
# was the step-1000 model, not the actually-good step-193 model. The
# trainer needs to checkpoint the best instead, and a cosine schedule
# stops the late-training divergence in the first place.
# ---------------------------------------------------------------------------


class TestTrainerBestCheckpoint:
    """The trainer must return the model from the lowest-loss step, not
    the model from the final step. With high learning rate or no schedule,
    training can overshoot its useful basin late in the run."""

    def _tiny_model_and_reader(self, reader, first_sample, n_t):
        from brain_fwi.surrogate.train import _extract_receiver_positions
        c = np.asarray(first_sample["sound_speed_voxel"])
        obs = np.asarray(first_sample["observed_data"])
        rec_pos = _extract_receiver_positions(first_sample)
        model = CToTraceFNO3D(
            grid_shape=tuple(c.shape),
            n_timesteps=n_t,
            n_receivers=int(obs.shape[2]),
            hidden_channels=8,
            num_modes=4,
            depth=1,
            receiver_positions=rec_pos,
            key=jr.PRNGKey(0),
        )
        first_two = list(reader.sample_ids)[:2]

        class _Slice:
            sample_ids = first_two
            def __getitem__(self, sid):
                return reader[sid]
            def __iter__(self):
                return (reader[sid] for sid in first_two)

        return model, _Slice()

    def test_trainer_returns_best_loss_model(self, reader, first_sample, n_timesteps_distribution):
        """Run with a high LR + constant schedule to provoke late-stage
        divergence. The returned model's loss on the training set should
        match min(losses), not losses[-1]."""
        from brain_fwi.surrogate.train import (
            train_fno_surrogate, _extract_source_positions, _normalise_c,
            surrogate_loss,
        )
        n_t = min(n_timesteps_distribution)
        model, slice_reader = self._tiny_model_and_reader(reader, first_sample, n_t)
        # 6 steps with a deliberately huge LR to force the loss to bounce.
        trained, losses = train_fno_surrogate(
            model, slice_reader,
            n_steps=6,
            key=jr.PRNGKey(0),
            learning_rate=5e-1,
            lr_schedule="constant",
            log_every=99, verbose=False,
        )
        # Re-evaluate the returned model against the training samples and
        # confirm it sits at-or-below min(losses) — i.e. it isn't the
        # step-N model that bounced.
        src_pos = _extract_source_positions(first_sample)
        sample = first_sample
        c = jnp.asarray(sample["sound_speed_voxel"], dtype=jnp.float32)
        d = jnp.asarray(sample["observed_data"], dtype=jnp.float32)[
            :, :n_t, :
        ]
        c_norm = _normalise_c(c, 1400.0, 3200.0)
        trained_loss = float(surrogate_loss(trained, c_norm, d, src_pos, 0.3))
        # Allow some slack for sampling noise — the assertion is that
        # the returned model is closer to min than to max of the
        # observed loss curve.
        assert trained_loss <= max(losses) - (max(losses) - min(losses)) * 0.5, (
            f"returned model loss {trained_loss:.4f} is closer to the worst "
            f"step (max losses {max(losses):.4f}) than to the best "
            f"(min losses {min(losses):.4f}); trainer is returning the "
            f"final-step model rather than the best one"
        )


class TestTrainerGradientAccumulation:
    """Accumulating gradients across N samples per Adam step should
    behave like a larger batch: lower per-step variance, smoother loss
    curve. Direct invariant we can check: ``accumulation_steps=N``
    consumes N samples per logged step (so the inner loop reads
    ``n_steps * N`` samples total)."""

    def _tiny_model_and_reader(self, reader, first_sample, n_t):
        from brain_fwi.surrogate.train import _extract_receiver_positions
        c = np.asarray(first_sample["sound_speed_voxel"])
        obs = np.asarray(first_sample["observed_data"])
        rec_pos = _extract_receiver_positions(first_sample)
        model = CToTraceFNO3D(
            grid_shape=tuple(c.shape),
            n_timesteps=n_t,
            n_receivers=int(obs.shape[2]),
            hidden_channels=8,
            num_modes=4,
            depth=1,
            receiver_positions=rec_pos,
            key=jr.PRNGKey(0),
        )
        first_two = list(reader.sample_ids)[:2]
        access_count = {"n": 0}

        class _Slice:
            sample_ids = first_two
            def __getitem__(self, sid):
                access_count["n"] += 1
                return reader[sid]
            def __iter__(self):
                return (reader[sid] for sid in first_two)

        return model, _Slice(), access_count

    def test_accumulation_consumes_n_samples_per_step(self, reader, first_sample, n_timesteps_distribution):
        """With accumulation_steps=4 and n_steps=2, the per-step inner
        loop should read 4*2=8 samples — proves accumulation is doing
        real work and not collapsing to a single grad call."""
        from brain_fwi.surrogate.train import (
            train_fno_surrogate, _extract_source_positions,
        )

        n_t = min(n_timesteps_distribution)
        model, slice_reader, access = self._tiny_model_and_reader(
            reader, first_sample, n_t,
        )
        # Pass source_positions explicitly so we don't pollute the
        # access counter with the trainer's setup-time read of sample[0].
        src_pos = _extract_source_positions(first_sample)
        access["n"] = 0  # reset after the setup read above

        trained, losses = train_fno_surrogate(
            model, slice_reader,
            n_steps=2,
            key=jr.PRNGKey(0),
            learning_rate=1e-3,
            accumulation_steps=4,
            source_positions=src_pos,
            log_every=99, verbose=False,
        )
        assert access["n"] == 8, (
            f"accumulation_steps=4 × n_steps=2 should read 8 samples; "
            f"got {access['n']}"
        )
        assert len(losses) == 2, (
            f"loss history should have one entry per logical step (n_steps=2), "
            f"got {len(losses)}"
        )

    def test_accumulation_default_is_one(self, reader, first_sample, n_timesteps_distribution):
        """Backward compat: the default accumulation_steps should be 1
        so existing call sites behave the same as before."""
        from brain_fwi.surrogate.train import (
            train_fno_surrogate, _extract_source_positions,
        )

        n_t = min(n_timesteps_distribution)
        model, slice_reader, access = self._tiny_model_and_reader(
            reader, first_sample, n_t,
        )
        src_pos = _extract_source_positions(first_sample)
        access["n"] = 0

        trained, losses = train_fno_surrogate(
            model, slice_reader,
            n_steps=2,
            key=jr.PRNGKey(0),
            learning_rate=1e-3,
            source_positions=src_pos,
            log_every=99, verbose=False,
        )
        assert access["n"] == 2, (
            f"default accumulation_steps=1 means n_steps=2 reads 2 samples; "
            f"got {access['n']} — accumulation default is leaking"
        )


class TestTrainerCosineSchedule:
    """A cosine schedule must actually reduce the LR over the training
    horizon — otherwise high-LR runs keep diverging."""

    def test_cosine_schedule_lowers_lr_over_steps(self):
        """Build the schedule the trainer would build and confirm the
        effective LR at the end is near alpha * init."""
        import optax
        init = 1e-3
        n_steps = 100
        alpha = 0.01
        sched = optax.cosine_decay_schedule(
            init_value=init, decay_steps=n_steps, alpha=alpha,
        )
        lr_first = float(sched(0))
        lr_last = float(sched(n_steps - 1))
        assert abs(lr_first - init) < 1e-9
        assert lr_last < init * 0.05, (
            f"cosine schedule didn't decay: lr@0={lr_first:.6f}, "
            f"lr@{n_steps-1}={lr_last:.6f}; expected ~ {init*alpha:.6f}"
        )

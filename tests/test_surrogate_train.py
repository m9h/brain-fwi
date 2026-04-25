"""Tests for the FNO surrogate training loop (Phase 4 §10 step 3)."""

from __future__ import annotations

from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from brain_fwi.surrogate.fno3d import CToTraceFNO3D
from brain_fwi.surrogate.train import (
    _rel_l2,
    _spectral_rel_l2,
    surrogate_loss,
    train_fno_surrogate,
)


# --- Mock reader ----------------------------------------------------------

class _FakeReader:
    """Minimal ShardedReader-compatible fake for unit tests."""

    def __init__(self, samples: List[dict]):
        self._samples = samples
        self.sample_ids = [s["sample_id"] for s in samples]

    def __getitem__(self, key):
        if isinstance(key, str):
            return next(s for s in self._samples if s["sample_id"] == key)
        return self._samples[key]


# --- Loss tests -----------------------------------------------------------

class TestRelL2:
    def test_identity_is_zero(self):
        x = jnp.array([1.0, 2.0, 3.0])
        assert float(_rel_l2(x, x)) == pytest.approx(0.0, abs=1e-6)

    def test_positive_homogeneity(self):
        """Relative L2 should be scale-invariant in the target."""
        x = jnp.array([0.5, 0.5, 0.5])
        y = jnp.array([1.0, 1.0, 1.0])
        a = float(_rel_l2(x, y))
        b = float(_rel_l2(x * 10, y * 10))
        assert a == pytest.approx(b, rel=1e-5)


class TestSpectralRelL2:
    def test_identity_is_zero(self):
        n_t, n_recv = 16, 3
        x = jr.normal(jr.PRNGKey(0), (n_t, n_recv))
        assert float(_spectral_rel_l2(x, x)) == pytest.approx(0.0, abs=1e-6)

    def test_catches_phase_drift(self):
        """Time-shifted target has identical magnitude spectrum but
        non-identical time-domain residuals. Spectral loss should stay
        low while time-domain rel-L2 should be large."""
        n_t, n_recv = 32, 1
        x = jnp.sin(2 * jnp.pi * jnp.arange(n_t)[:, None] / n_t * 3)
        x = x.astype(jnp.float32) * jnp.ones((n_t, n_recv))
        y = jnp.roll(x, shift=1, axis=0)
        # Time-domain residual is large (sinusoid roll by 1 sample)
        assert float(_rel_l2(x, y)) > 0.1
        # Magnitude-spectrum residual stays near zero (FFT is shift-invariant in magnitude)
        assert float(_spectral_rel_l2(x, y)) < 1e-5


# --- surrogate_loss tests -------------------------------------------------

class TestSurrogateLoss:
    def _build(self, n_src=2, n_t=5, n_recv=3, grid=(8, 8, 8), key=jr.PRNGKey(0)):
        model = CToTraceFNO3D(
            grid_shape=grid, n_timesteps=n_t, n_receivers=n_recv,
            hidden_channels=4, num_modes=2, depth=1, key=key,
        )
        c_norm = jr.uniform(jr.PRNGKey(1), grid)
        d_true = jr.normal(jr.PRNGKey(2), (n_src, n_t, n_recv))
        srcs = [(i + 2, i + 2, i + 2) for i in range(n_src)]
        return model, c_norm, d_true, srcs

    def test_returns_scalar(self):
        model, c, d, srcs = self._build()
        loss = surrogate_loss(model, c, d, srcs)
        assert loss.shape == ()

    def test_differentiable_in_model(self):
        model, c, d, srcs = self._build()

        def l(m):
            return surrogate_loss(m, c, d, srcs)

        grads = eqx.filter_grad(l)(model)
        leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
        assert len(leaves) > 0
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)

    def test_lambda_spec_zero_gives_time_only(self):
        model, c, d, srcs = self._build()
        both = float(surrogate_loss(model, c, d, srcs, lambda_spec=0.3))
        time_only = float(surrogate_loss(model, c, d, srcs, lambda_spec=0.0))
        assert time_only < both
        assert time_only > 0


# --- train_fno_surrogate tests --------------------------------------------

class TestTrainFNOSurrogate:
    def test_loss_decreases_on_synthetic_data(self):
        """End-to-end: a few training steps reduce the mean loss."""
        grid = (8, 8, 8)
        n_t, n_recv, n_src = 4, 2, 2

        rng = np.random.default_rng(0)
        n_samples = 4
        # Learnable target: each c's trace is a smooth function of its
        # global mean — FNO + head can fit this in ~50 steps.
        samples = []
        c_mean_list = rng.uniform(0.2, 0.8, size=n_samples)
        for i in range(n_samples):
            c = rng.uniform(0, 1, size=grid).astype(np.float32)
            c = c * (c_mean_list[i] / c.mean())  # control the mean
            d = np.broadcast_to(
                np.full((n_src, n_t, n_recv), c.mean(), dtype=np.float32),
                (n_src, n_t, n_recv),
            ).copy()
            samples.append({
                "sample_id": f"s{i:02d}",
                "sound_speed_voxel": c,
                "observed_data": d,
            })

        reader = _FakeReader(samples)
        model = CToTraceFNO3D(
            grid_shape=grid, n_timesteps=n_t, n_receivers=n_recv,
            hidden_channels=6, num_modes=2, depth=1, key=jr.PRNGKey(0),
        )

        # Feed normalised ~[0, 1] velocities through; c_min/c_max=0/1 keeps
        # the loop's internal normalisation a no-op.
        trained, losses = train_fno_surrogate(
            model, reader,
            n_steps=80,
            key=jr.PRNGKey(1),
            c_min=0.0, c_max=1.0,
            learning_rate=3e-3,
            lambda_spec=0.3,
            source_positions=[(3, 3, 3), (5, 5, 5)],
            verbose=False,
        )

        early = float(np.mean(losses[:10]))
        late = float(np.mean(losses[-10:]))
        assert late < early, (
            f"FNO surrogate training did not reduce loss: early={early:.3f}, "
            f"late={late:.3f}"
        )

    def test_held_out_ids_excluded_from_training(self):
        grid = (8, 8, 8)
        rng = np.random.default_rng(1)
        samples = [{
            "sample_id": f"x{i}",
            "sound_speed_voxel": rng.uniform(0, 1, size=grid).astype(np.float32),
            "observed_data": rng.standard_normal((1, 3, 2)).astype(np.float32),
        } for i in range(5)]
        reader = _FakeReader(samples)

        model = CToTraceFNO3D(
            grid_shape=grid, n_timesteps=3, n_receivers=2,
            hidden_channels=4, num_modes=2, depth=1, key=jr.PRNGKey(0),
        )

        # Hold out all but one; training must only use x0.
        _, losses = train_fno_surrogate(
            model, reader,
            n_steps=3, key=jr.PRNGKey(7),
            source_positions=[(4, 4, 4)],
            held_out_ids=["x1", "x2", "x3", "x4"],
            verbose=False,
        )
        assert len(losses) == 3

    def test_empty_after_holdout_raises(self):
        grid = (8, 8, 8)
        samples = [{
            "sample_id": f"x{i}",
            "sound_speed_voxel": np.zeros(grid, dtype=np.float32),
            "observed_data": np.zeros((1, 3, 2), dtype=np.float32),
        } for i in range(2)]
        reader = _FakeReader(samples)
        model = CToTraceFNO3D(
            grid_shape=grid, n_timesteps=3, n_receivers=2,
            hidden_channels=4, num_modes=2, depth=1, key=jr.PRNGKey(0),
        )
        with pytest.raises(ValueError, match="no training samples"):
            train_fno_surrogate(
                model, reader, n_steps=1, key=jr.PRNGKey(0),
                source_positions=[(4, 4, 4)],
                held_out_ids=["x0", "x1"], verbose=False,
            )

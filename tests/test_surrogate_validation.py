"""Tests for Phase 4 §7.2 / §7.3 validation gates."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from brain_fwi.surrogate.fno3d import CToTraceFNO3D
from brain_fwi.surrogate.validation import (
    _per_trace_rel_l2,
    format_gate_report,
    gradient_accuracy,
    trace_fidelity,
)


class TestPerTraceRelL2:
    def test_identity_is_zero(self):
        x = jr.normal(jr.PRNGKey(0), (2, 10, 3))
        out = _per_trace_rel_l2(x, x)
        assert out.shape == (2, 3)
        assert jnp.all(out < 1e-5)

    def test_reduces_time_axis_only(self):
        """rel-L2 should be per (shot, receiver), not aggregate."""
        pred = jnp.ones((2, 10, 3))
        target = jnp.ones((2, 10, 3)) * 2
        out = _per_trace_rel_l2(pred, target)
        assert out.shape == (2, 3)
        expected = 0.5  # ||2-1|| / ||2|| per-trace = sqrt(10)/sqrt(40) = 0.5
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5)


class TestTraceFidelity:
    def _model(self, grid=(8, 8, 8), n_t=5, n_recv=2):
        return CToTraceFNO3D(
            grid_shape=grid, n_timesteps=n_t, n_receivers=n_recv,
            hidden_channels=4, num_modes=2, depth=1, key=jr.PRNGKey(0),
        )

    def test_metric_keys_and_shapes(self):
        model = self._model()
        samples = [{
            "sound_speed_voxel": jr.uniform(jr.PRNGKey(i), (8, 8, 8), minval=1500, maxval=2800),
            "observed_data": jr.normal(jr.PRNGKey(i + 100), (2, 5, 2)),
        } for i in range(3)]
        result = trace_fidelity(
            model, samples,
            source_positions=[(2, 2, 2), (5, 5, 5)],
        )
        for key in ("median_rel_l2", "p95_rel_l2", "mean_rel_l2",
                     "spectral_ratio", "gate_pass_7_2", "n_samples",
                     "n_traces"):
            assert key in result
        assert result["n_samples"] == 3
        assert result["n_traces"] == 3 * 2 * 2   # samples × shots × receivers
        assert isinstance(result["gate_pass_7_2"], bool)

    def test_perfect_predictor_passes_gate(self):
        """A model whose traces equal the target should pass the gate."""
        grid = (8, 8, 8)
        n_src, n_t, n_recv = 2, 5, 2
        # Fabricate samples where the 'ground truth' d_true has the same
        # structure as what our toy model would emit at the same input.
        # That's not what the gate really checks, but the test proves the
        # plumbing correctly reports near-zero error on a self-consistent
        # pair.
        model = self._model(grid, n_t, n_recv)
        srcs = [(2, 2, 2), (5, 5, 5)]
        c = jr.uniform(jr.PRNGKey(0), grid, minval=1500, maxval=2800)
        # Run the model forward, then mark that as ground truth.
        c_norm = (c - 1400.0) / (3200.0 - 1400.0)
        d_true = jnp.stack([model(c_norm, s) for s in srcs], axis=0)

        samples = [{
            "sound_speed_voxel": c,
            "observed_data": d_true,
        }]
        result = trace_fidelity(model, samples, source_positions=srcs)
        assert result["median_rel_l2"] < 1e-5
        assert result["p95_rel_l2"] < 1e-5
        assert result["gate_pass_7_2"]


class TestGradientAccuracy:
    def test_identical_forward_gives_perfect_cosine(self):
        """If surrogate ≡ ground truth, gradient cosine similarity must be 1."""
        def fwd(c):
            return jnp.sum(c ** 3)  # some nonlinear scalar -> scalar

        c_samples = [jr.normal(jr.PRNGKey(i), (3, 3, 3)) for i in range(3)]
        result = gradient_accuracy(
            surrogate_forward=fwd,
            jwave_forward=fwd,
            c_samples=c_samples,
        )
        assert result["mean_cosine"] == pytest.approx(1.0, abs=1e-5)
        assert result["gate_pass_7_3"]

    def test_scaled_forward_still_unit_cosine(self):
        """Scaling the forward by a positive constant preserves gradient direction."""
        def surrogate(c):
            return 3.0 * jnp.sum(c ** 2)

        def jwave(c):
            return jnp.sum(c ** 2)

        c_samples = [jr.normal(jr.PRNGKey(i), (3, 3, 3)) for i in range(2)]
        result = gradient_accuracy(surrogate, jwave, c_samples)
        assert result["mean_cosine"] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_forward_fails_gate(self):
        def surrogate(c):
            # gradient of c[0]^2 is 2c[0] along axis 0, zero elsewhere
            return c.ravel()[0] ** 2

        def jwave(c):
            return c.ravel()[-1] ** 2

        c_samples = [jr.normal(jr.PRNGKey(i), (2, 2, 2)) for i in range(2)]
        result = gradient_accuracy(surrogate, jwave, c_samples)
        assert result["mean_cosine"] == pytest.approx(0.0, abs=1e-5)
        assert not result["gate_pass_7_3"]


class TestFormatGateReport:
    def test_includes_trace_gate_section(self):
        fake_trace = {
            "n_samples": 10, "n_traces": 100,
            "median_rel_l2": 0.005, "p95_rel_l2": 0.02, "mean_rel_l2": 0.008,
            "spectral_ratio": 1.02, "gate_pass_7_2": True,
        }
        report = format_gate_report(fake_trace)
        assert "§7.2" in report
        assert "PASS" in report

    def test_fail_marker_on_bad_metrics(self):
        fake_trace = {
            "n_samples": 10, "n_traces": 100,
            "median_rel_l2": 0.05, "p95_rel_l2": 0.20, "mean_rel_l2": 0.10,
            "spectral_ratio": 1.5, "gate_pass_7_2": False,
        }
        report = format_gate_report(fake_trace)
        assert "FAIL" in report

    def test_gradient_section_optional(self):
        fake_trace = {
            "n_samples": 10, "n_traces": 100,
            "median_rel_l2": 0.005, "p95_rel_l2": 0.02, "mean_rel_l2": 0.008,
            "spectral_ratio": 1.02, "gate_pass_7_2": True,
        }
        report_no_grad = format_gate_report(fake_trace)
        assert "§7.3" not in report_no_grad

        fake_grad = {
            "n_samples": 5, "per_sample_cosine": [0.96, 0.97, 0.95, 0.98, 0.94],
            "mean_cosine": 0.96, "median_cosine": 0.96, "min_cosine": 0.94,
            "gate_pass_7_3": True,
        }
        report_grad = format_gate_report(fake_trace, fake_grad)
        assert "§7.3" in report_grad

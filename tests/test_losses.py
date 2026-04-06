"""Tests for FWI loss functions (RED phase)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.inversion.losses import l2_loss, envelope_loss, multiscale_loss


class TestL2Loss:
    """Validate L2 waveform misfit."""

    def test_zero_for_identical_signals(self):
        x = jnp.sin(jnp.linspace(0, 10, 200))
        loss = l2_loss(x, x)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_different_signals(self):
        x = jnp.sin(jnp.linspace(0, 10, 200))
        y = jnp.cos(jnp.linspace(0, 10, 200))
        loss = l2_loss(x, y)
        assert float(loss) > 0

    def test_symmetric(self):
        """Unnormalized L2 should be symmetric."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.1, 2.2, 3.3])
        assert float(l2_loss(x, y)) == pytest.approx(float(l2_loss(y, x)), rel=1e-5)

    def test_2d_input(self):
        """Should work with (n_timesteps, n_sensors) arrays."""
        x = jnp.ones((100, 32))
        y = jnp.ones((100, 32)) * 1.1
        loss = l2_loss(x, y)
        assert float(loss) > 0

    def test_differentiable(self):
        """Loss should be differentiable w.r.t. predicted."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.1, 2.1, 3.1])
        grad = jax.grad(lambda p: l2_loss(p, y))(x)
        assert grad.shape == x.shape
        assert not jnp.any(jnp.isnan(grad))


class TestEnvelopeLoss:
    """Validate Hilbert envelope loss."""

    def test_zero_for_identical(self):
        x = jnp.sin(jnp.linspace(0, 10, 256))
        loss = envelope_loss(x[:, None], x[:, None])
        assert float(loss) < 1e-4

    def test_positive_for_different(self):
        t = jnp.linspace(0, 10, 256)
        x = jnp.sin(t)[:, None]
        y = (0.5 * jnp.sin(t))[:, None]  # different amplitude
        loss = envelope_loss(x, y)
        assert float(loss) > 0

    def test_invariant_to_phase_shift(self):
        """Envelope loss should be robust to phase shifts (unlike L2)."""
        t = jnp.linspace(0, 10, 256)
        x = jnp.sin(20 * t)[:, None]
        y = jnp.sin(20 * t + 0.5)[:, None]  # phase-shifted
        envelope_l = float(envelope_loss(x, y))
        l2_l = float(l2_loss(x, y))
        # Envelope loss should be much smaller than L2 for pure phase shift
        assert envelope_l < l2_l

    def test_differentiable(self):
        x = jnp.sin(jnp.linspace(0, 10, 128))[:, None]
        y = jnp.cos(jnp.linspace(0, 10, 128))[:, None]
        grad = jax.grad(lambda p: envelope_loss(p, y))(x)
        assert not jnp.any(jnp.isnan(grad))


class TestMultiscaleLoss:
    """Validate combined loss function."""

    def test_pure_l2_when_weight_zero(self):
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        y = jnp.array([1.1, 2.1, 3.1, 4.1])
        ms = multiscale_loss(x, y, envelope_weight=0.0)
        l2 = l2_loss(x, y)
        assert float(ms) == pytest.approx(float(l2), rel=1e-4)

    def test_between_l2_and_envelope(self):
        """Multiscale with 0 < weight < 1 should be between L2 and envelope."""
        t = jnp.linspace(0, 10, 128)
        x = jnp.sin(20 * t)[:, None]
        y = jnp.sin(20 * t + 1.0)[:, None]
        l2 = float(l2_loss(x, y))
        env = float(envelope_loss(x, y))
        ms = float(multiscale_loss(x, y, envelope_weight=0.5))
        assert min(l2, env) <= ms <= max(l2, env) + 0.01

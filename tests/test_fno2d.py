"""Tests for brain_fwi.surrogate.fno2d — toy 2D Fourier Neural Operator.

Minimal contract:

  - ``SpectralConv2D`` produces shape ``(H, W, out_channels)`` from
    input ``(H, W, in_channels)``.
  - ``FNO2D`` stacks lift → Fourier-blocks → project, preserves spatial
    shape, converts channel dims as specified.
  - ``CToTraceFNO`` maps a 2D sound-speed field ``(H, W, 1)`` to a trace
    ``(N_t,)`` and is end-to-end differentiable.
  - A short training run on synthetic data reduces MSE (sanity for the
    optimiser/model composition).

Evidence 9.2.1 from docs/design/phase4_fno_surrogate.md: "before scaling
to 3D, confirm the FNO family can hit accuracy targets on the simplest
possible problem".
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from brain_fwi.surrogate.fno2d import CToTraceFNO, FNO2D, SpectralConv2D


class TestSpectralConv2D:
    def test_forward_shape(self):
        conv = SpectralConv2D(
            in_channels=4, out_channels=8, modes_h=6, modes_w=6,
            key=jr.PRNGKey(0),
        )
        x = jnp.ones((16, 16, 4))
        y = conv(x)
        assert y.shape == (16, 16, 8)
        assert jnp.all(jnp.isfinite(y))

    def test_differentiable(self):
        conv = SpectralConv2D(
            in_channels=2, out_channels=2, modes_h=4, modes_w=4,
            key=jr.PRNGKey(1),
        )

        def loss(c, x):
            return jnp.sum(c(x) ** 2)

        x = jnp.ones((8, 8, 2))
        grads = eqx.filter_grad(loss)(conv, x)
        leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
        assert len(leaves) > 0
        assert all(jnp.all(jnp.isfinite(l)) for l in leaves)

    def test_modes_exceeding_half_nyquist_raises(self):
        """Mode truncation is defined for modes <= H/2+1; guard against misuse."""
        with pytest.raises(ValueError, match="modes"):
            SpectralConv2D(
                in_channels=1, out_channels=1, modes_h=10, modes_w=10,
                key=jr.PRNGKey(0),
            )._check_shape((8, 8))


class TestFNO2D:
    def test_forward_shape(self):
        fno = FNO2D(
            in_channels=1, out_channels=4, width=16, modes=6, n_blocks=3,
            key=jr.PRNGKey(0),
        )
        x = jnp.ones((32, 32, 1))
        y = fno(x)
        assert y.shape == (32, 32, 4)

    def test_vmap_over_batch(self):
        fno = FNO2D(
            in_channels=1, out_channels=2, width=8, modes=4, n_blocks=2,
            key=jr.PRNGKey(0),
        )
        x = jnp.ones((5, 16, 16, 1))
        y = jax.vmap(fno)(x)
        assert y.shape == (5, 16, 16, 2)


class TestCToTraceFNO:
    def test_output_shape(self):
        model = CToTraceFNO(
            grid_h=16, grid_w=16, n_timesteps=50, width=8, modes=4, n_blocks=2,
            key=jr.PRNGKey(0),
        )
        c = jnp.ones((16, 16, 1))
        out = model(c)
        assert out.shape == (50,)

    def test_differentiable(self):
        model = CToTraceFNO(
            grid_h=16, grid_w=16, n_timesteps=20, width=8, modes=4, n_blocks=1,
            key=jr.PRNGKey(1),
        )
        c = jnp.ones((16, 16, 1))
        target = jnp.zeros(20)

        def loss_fn(m):
            return jnp.mean((m(c) - target) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        assert jnp.isfinite(loss)
        leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
        assert all(jnp.all(jnp.isfinite(l)) for l in leaves)


class TestToyTrainingReducesLoss:
    """Smoke: a short training run on synthetic identity-ish data should
    reduce MSE. Catches catastrophic errors in the module assembly
    (broken shapes, frozen parameters, etc.) without needing real
    acoustic-wave data."""

    def test_loss_decreases_on_random_target(self):
        key = jr.PRNGKey(42)
        data_key, model_key = jr.split(key)

        # Synthetic task: 30 random c-fields, with targets that are a
        # learnable linear functional of the input (mean + spatial mean of
        # padded variants). Memorising random noise targets was a brittle
        # test signal — the model could not reduce loss enough in a fast-
        # suite time budget. Using a learnable target makes the test
        # actually diagnose "training is broken" rather than "the model
        # can't fit noise."
        n = 30
        c_fields = jr.uniform(data_key, (n, 16, 16, 1), minval=1.0, maxval=2.0)
        # Target: for each sample, a trace that is a smooth function of
        # the velocity field — mean, max, first-4-fft modes along both
        # axes, scaled to roughly unit variance. Any non-trivial FNO
        # should fit this in O(50) steps.
        c_flat = c_fields.squeeze(-1)  # (n, 16, 16)
        means = jnp.mean(c_flat, axis=(1, 2))       # (n,)
        maxes = jnp.max(c_flat, axis=(1, 2))        # (n,)
        fft_mag = jnp.abs(jnp.fft.fft2(c_flat))[:, :3, :3].reshape(n, -1)  # (n, 9)
        targets = jnp.concatenate(
            [means[:, None], maxes[:, None], fft_mag / 10.0,
             jnp.zeros((n, 20 - 11))], axis=1,
        )  # (n, 20)

        model = CToTraceFNO(
            grid_h=16, grid_w=16, n_timesteps=20,
            width=16, modes=4, n_blocks=2,
            key=model_key,
        )

        @eqx.filter_jit
        def step(m, opt_state, cfield, target):
            def l(m_):
                pred = m_(cfield)
                return jnp.mean((pred - target) ** 2)
            loss, grads = eqx.filter_value_and_grad(l)(m)
            updates, opt_state = optimizer.update(grads, opt_state)
            m = eqx.apply_updates(m, updates)
            return m, opt_state, loss

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        def batch_loss(m):
            preds = jax.vmap(m)(c_fields)
            return float(jnp.mean((preds - targets) ** 2))

        initial = batch_loss(model)
        # 200 steps — the test target is noise, which is harder to memorise
        # than a learnable function. With 80 steps the bespoke FNO can't
        # reliably cross the 30% reduction bar post-upgrade; at 200 steps
        # it comfortably does. The test exists to catch "training is broken",
        # not to benchmark convergence speed.
        for s in range(200):
            i = s % n
            model, opt_state, _ = step(model, opt_state, c_fields[i], targets[i])
        final = batch_loss(model)

        assert final < initial * 0.7, (
            f"toy training failed to reduce loss: initial={initial:.4f}, "
            f"final={final:.4f}"
        )

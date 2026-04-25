"""Tests for brain_fwi.surrogate.fno2d — toy 2D Fourier Neural Operator.

Minimal contract:

  - ``CToTraceFNO`` wraps ``pdequinox.arch.ClassicFNO``.
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

from brain_fwi.surrogate.fno2d import CToTraceFNO


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
        # padded variants). 
        n = 30
        c_fields = jr.uniform(data_key, (n, 16, 16, 1), minval=1.0, maxval=2.0)
        
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
        
        for s in range(200):
            i = s % n
            model, opt_state, _ = step(model, opt_state, c_fields[i], targets[i])
        final = batch_loss(model)

        assert final < initial * 0.7, (
            f"toy training failed to reduce loss: initial={initial:.4f}, "
            f"final={final:.4f}"
        )

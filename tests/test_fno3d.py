"""Tests for the 3D FNO surrogate (Phase 4 §10 step 2)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from brain_fwi.surrogate.fno3d import CToTraceFNO3D, _source_spike


class TestSourceSpike:
    def test_shape(self):
        spike = _source_spike(grid_shape=(8, 8, 8), src_pos_grid=(4, 4, 4))
        assert spike.shape == (8, 8, 8)

    def test_peak_at_source(self):
        spike = _source_spike(grid_shape=(16, 16, 16), src_pos_grid=(10, 3, 7))
        peak = np.unravel_index(int(jnp.argmax(spike)), spike.shape)
        assert peak == (10, 3, 7)

    def test_decays_away_from_source(self):
        spike = _source_spike(grid_shape=(8, 8, 8), src_pos_grid=(4, 4, 4),
                               sigma_voxels=1.0)
        assert float(spike[4, 4, 4]) == pytest.approx(1.0, rel=1e-5)
        assert float(spike[0, 0, 0]) < 1e-10


class TestCToTraceFNO3DShape:
    def test_forward_shape_small(self):
        model = CToTraceFNO3D(
            grid_shape=(8, 8, 8), n_timesteps=5, n_receivers=3,
            hidden_channels=4, num_modes=2, num_blocks=1,
            key=jr.PRNGKey(0),
        )
        c = jr.uniform(jr.PRNGKey(1), (8, 8, 8))
        out = model(c, src_pos_grid=(4, 4, 4))
        assert out.shape == (5, 3)
        assert jnp.all(jnp.isfinite(out))

    def test_different_sources_give_different_outputs(self):
        """Conditioning on source-position spike should actually affect output."""
        model = CToTraceFNO3D(
            grid_shape=(8, 8, 8), n_timesteps=5, n_receivers=3,
            hidden_channels=4, num_modes=2, num_blocks=1,
            key=jr.PRNGKey(0),
        )
        c = jr.uniform(jr.PRNGKey(1), (8, 8, 8))
        out_a = model(c, src_pos_grid=(2, 2, 2))
        out_b = model(c, src_pos_grid=(6, 6, 6))
        assert not jnp.allclose(out_a, out_b)


class TestCToTraceFNO3DGradient:
    def test_grad_flows_through_model_and_input(self):
        model = CToTraceFNO3D(
            grid_shape=(8, 8, 8), n_timesteps=4, n_receivers=2,
            hidden_channels=4, num_modes=2, num_blocks=1,
            key=jr.PRNGKey(0),
        )
        c = jr.uniform(jr.PRNGKey(1), (8, 8, 8))

        def loss(m, cf):
            return jnp.sum(m(cf, src_pos_grid=(4, 4, 4)) ** 2)

        g_model = eqx.filter_grad(loss)(model, c)
        leaves = jax.tree.leaves(eqx.filter(g_model, eqx.is_inexact_array))
        assert len(leaves) > 0
        assert all(jnp.all(jnp.isfinite(l)) for l in leaves)

        g_c = jax.grad(lambda cf: loss(model, cf))(c)
        assert g_c.shape == c.shape
        assert jnp.all(jnp.isfinite(g_c))
        assert float(jnp.max(jnp.abs(g_c))) > 0


class TestCToTraceFNO3DTrainingReducesLoss:
    def test_loss_drops_on_learnable_target(self):
        """MVP smoke: 3D FNO can fit a small learnable trace target."""
        key = jr.PRNGKey(7)
        data_key, model_key = jr.split(key)

        n = 10
        c_fields = jr.uniform(data_key, (n, 8, 8, 8), minval=0.0, maxval=1.0)

        means = jnp.mean(c_fields, axis=(1, 2, 3))
        vars_ = jnp.var(c_fields, axis=(1, 2, 3))
        targets = jnp.stack([means, vars_], axis=1)
        targets = jnp.repeat(targets[:, None, :], 3, axis=1)

        model = CToTraceFNO3D(
            grid_shape=(8, 8, 8), n_timesteps=3, n_receivers=2,
            hidden_channels=8, num_modes=3, num_blocks=2,
            key=model_key,
        )

        optimizer = optax.adam(3e-3)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        @eqx.filter_jit
        def step(m, opt_state, cfield, target):
            def l(m_):
                return jnp.mean((m_(cfield, src_pos_grid=(4, 4, 4)) - target) ** 2)
            loss, grads = eqx.filter_value_and_grad(l)(m)
            updates, opt_state = optimizer.update(grads, opt_state)
            m = eqx.apply_updates(m, updates)
            return m, opt_state, loss

        def batch_loss(m):
            preds = jax.vmap(
                lambda cf: m(cf, src_pos_grid=(4, 4, 4))
            )(c_fields)
            return float(jnp.mean((preds - targets) ** 2))

        initial = batch_loss(model)
        for s in range(120):
            i = s % n
            model, opt_state, _ = step(model, opt_state, c_fields[i], targets[i])
        final = batch_loss(model)

        assert final < initial * 0.7, (
            f"3D FNO training stalled: initial={initial:.4f}, final={final:.4f}"
        )

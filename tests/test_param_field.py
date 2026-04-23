"""Tests for ParameterField abstraction (voxel + SIREN).

Direct-velocity semantics: neither parameterisation uses sigmoid. Bounds
are enforced by clip in ``to_velocity`` (and again after each optimiser
step in the FWI loop).

Covers:
  - SIREN init / forward pass shape + finiteness
  - Pretrain regression quality (fits a known target)
  - VoxelField / SIRENField.to_velocity bounds and gradient flow
  - FWIConfig + _init_param_field wiring for both paths

The run_fwi integration smoke test is in tests/test_fwi.py (covered once
a GPU is available).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from brain_fwi.inversion.fwi import FWIConfig, _init_param_field
from brain_fwi.inversion.param_field import (
    SIREN,
    ParameterField,
    SIRENField,
    VoxelField,
    init_siren_from_velocity,
    init_voxel_from_velocity,
)


C_MIN, C_MAX = 1400.0, 3200.0


class TestSIRENForward:
    def test_scalar_output_shape(self):
        siren = SIREN(in_dim=2, hidden_dim=32, n_hidden=2, out_dim=1,
                      omega_0=30.0, key=jr.PRNGKey(0))
        x = jnp.array([0.1, -0.4])
        y = siren(x)
        assert y.shape == ()
        assert jnp.isfinite(y)

    def test_vmap_over_coords(self):
        siren = SIREN(in_dim=3, hidden_dim=32, n_hidden=2, out_dim=1,
                      omega_0=30.0, key=jr.PRNGKey(1))
        coords = jr.uniform(jr.PRNGKey(2), (100, 3), minval=-1.0, maxval=1.0)
        out = jax.vmap(siren)(coords)
        assert out.shape == (100,)
        assert jnp.all(jnp.isfinite(out))


class TestVoxelField:
    def test_to_velocity_shape_and_bounds(self):
        v = jnp.full((8, 8), 1800.0)
        field = init_voxel_from_velocity(v, C_MIN, C_MAX)
        v_back = field.to_velocity(C_MIN, C_MAX)
        assert v_back.shape == (8, 8)
        assert float(jnp.min(v_back)) >= C_MIN
        assert float(jnp.max(v_back)) <= C_MAX
        # Direct velocity: params are stored in m/s, round-trip is exact
        # except where clip engaged.
        np.testing.assert_allclose(np.array(v_back), 1800.0, atol=1e-3)

    def test_clip_engages_out_of_bounds(self):
        """Out-of-range params get clipped in to_velocity."""
        v = jnp.array([[1000.0, 2000.0], [3500.0, 1500.0]])
        field = VoxelField(params=v)
        v_back = field.to_velocity(C_MIN, C_MAX)
        np.testing.assert_array_equal(
            np.array(v_back),
            np.array([[C_MIN, 2000.0], [C_MAX, 1500.0]]),
        )

    def test_gradient_flows(self):
        """to_velocity is differentiable w.r.t. the voxel params."""
        v = jnp.full((4, 4), 1800.0)
        field = init_voxel_from_velocity(v, C_MIN, C_MAX)

        def loss(f):
            return jnp.sum(f.to_velocity(C_MIN, C_MAX))

        grads = eqx.filter_grad(loss)(field)
        assert grads.params.shape == (4, 4)
        assert jnp.all(jnp.isfinite(grads.params))
        # In the interior (no clip saturation), grad of sum is all-ones.
        np.testing.assert_allclose(np.array(grads.params), 1.0, atol=1e-5)


class TestSIRENField:
    def test_to_velocity_shape_and_bounds(self):
        siren = SIREN(in_dim=2, hidden_dim=32, n_hidden=2, out_dim=1,
                      omega_0=30.0, key=jr.PRNGKey(5))
        field = SIRENField(siren=siren, grid_shape=(12, 10))
        v = field.to_velocity(C_MIN, C_MAX)
        assert v.shape == (12, 10)
        assert float(jnp.min(v)) >= C_MIN
        assert float(jnp.max(v)) <= C_MAX

    def test_gradient_flows_to_weights(self):
        siren = SIREN(in_dim=2, hidden_dim=16, n_hidden=1, out_dim=1,
                      omega_0=30.0, key=jr.PRNGKey(7))
        field = SIRENField(siren=siren, grid_shape=(8, 8))

        def loss(f):
            return jnp.sum(f.to_velocity(C_MIN, C_MAX) ** 2)

        grads = eqx.filter_grad(loss)(field)
        flat = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
        assert len(flat) > 0
        assert all(jnp.all(jnp.isfinite(g)) for g in flat)


class TestSIRENPretrain:
    @pytest.mark.parametrize("ndim", [2, 3])
    def test_pretrain_reproduces_constant_field(self, ndim):
        shape = (16,) * ndim
        target = jnp.full(shape, 2000.0)
        field = init_siren_from_velocity(
            target, c_min=C_MIN, c_max=C_MAX,
            hidden_dim=32, n_hidden=2, omega_0=30.0,
            pretrain_steps=400, learning_rate=1e-3,
            key=jr.PRNGKey(11), verbose=False,
        )
        v = field.to_velocity(C_MIN, C_MAX)
        np.testing.assert_allclose(np.array(v), 2000.0, atol=50.0)

    def test_pretrain_reproduces_smooth_field(self):
        """SIREN should fit a low-frequency spatial pattern accurately."""
        n = 24
        xs = jnp.linspace(-1, 1, n)
        x, y = jnp.meshgrid(xs, xs, indexing="ij")
        target = 1500.0 + 500.0 * jnp.exp(-(x ** 2 + y ** 2) / 0.3)

        field = init_siren_from_velocity(
            target, c_min=C_MIN, c_max=C_MAX,
            hidden_dim=64, n_hidden=3, omega_0=30.0,
            pretrain_steps=800, learning_rate=1e-3,
            key=jr.PRNGKey(13), verbose=False,
        )
        v = field.to_velocity(C_MIN, C_MAX)
        rel_err = float(jnp.mean(jnp.abs(v - target)) / jnp.mean(target))
        assert rel_err < 0.05, f"SIREN fit too poor: {rel_err:.3%}"


class TestSIRENOnCoupledHeadPhantom:
    """Representation-fidelity gate: SIREN fits a USCT head phantom to
    p95 rel-err < 2% when the ambient air has been replaced with water
    coupling medium.

    With direct-velocity + clip, air voxels (c=343) still cannot be
    represented because clip enforces c >= c_min=1400. Water coupling
    is therefore still mandatory — same representation limit, different
    mechanism. See docs/design/data_pipeline.md §6 validation gate #2.
    """

    def _build_phantom(self, grid_shape=(16, 16, 16), dx=0.004):
        """Minimal 3D head phantom with jittered acoustic properties."""
        from brain_fwi.phantoms.augment import jittered_properties

        nx, ny, nz = grid_shape
        cx, cy, cz = nx // 2, ny // 2, nz // 2
        head_a = min(0.095 / dx, cx - 3)
        head_b = min(0.075 / dx, cy - 3)
        head_c = min(0.090 / dx, cz - 3)

        x, y, z = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij",
        )
        r = np.sqrt(
            ((x - cx) / head_a) ** 2
            + ((y - cy) / head_b) ** 2
            + ((z - cz) / head_c) ** 2
        )
        labels = np.zeros(grid_shape, dtype=np.int32)
        labels = np.where(r <= 1.0, 6, labels)       # scalp
        labels = np.where(r <= 0.93, 7, labels)      # skull
        labels = np.where(r <= 0.85, 1, labels)      # CSF
        labels = np.where(r <= 0.80, 2, labels)      # GM
        labels = np.where(r <= 0.55, 3, labels)      # WM

        props = jittered_properties(
            jnp.asarray(labels), jr.PRNGKey(0), intensity=1.0,
        )
        return jnp.asarray(labels), props["sound_speed"]

    def test_uncoupled_phantom_violates_gate(self):
        """Red-anchor: without water coupling the gate is NOT met.

        Air voxels clip to c_min=1400, but the target has c=343, so
        relative error in that region is ~3x. If this ever passes,
        either (a) c_min has been lowered for air support, or (b) air
        is being silently replaced upstream (investigate).
        """
        labels, c_with_air = self._build_phantom()
        field = init_siren_from_velocity(
            c_with_air, c_min=C_MIN, c_max=C_MAX,
            hidden_dim=128, n_hidden=3, omega_0=30.0,
            pretrain_steps=200, learning_rate=1e-3,
            key=jr.PRNGKey(1), verbose=False,
        )
        v = field.to_velocity(C_MIN, C_MAX)
        p95 = float(jnp.percentile(jnp.abs(v - c_with_air) / c_with_air, 95))
        assert p95 > 0.1, (
            f"uncoupled phantom unexpectedly fit with p95={p95:.3%}; "
            f"the clip-based representation limit on air may no longer hold"
        )

    def test_coupled_phantom_meets_validation_gate(self):
        """Green: water-coupled phantom fits to p95 rel-err < 2%."""
        labels, c_raw = self._build_phantom()
        c_coupled = jnp.where(labels == 0, 1500.0, c_raw)

        field = init_siren_from_velocity(
            c_coupled, c_min=C_MIN, c_max=C_MAX,
            hidden_dim=128, n_hidden=3, omega_0=30.0,
            pretrain_steps=200, learning_rate=1e-3,
            key=jr.PRNGKey(1), verbose=False,
        )
        v = field.to_velocity(C_MIN, C_MAX)
        p95 = float(jnp.percentile(jnp.abs(v - c_coupled) / c_coupled, 95))
        assert p95 < 0.02, f"SIREN fit missed validation gate: p95={p95:.3%}"


class TestInitParamField:
    def test_voxel_path(self):
        config = FWIConfig(parameterization="voxel")
        v = jnp.full((8, 8), 1800.0)
        f = _init_param_field(v, config)
        assert isinstance(f, VoxelField)

    def test_siren_path(self):
        config = FWIConfig(
            parameterization="siren",
            siren_hidden=16, siren_layers=1,
            siren_pretrain_steps=20, verbose=False,
        )
        v = jnp.full((8, 8), 1800.0)
        f = _init_param_field(v, config)
        assert isinstance(f, SIRENField)
        assert f.grid_shape == (8, 8)

    def test_unknown_parameterization_raises(self):
        config = FWIConfig(parameterization="voxel")
        config.parameterization = "bogus"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown parameterization"):
            _init_param_field(jnp.zeros((4, 4)), config)

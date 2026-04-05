"""Tests for forward acoustic simulation (RED phase).

These tests validate the j-Wave integration layer without requiring
GPU or long simulation times. Uses small grids and short durations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.simulation.forward import (
    build_domain,
    build_medium,
    build_time_axis,
)


class TestDomainConstruction:
    """Validate j-Wave domain construction."""

    def test_2d_domain(self):
        domain = build_domain((64, 64), dx=0.001)
        assert domain.N == (64, 64)
        assert domain.dx == (0.001, 0.001)

    def test_3d_domain(self):
        domain = build_domain((32, 32, 32), dx=0.0005)
        assert domain.N == (32, 32, 32)

    def test_domain_dx_tuple(self):
        domain = build_domain((128, 128), dx=0.5e-3)
        assert len(domain.dx) == 2
        assert domain.dx[0] == pytest.approx(0.5e-3)


class TestMediumConstruction:
    """Validate j-Wave medium construction."""

    def test_homogeneous_medium(self):
        domain = build_domain((64, 64), 0.001)
        medium = build_medium(domain, sound_speed=1500.0, density=1000.0)
        assert medium is not None

    def test_heterogeneous_medium(self):
        domain = build_domain((64, 64), 0.001)
        c = jnp.ones((64, 64)) * 1500.0
        rho = jnp.ones((64, 64)) * 1000.0
        medium = build_medium(domain, c, rho)
        assert medium is not None

    def test_pml_size(self):
        domain = build_domain((64, 64), 0.001)
        medium = build_medium(domain, 1500.0, 1000.0, pml_size=10)
        assert medium.pml_size == 10


class TestTimeAxis:
    """Validate time axis computation."""

    def test_time_axis_from_medium(self):
        domain = build_domain((64, 64), 0.001)
        medium = build_medium(domain, 1500.0, 1000.0)
        ta = build_time_axis(medium, cfl=0.3)
        assert float(ta.dt) > 0
        assert float(ta.t_end) > 0

    def test_cfl_affects_dt(self):
        """Smaller CFL should give smaller dt."""
        domain = build_domain((64, 64), 0.001)
        medium = build_medium(domain, 1500.0, 1000.0)
        ta_03 = build_time_axis(medium, cfl=0.3)
        ta_01 = build_time_axis(medium, cfl=0.1)
        assert float(ta_01.dt) < float(ta_03.dt)

    def test_custom_t_end(self):
        domain = build_domain((64, 64), 0.001)
        medium = build_medium(domain, 1500.0, 1000.0)
        ta = build_time_axis(medium, cfl=0.3, t_end=50e-6)
        assert float(ta.t_end) == pytest.approx(50e-6, rel=0.1)

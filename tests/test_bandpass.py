"""Tests for multi-frequency bandpass FWI (RED phase).

Validates that:
1. The bandpass filter preserves signal energy within the band
2. Bandpassed data + source produce consistent FWI gradients
3. Multi-frequency banding converges (low-to-high frequency progression)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.inversion.fwi import _bandpass_signal
from brain_fwi.utils.wavelets import ricker_wavelet


class TestBandpassFilter:
    """Validate the FFT bandpass filter."""

    def test_passband_energy_preserved(self):
        """Signal energy within the passband should be mostly preserved."""
        dt = 1e-7
        n = 1024
        f0 = 200e3
        signal = ricker_wavelet(f0, dt, n)

        # Bandpass around the peak frequency
        filtered = _bandpass_signal(signal, dt, 100e3, 300e3)

        # Energy ratio should be high (>50%) since Ricker peak is at f0
        energy_in = float(jnp.sum(signal ** 2))
        energy_out = float(jnp.sum(filtered ** 2))
        ratio = energy_out / energy_in
        assert ratio > 0.5, f"Too much energy lost: {ratio:.2%}"

    def test_stopband_rejection(self):
        """Frequencies outside the band should be attenuated."""
        dt = 1e-7
        n = 1024
        # Pure 500 kHz tone
        t = jnp.arange(n) * dt
        signal = jnp.sin(2 * jnp.pi * 500e3 * t)

        # Bandpass at 100-200 kHz — should reject 500 kHz
        filtered = _bandpass_signal(signal, dt, 100e3, 200e3)

        energy_in = float(jnp.sum(signal ** 2))
        energy_out = float(jnp.sum(filtered ** 2))
        ratio = energy_out / energy_in
        assert ratio < 0.1, f"Insufficient stopband rejection: {ratio:.2%}"

    def test_bandpass_preserves_length(self):
        signal = jnp.ones(256)
        filtered = _bandpass_signal(signal, 1e-7, 100e3, 500e3)
        assert filtered.shape == signal.shape

    def test_bandpass_smooth_edges(self):
        """Filter response should be smooth (no Gibbs ringing)."""
        dt = 1e-7
        n = 2048
        # Impulse
        signal = jnp.zeros(n).at[n // 2].set(1.0)

        filtered = _bandpass_signal(signal, dt, 100e3, 300e3)

        # Check that the filtered impulse response decays smoothly
        # (no large sidelobes relative to main lobe)
        peak = float(jnp.max(jnp.abs(filtered)))
        # Sidelobes should be < 50% of peak (generous for cosine taper)
        tail = jnp.abs(filtered[:n // 4])
        max_sidelobe = float(jnp.max(tail))
        if peak > 0:
            assert max_sidelobe / peak < 0.5

    def test_bandpass_differentiable(self):
        """Bandpass should be differentiable w.r.t. input signal."""
        signal = jnp.sin(jnp.linspace(0, 10, 256))
        grad = jax.grad(lambda s: jnp.sum(_bandpass_signal(s, 1e-7, 100e3, 500e3) ** 2))(signal)
        assert jnp.all(jnp.isfinite(grad))


class TestMultiFrequencyFWI:
    """Validate multi-frequency FWI convergence."""

    def test_bandpass_data_nonzero(self):
        """Bandpassed data should still contain signal."""
        from brain_fwi.simulation.forward import (
            build_domain, build_medium, build_time_axis,
            simulate_shot_sensors, _build_source_signal,
        )

        grid_shape = (32, 32)
        dx = 0.003
        domain = build_domain(grid_shape, dx)
        medium = build_medium(domain, 1500.0, 1000.0, pml_size=8)
        time_axis = build_time_axis(medium, cfl=0.3, t_end=30e-6)
        dt = float(time_axis.dt)
        n_samples = int(float(time_axis.t_end) / dt)

        freq = 50e3
        source_signal = _build_source_signal(freq, dt, n_samples)

        # Simulate a shot
        data = simulate_shot_sensors(
            medium, time_axis, (16, 16),
            (jnp.array([8, 24]), jnp.array([8, 24])),
            source_signal, dt,
        )

        # Bandpass the data
        bp_data = jax.vmap(
            lambda col: _bandpass_signal(col, dt, 20e3, 80e3)
        )(data.T).T

        assert float(jnp.max(jnp.abs(bp_data))) > 0, \
            "Bandpassed data is all zeros"

        # Energy ratio
        e_orig = float(jnp.sum(data ** 2))
        e_bp = float(jnp.sum(bp_data ** 2))
        if e_orig > 0:
            assert e_bp / e_orig > 0.1, \
                f"Bandpass removed too much energy: {e_bp/e_orig:.2%}"

    def test_bandpassed_loss_gradient_nonzero(self):
        """Gradient through bandpassed forward should be non-zero."""
        from brain_fwi.simulation.forward import (
            build_domain, build_medium, build_time_axis,
            simulate_shot_sensors, _build_source_signal,
        )
        from brain_fwi.inversion.fwi import _params_to_velocity, _velocity_to_params
        from brain_fwi.inversion.losses import l2_loss

        grid_shape = (32, 32)
        dx = 0.003
        c_min, c_max = 1400.0, 1800.0

        domain = build_domain(grid_shape, dx)
        ref_medium = build_medium(domain, c_max, 1000.0, pml_size=8)
        time_axis = build_time_axis(ref_medium, cfl=0.3, t_end=30e-6)
        dt = float(time_axis.dt)
        n_samples = int(float(time_axis.t_end) / dt)
        freq = 50e3
        source_signal = _build_source_signal(freq, dt, n_samples)

        # Bandpass source
        bp_signal = _bandpass_signal(source_signal, dt, 20e3, 80e3)

        sensor_pos = (jnp.array([8, 24]), jnp.array([8, 24]))

        # Generate observed with true model
        true_medium = build_medium(domain, 1500.0, 1000.0, pml_size=8)
        observed = simulate_shot_sensors(
            true_medium, time_axis, (16, 16), sensor_pos, bp_signal, dt,
        )

        # Gradient w.r.t. perturbed model
        params = _velocity_to_params(jnp.full(grid_shape, 1480.0), c_min, c_max)

        def loss_fn(p):
            vel = _params_to_velocity(p, c_min, c_max)
            med = build_medium(domain, vel, jnp.ones(grid_shape) * 1000.0, pml_size=8)
            pred = simulate_shot_sensors(med, time_axis, (16, 16), sensor_pos, bp_signal, dt)
            min_t = min(pred.shape[0], observed.shape[0])
            return l2_loss(pred[:min_t], observed[:min_t])

        loss_val, grad = jax.value_and_grad(loss_fn)(params)
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.max(jnp.abs(grad))) > 0, "Gradient is zero through bandpass"

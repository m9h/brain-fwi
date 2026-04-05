"""Tests for source wavelets (RED phase)."""

import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.utils.wavelets import ricker_wavelet, toneburst


class TestRickerWavelet:
    """Validate Ricker (Mexican hat) wavelet."""

    def test_output_length(self):
        w = ricker_wavelet(f0=200e3, dt=1e-7, n_samples=500)
        assert w.shape == (500,)

    def test_peak_near_delay(self):
        """Peak amplitude should occur near the specified delay."""
        dt = 1e-7
        f0 = 200e3
        delay = 1.5 / f0
        w = ricker_wavelet(f0, dt, n_samples=1000)
        peak_idx = int(jnp.argmax(jnp.abs(w)))
        peak_time = peak_idx * dt
        assert abs(peak_time - delay) < 3 * dt

    def test_zero_at_boundaries(self):
        """Ricker wavelet should decay to ~0 at the ends."""
        w = ricker_wavelet(f0=100e3, dt=1e-7, n_samples=2000)
        assert abs(float(w[-1])) < 0.01

    def test_symmetric_about_peak(self):
        """Ricker wavelet is symmetric about its peak."""
        w = ricker_wavelet(f0=200e3, dt=1e-7, n_samples=1000)
        peak = int(jnp.argmax(jnp.abs(w)))
        # Check symmetry in a window around peak
        radius = min(peak, len(w) - peak - 1, 50)
        left = w[peak - radius:peak]
        right = jnp.flip(w[peak + 1:peak + radius + 1])
        np.testing.assert_allclose(np.array(left), np.array(right), atol=0.05)

    def test_dominant_frequency(self):
        """FFT peak should be near the specified f0."""
        f0 = 200e3
        dt = 1e-7
        n = 4096
        w = ricker_wavelet(f0, dt, n)
        spectrum = jnp.abs(jnp.fft.rfft(w))
        freqs = jnp.fft.rfftfreq(n, d=dt)
        peak_freq = float(freqs[jnp.argmax(spectrum)])
        assert abs(peak_freq - f0) / f0 < 0.15  # within 15%


class TestToneburst:
    """Validate windowed toneburst."""

    def test_output_shape_auto_size(self):
        w = toneburst(f0=500e3, dt=1e-7, n_cycles=5)
        assert w.ndim == 1
        assert len(w) > 0

    def test_output_shape_fixed_size(self):
        w = toneburst(f0=500e3, dt=1e-7, n_cycles=5, n_samples=1000)
        assert w.shape == (1000,)

    def test_zero_before_delay(self):
        """Signal should be zero before the delay."""
        w = toneburst(f0=500e3, dt=1e-7, n_cycles=5, delay=10e-6)
        # First 50 samples (5 us) should be ~0
        early = w[:50]
        assert float(jnp.max(jnp.abs(early))) < 1e-6

    def test_energy_in_burst_window(self):
        """Most energy should be within the burst duration."""
        f0 = 500e3
        dt = 1e-7
        n_cycles = 5
        w = toneburst(f0, dt, n_cycles, n_samples=2000)
        duration_samples = int(n_cycles / f0 / dt) + 10
        energy_burst = float(jnp.sum(w[:duration_samples] ** 2))
        energy_total = float(jnp.sum(w ** 2))
        if energy_total > 0:
            assert energy_burst / energy_total > 0.95

    def test_hann_windowed(self):
        """Toneburst should start and end smoothly (no abrupt edges)."""
        w = toneburst(f0=500e3, dt=1e-7, n_cycles=10, n_samples=2000)
        # Find the active region
        active = jnp.abs(w) > 0.01 * float(jnp.max(jnp.abs(w)))
        if jnp.any(active):
            first_active = int(jnp.argmax(active))
            # First few samples of active region should be small (Hann window)
            assert abs(float(w[first_active])) < 0.3 * float(jnp.max(jnp.abs(w)))

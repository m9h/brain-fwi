"""Gradient smoothing specified in WAVELENGTHS, not grid points.

Why (scope_3d_gmwm_run.md, 2026-10-23): speckle in the GM/WM reconstructions is set by the
regularisation scale measured in wavelengths. A sigma fixed in grid points shrinks
physically when the grid is refined (ppw 3.2 -> 4.9 took speckle 5.0 -> 7.0 m/s) and is the
wrong size at every other band. `FWIConfig.gradient_smooth_wavelengths` makes the rule
(~lambda/2 at each band's top frequency) enforced rather than remembered.
"""
import pytest

from brain_fwi.inversion.fwi import FWIConfig, _smooth_sigma_for_band


def test_default_is_grid_point_sigma_unchanged():
    cfg = FWIConfig(freq_bands=[(150e3, 300e3)], gradient_smooth_sigma=1.0)
    assert cfg.gradient_smooth_wavelengths is None
    assert _smooth_sigma_for_band(cfg, fmax=300e3, dx=1.61e-3) == pytest.approx(1.0)


def test_wavelength_sigma_is_fraction_of_top_band_wavelength_in_grid_points():
    # lambda = 1540 / 300 kHz = 5.133 mm; half of it on a 1.61 mm grid = 1.594 grid points
    cfg = FWIConfig(freq_bands=[(150e3, 300e3)], gradient_smooth_wavelengths=0.5)
    assert _smooth_sigma_for_band(cfg, fmax=300e3, dx=1.61e-3) == pytest.approx(0.5 * (1540.0 / 300e3) / 1.61e-3, rel=1e-6)


def test_wavelength_sigma_is_invariant_to_grid_refinement_in_physical_units():
    cfg = FWIConfig(freq_bands=[(150e3, 300e3)], gradient_smooth_wavelengths=0.5)
    coarse = _smooth_sigma_for_band(cfg, fmax=300e3, dx=1.61e-3) * 1.61e-3
    fine = _smooth_sigma_for_band(cfg, fmax=300e3, dx=1.02e-3) * 1.02e-3
    assert coarse == pytest.approx(fine, rel=1e-6)          # same physical sigma
    assert _smooth_sigma_for_band(cfg, fmax=300e3, dx=1.02e-3) > _smooth_sigma_for_band(cfg, fmax=300e3, dx=1.61e-3)


def test_wavelength_sigma_tracks_the_band():
    cfg = FWIConfig(freq_bands=[(150e3, 225e3), (225e3, 300e3)], gradient_smooth_wavelengths=0.5)
    lo = _smooth_sigma_for_band(cfg, fmax=225e3, dx=1.61e-3)
    hi = _smooth_sigma_for_band(cfg, fmax=300e3, dx=1.61e-3)
    assert lo == pytest.approx(hi * 300e3 / 225e3, rel=1e-6)   # longer wavelength -> wider kernel


def test_wavelength_sigma_overrides_grid_point_sigma():
    cfg = FWIConfig(freq_bands=[(150e3, 300e3)], gradient_smooth_sigma=1.0, gradient_smooth_wavelengths=0.5)
    assert _smooth_sigma_for_band(cfg, fmax=300e3, dx=1.61e-3) != pytest.approx(1.0)

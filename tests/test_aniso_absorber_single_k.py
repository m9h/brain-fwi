"""Single-k periodic calibration of the anisotropic absorber (diffraction-free).

The plane-wave (finite-aperture) harness showed the correct SIGN but a contrast
compressed by beam diffraction. Here we remove diffraction entirely: a pure
single-wavevector standing wave ``p0 = cos(k.x)`` in a PERIODIC domain (pml=0).
With a single k there is no k-spread, so the measured decay must match the
analytic per-k rate ``alpha(k_hat) = (k_hat^T D k_hat) * (f/1e6)^y`` [dB/cm]
exactly (for k along vs across the fibre).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
@pytest.mark.parametrize("prop_axis,eig_name", [(0, "a_par"), (1, "a_perp")])
def test_single_k_alpha_matches_analytic(prop_axis, eig_name):
    from brain_fwi.simulation.aniso_absorber_validate import measure_single_k_alpha_db
    a_par, a_perp = 3.0, 9.0
    # fibre along x. k along x -> along-fibre (eigenvalue a_par); k along y -> across (a_perp).
    eig = a_par if prop_axis == 0 else a_perp
    meas, ana = measure_single_k_alpha_db(
        a_par=a_par, a_perp=a_perp, fibre_axis=0, prop_axis=prop_axis,
        N=32, dx=3e-4, y=1.1, m=2)
    assert ana == pytest.approx(eig * (0.3125 ** 1.1), rel=1e-3)   # sanity on analytic (f=312.5kHz)
    assert meas / ana == pytest.approx(1.0, abs=0.2), (
        f"{eig_name}: single-k alpha {meas:.3f} vs analytic {ana:.3f} dB/cm (ratio {meas/ana:.2f})")

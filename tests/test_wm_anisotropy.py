"""Red-green tests for the WM attenuation-anisotropy translation module.

The physics: white matter is a transversely-isotropic viscoelastic medium.
A compressional wave's attenuation depends on the angle theta between the
propagation direction and the fibre axis, interpolating between alpha_par
(along fibre) and alpha_perp (across). The MAGNITUDE of the anisotropy in
brain WM is unmeasured acoustically, so the module brackets it from real-tissue
evidence in adjacent modalities/tissues:
  * skeletal muscle -- the fibrous analog where longitudinal attenuation
    anisotropy IS directly measured (Nassiri 1979: 2.9 vs 1.1 dB/cm/MHz
    parallel vs perpendicular -> ratio ~2.6; Topp & O'Brien 2000 ~2x).
  * white-matter MRE -- anisotropic *shear* damping directly measured in real
    WM (Anderson 2018: 27-34% loss-modulus anisotropy).
"""
import numpy as np
import pytest

from brain_fwi.tissue.wm_anisotropy import (
    predicted_alpha_profile, muscle_reference, wm_ratio_bracket,
    ratio_from_fa,
)


def test_profile_endpoints_and_monotonic():
    a_par, a_perp = 1.1, 2.9  # across-fibre higher, as in muscle (dB/cm/MHz)
    assert predicted_alpha_profile(0.0, a_par, a_perp) == pytest.approx(a_par, rel=1e-6)
    assert predicted_alpha_profile(np.pi / 2, a_par, a_perp) == pytest.approx(a_perp, rel=1e-6)
    th = np.linspace(0, np.pi / 2, 20)
    prof = predicted_alpha_profile(th, a_par, a_perp)
    assert np.all(np.diff(prof) >= -1e-9), "profile must be monotonic between endpoints"


def test_muscle_reference_matches_measurement():
    """The calibration anchor: real muscle longitudinal attenuation anisotropy."""
    ref = muscle_reference()
    # Nassiri 1979: parallel 2.9, perpendicular 1.1 dB/cm/MHz -> ratio ~2.6
    assert ref["ratio"] == pytest.approx(2.6, abs=0.2)
    assert ref["alpha_par"] > ref["alpha_perp"]  # muscle: more attenuation along fibre


def test_wm_bracket_is_ordered_and_physical():
    lo, mid, hi = wm_ratio_bracket()
    assert 1.0 < lo < mid < hi
    # conservative >= MRE shear-anisotropy analog (~1.3), optimistic <= muscle (~2.6)
    assert lo >= 1.25
    assert hi <= 2.7


def test_ratio_scales_with_fa_and_grounds_at_isotropy():
    # zero anisotropy -> isotropic (ratio 1); FA drives it toward the tissue max
    assert ratio_from_fa(0.0, ratio_max=2.0) == pytest.approx(1.0, abs=1e-6)
    r_lo = ratio_from_fa(0.3, ratio_max=2.0)
    r_hi = ratio_from_fa(0.6, ratio_max=2.0)
    assert 1.0 < r_lo < r_hi <= 2.0

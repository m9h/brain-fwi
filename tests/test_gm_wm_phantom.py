"""GM/WM contrast phantom (issue #47).

The ITRUSST default table gives grey and white matter *identical* acoustic
properties, so a reconstruction cannot be scored for GM/WM. The contrasted
phantom builder opts into the Kang (2022) attenuation contrast: white matter
attenuates ~1.5x grey matter, while sound speed is (deliberately) unchanged.

These tests assert the physics: the ground-truth ATTENUATION field separates
GM from WM (separability >> 0) while the SOUND-SPEED field does not (~0).
"""

import numpy as np

from brain_fwi.phantoms.synthetic import make_gm_wm_contrast_head
from brain_fwi.robustness.metrics import gm_wm_separability


# Coarse grid + 2 mm spacing -> a physically full-size head (~9 cm) so the
# fixed-thickness anatomical layers leave room for GM ribbon + WM interior.
# No FWI here, so this stays fast.
GRID = (64, 64, 64)
DX = 2.0e-3


def _build():
    return make_gm_wm_contrast_head(GRID, DX)


def test_returns_expected_arrays():
    c, rho, alpha, labels, dx, gm_mask, wm_mask = _build()
    assert c.shape == GRID
    assert rho.shape == GRID
    assert alpha.shape == GRID
    assert labels.shape == GRID
    assert gm_mask.shape == GRID
    assert wm_mask.shape == GRID
    assert dx == DX
    # Both tissue regions are actually present and disjoint.
    assert gm_mask.sum() > 0
    assert wm_mask.sum() > 0
    assert not np.any(gm_mask & wm_mask)


def test_attenuation_separates_gm_wm():
    c, rho, alpha, labels, dx, gm_mask, wm_mask = _build()
    sep_alpha = gm_wm_separability(alpha, gm_mask, wm_mask)
    # WM alpha (0.9) vs GM alpha (0.6): the regions are homogeneous so pooled
    # variance is ~0 and separability is huge. Anything well above zero proves
    # the contrast exists.
    assert sep_alpha > 10.0
    # WM should attenuate MORE than GM.
    assert alpha[wm_mask].mean() > alpha[gm_mask].mean()
    # Sanity on the actual values.
    assert np.allclose(alpha[gm_mask], 0.6)
    assert np.allclose(alpha[wm_mask], 0.9)


def test_sound_speed_does_not_separate_gm_wm():
    c, rho, alpha, labels, dx, gm_mask, wm_mask = _build()
    sep_c = gm_wm_separability(c, gm_mask, wm_mask)
    # GM and WM share the same sound speed -> no contrast in c.
    assert sep_c < 1e-6
    assert np.allclose(c[gm_mask], c[wm_mask].mean())

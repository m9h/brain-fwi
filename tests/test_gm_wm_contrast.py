"""Phase 6.4 (issue #47) — per-tissue GM/WM contrast phantoms + separability.

The default ITRUSST table treats grey and white matter as identical (both
1560 m/s, 0.6 dB/cm/MHz) — faithful to the benchmark but *zero contrast*, so a
reconstruction can't be scored for GM/WM. Real tissue separates mainly in
ATTENUATION (Kang et al., Ultrasonics 2022: WM ~1.5x GM; sound speed at the
noise floor). This spec adds an opt-in contrasted table + a separability metric.

One behaviour per test, each written failing first.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest


def test_default_table_is_zero_contrast_and_untouched():
    """Regression: the canonical table keeps GM==WM (ITRUSST-faithful) so every
    existing phantom is unaffected by #47."""
    from brain_fwi.phantoms.properties import TISSUE_PROPERTIES
    gm, wm = TISSUE_PROPERTIES[2], TISSUE_PROPERTIES[3]
    assert gm == wm
    assert gm[2] == pytest.approx(0.6)  # alpha


def test_contrasted_table_gives_wm_15x_gm_attenuation():
    """Opt-in contrast injects the measured WM ~1.5x GM attenuation; GM
    unchanged; sound speed left equal by default (barely separable)."""
    from brain_fwi.phantoms.properties import (
        TISSUE_PROPERTIES, tissue_properties_contrasted)
    props = tissue_properties_contrasted()
    gm, wm = props[2], props[3]
    assert gm == TISSUE_PROPERTIES[2], "GM must be unchanged"
    assert wm[2] == pytest.approx(1.5 * gm[2])   # alpha: 0.9 vs 0.6
    assert wm[0] == pytest.approx(gm[0])         # c equal by default
    # the global table is NOT mutated
    assert TISSUE_PROPERTIES[3][2] == pytest.approx(0.6)


def test_map_labels_uses_contrasted_properties():
    """map_labels_to_all can consume a custom (contrasted) table, producing an
    attenuation field that differs between GM and WM voxels."""
    from brain_fwi.phantoms.properties import (
        map_labels_to_all, tissue_properties_contrasted)
    labels = jnp.array([2, 3, 2, 3], dtype=jnp.int32)  # GM, WM, GM, WM
    props = tissue_properties_contrasted()
    a = np.asarray(map_labels_to_all(labels, properties=props)["attenuation"])
    assert a[0] == pytest.approx(0.6) and a[1] == pytest.approx(0.9)
    # default call unchanged (no contrast)
    a0 = np.asarray(map_labels_to_all(labels)["attenuation"])
    assert np.allclose(a0, 0.6)


def test_gm_wm_separability_zero_on_equal_field_large_on_contrasted():
    """The separability metric (WM-GM contrast-to-noise) is ~0 when the field
    cannot tell the tissues apart, and large when it can."""
    from brain_fwi.robustness.metrics import gm_wm_separability
    rng = np.random.default_rng(0)
    gm_mask = np.zeros((8, 8, 8), bool); gm_mask[:4] = True
    wm_mask = np.zeros((8, 8, 8), bool); wm_mask[4:] = True

    equal = np.full((8, 8, 8), 0.6, np.float32) + rng.normal(0, 0.02, (8, 8, 8))
    contrasted = np.where(wm_mask, 0.9, 0.6).astype(np.float32) + rng.normal(0, 0.02, (8, 8, 8))

    s_equal = gm_wm_separability(equal, gm_mask, wm_mask)
    s_contrast = gm_wm_separability(contrasted, gm_mask, wm_mask)
    assert s_equal < 0.5, f"equal field should not separate: {s_equal:.3f}"
    assert s_contrast > 3.0, f"contrasted field should separate strongly: {s_contrast:.3f}"

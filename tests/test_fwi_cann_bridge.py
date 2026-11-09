"""FWI -> CANN bridge: discover a tissue's alpha(omega) law from multi-band FWI.

Multi-band FWI recovers attenuation at several centre frequencies; assembling
those per-band estimates gives samples of alpha(omega) per tissue region, on
which the CANN L0 discovery (Kuhl's automated model discovery) fits an
interpretable constitutive law — the two-stage "measure at many conditions, then
discover" workflow, without needing the forward to evaluate a CANN.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_discover_tissue_alpha_law_recovers_power_law_from_bands():
    """Given per-band alpha(f_k) samples of a power law, discovery returns a
    sparse law that fits the samples."""
    from brain_fwi.constitutive import discover_tissue_alpha_law
    f = np.array([75e3, 125e3, 200e3, 275e3])
    alpha = 0.6 * (f / 1e6) ** 1.3
    res = discover_tissue_alpha_law(f, alpha, exponents=(1.0, 1.3, 2.0), l0_penalty=1e-3)
    assert res.n_terms >= 1 and res.rel_rmse < 0.1
    assert np.all(np.asarray(res.weights) >= -1e-9)


def test_bridge_ranks_gm_below_wm_under_realistic_bias():
    """Even with the measured FWI bias (WM attenuation over-estimated), the
    discovered total attenuation ranks GM < WM — the tissues stay ordered."""
    from brain_fwi.constitutive import discover_tissue_alpha_law
    f = np.array([75e3, 125e3, 200e3, 275e3])
    rng = np.random.default_rng(0)
    gm = 0.6 * (f / 1e6) ** 1.3 * (1 + rng.normal(0, 0.05, 4))
    wm = 1.6 * (f / 1e6) ** 1.3 * (1 + rng.normal(0, 0.05, 4))   # over-estimated, WM>GM
    rg = discover_tissue_alpha_law(f, gm, exponents=(1.0, 1.3, 2.0))
    rw = discover_tissue_alpha_law(f, wm, exponents=(1.0, 1.3, 2.0))
    assert float(np.sum(rw.weights)) > float(np.sum(rg.weights)), "WM should rank above GM"


def test_fwi_result_exposes_attenuation_history_field():
    """run_fwi's result carries a per-band attenuation history (None-filled for
    the velocity-only path)."""
    from brain_fwi.inversion.fwi import FWIResult
    r = FWIResult(velocity=None, velocity_history=[], loss_history=[], params=None)
    assert hasattr(r, "attenuation_history")
    assert r.attenuation_history == [] or r.attenuation_history is None

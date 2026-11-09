"""Reconstruction-quality metrics for the robustness harness (MAITE Metric-like)."""
from __future__ import annotations
import numpy as np


def brain_roi_rmse(recon, c_true, roi) -> float:
    """RMSE (m/s) of the reconstructed sound speed over the brain ROI."""
    recon = np.asarray(recon); c_true = np.asarray(c_true); roi = np.asarray(roi, bool)
    return float(np.sqrt(np.mean((recon[roi] - c_true[roi]) ** 2)))


def lesion_rmse(recon, c_true, lesion_mask) -> float:
    m = np.asarray(lesion_mask, bool)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((np.asarray(recon)[m] - np.asarray(c_true)[m]) ** 2)))


def gm_wm_separability(field, gm_mask, wm_mask) -> float:
    """Grey/white separability of a scalar field (issue #47).

    Contrast-to-noise between the two tissues::

        |mean_WM - mean_GM| / sqrt(0.5 * (var_WM + var_GM))

    ~0 when the field cannot tell GM from WM (the ITRUSST default, where both
    are 0.6 dB/cm/MHz), and grows as the field (e.g. a recovered attenuation
    map) separates them. This is the metric a GM/WM demonstration is scored on.
    """
    f = np.asarray(field, np.float64)
    gm = f[np.asarray(gm_mask, bool)]
    wm = f[np.asarray(wm_mask, bool)]
    if gm.size == 0 or wm.size == 0:
        return float("nan")
    pooled = np.sqrt(0.5 * (gm.var() + wm.var())) + 1e-9
    return float(abs(wm.mean() - gm.mean()) / pooled)

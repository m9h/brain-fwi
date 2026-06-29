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

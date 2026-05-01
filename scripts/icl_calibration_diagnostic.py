"""ICL syn-vs-exp gap: is it calibration or physics?

Follow-up to ``icl_syn_vs_exp_diagnostic.py`` — that script found a
median 0.871 correlation between synthetic and experimental traces
that did NOT stratify with phantom-traversal, ruling out skull
attenuation/dispersion as the dominant gap source.

This script tests the alternative hypothesis: the gap is dominated
by per-pair time-shifts (probe-position / source-signature drift /
clock skew). If applying the optimal lag per pair collapses the
residual, the gap is calibration, not missing physics.

Method per pair:
  1. Bandpass 200 kHz – 1 MHz (matches §10 of the previous diag).
  2. Cross-correlate syn vs exp; take the lag at the peak.
  3. Score correlation BEFORE shift and AFTER applying the shift.
  4. Report the distribution of lags (structured = calibration) and
     the median correlation gain.

Run:
    ICL_DUAL_PROBE_PATH=/path/to/icl-dual-probe-2023 \
        uv run python scripts/icl_calibration_diagnostic.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from brain_fwi.data.icl_dual_probe import load_icl_dual_probe
from brain_fwi.diagnostics.calibration import (
    best_lag_correlation,
    perp_distance_xz,
)


def _bandpass(x: np.ndarray, dt: float, lo: float, hi: float) -> np.ndarray:
    fs = 1.0 / dt
    sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def _norm_corr(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main(n_pairs: int = 300, seed: int = 0, max_lag_us: float = 5.0):
    data_path = Path(
        os.environ.get(
            "ICL_DUAL_PROBE_PATH",
            "/data/datasets/icl-dual-probe-2023",
        )
    )
    syn = load_icl_dual_probe(data_path, mode="synthetic")
    exp = load_icl_dual_probe(data_path, mode="experimental")
    try:
        positions = syn["transducer_positions_m"]
        pairs = syn["src_recv_pairs"]
        dt = float(syn["dt"])
        max_lag = int(max_lag_us * 1e-6 / dt)

        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(pairs.shape[0], size=n_pairs, replace=False))
        syn_chunk = np.asarray(syn["traces"][idx, :], dtype=np.float64)
        exp_chunk = np.asarray(exp["traces"][idx, :], dtype=np.float64)

        syn_bp = _bandpass(syn_chunk, dt, 200e3, 1e6)
        exp_bp = _bandpass(exp_chunk, dt, 200e3, 1e6)

        centre = positions.mean(axis=0)
        before, after, lags_samp, perps_mm = [], [], [], []
        kept_pair_idx = []
        for j, (s, e) in enumerate(zip(syn_bp, exp_bp)):
            if np.linalg.norm(s) < 1e-9 or np.linalg.norm(e) < 1e-9:
                continue
            before.append(_norm_corr(s, e))
            k, c_aft = best_lag_correlation(s, e, max_lag)
            after.append(c_aft)
            lags_samp.append(k)
            src_pos = positions[pairs[idx[j], 0]]
            rcv_pos = positions[pairs[idx[j], 1]]
            perps_mm.append(perp_distance_xz(src_pos, rcv_pos, centre) * 1000)
            kept_pair_idx.append(j)

        before = np.asarray(before)
        after = np.asarray(after)
        lags_us = np.asarray(lags_samp) * dt * 1e6
        perps_mm = np.asarray(perps_mm)
        gain = after - before

        print(f"{len(before)} pairs analysed (max_lag = {max_lag_us:.1f} µs)")
        print()
        print(f"Median corr BEFORE shift: {np.median(before):+.3f}")
        print(f"Median corr AFTER shift:  {np.median(after):+.3f}")
        print(f"Median gain:              {np.median(gain):+.3f}")
        print(f"Fraction with corr>0.95 BEFORE: {(before > 0.95).mean():.1%}")
        print(f"Fraction with corr>0.95 AFTER:  {(after > 0.95).mean():.1%}")
        print()
        print("Lag distribution (µs):")
        for q in (5, 25, 50, 75, 95):
            print(f"  p{q:>2d}: {np.percentile(lags_us, q):+.2f} µs")
        print(f"  mean:  {lags_us.mean():+.2f} µs")
        print(f"  std:   {lags_us.std():.2f} µs")
        print()

        # Stratify the gain by whether the lag is "small" or "large".
        # Large structured lags = calibration; small = signature/coupling.
        small_lag = np.abs(lags_us) < 0.5
        print(f"Pairs with |lag| < 0.5 µs:  n={small_lag.sum()}, "
              f"median gain {np.median(gain[small_lag]):+.3f}")
        print(f"Pairs with |lag| >= 0.5 µs: n={(~small_lag).sum()}, "
              f"median gain {np.median(gain[~small_lag]):+.3f}")

        # ── Post-shift residual stratified by phantom traversal ──
        # If physics is the cause of the remaining ~6%, paths through
        # the phantom (small perp) should have lower post-shift corr
        # than tangential paths.
        print("Post-shift correlation stratified by perp from centre:")
        bins = [(0, 20), (20, 40), (40, 60), (60, 80)]
        print(f"  {'perp (mm)':<12} {'n':>4} {'med after':>10}  {'med before':>10}")
        for lo, hi in bins:
            mask = (perps_mm >= lo) & (perps_mm < hi)
            if mask.sum() < 5:
                continue
            print(f"  [{lo:>3},{hi:>3})    {mask.sum():>4} "
                  f"{np.median(after[mask]):>+9.3f}  {np.median(before[mask]):>+9.3f}")
        print()

        # Diagnosis verdict
        gain_med = float(np.median(gain))
        after_med = float(np.median(after))
        print()
        print("=" * 60)
        if after_med > 0.95 and gain_med > 0.05:
            print("VERDICT: time-shift recovery dominates the gap.")
            print("→ Calibration/positioning is the residual driver.")
            print("→ Phase 5 missing-physics thesis NOT supported by this data.")
        elif gain_med > 0.03:
            print("VERDICT: partial — lag fixes some, residual remains.")
            print("→ Mixed: calibration AND another effect.")
        else:
            print("VERDICT: time-shift gives no gain.")
            print("→ Gap is amplitude/spectral, not timing.")
            print("→ Could be missing physics OR amplitude calibration.")
        print("=" * 60)
    finally:
        syn["_h5_file"].close()
        exp["_h5_file"].close()


if __name__ == "__main__":
    main()

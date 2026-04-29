"""ICL dual-probe: how different are the synthetic and experimental
traces, and does the gap depend on the path geometry?

Computes the bandpassed cosine similarity between the published
synthetic trace and the matching experimental trace for a sample of
src/rcv pairs, stratified by:

  - **baseline** (src–rcv straight-line distance)
  - **perp from ring centre** (small = path passes near the phantom
    centre and traverses the most skull/brain; large = tangential)

Result on 300 random pairs (seed 0), 200 kHz – 1 MHz bandpass,
measured 2026-04-29 with `dt = 44 ns`:

    Median syn-vs-exp correlation: 0.871
    perp [0,20)  mm  n=63   median corr 0.893
    perp [20,40) mm  n=65   median corr 0.864
    perp [40,60) mm  n=108  median corr 0.898
    perp [60,80) mm  n=40   median corr 0.744
    Fraction with corr > 0.95: 12.3 %
    Fraction with corr < 0.50: 5.1 %

Interpretation: there IS a real synthetic↔experimental gap (median
~13 % residual energy), but it does **not** stratify cleanly with
"more skull in path". So this dataset alone does not provide a clean
single-number motivation for "Phase 5 needed because skull losses".
A controlled benchmark where both lossless and lossy forward results
are owned end-to-end (e.g. ITRUSST) is the right place to demonstrate
the attenuation contribution.

Run:
    ICL_DUAL_PROBE_PATH=/path/to/icl-dual-probe-2023 \
        uv run python scripts/icl_syn_vs_exp_diagnostic.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from brain_fwi.data.icl_dual_probe import load_icl_dual_probe


def _perp_distance(s: np.ndarray, r: np.ndarray, c: np.ndarray) -> float:
    """Perpendicular distance from point ``c`` to the line through ``s``
    and ``r`` in the xz plane."""
    s2 = np.array([s[0], s[2]])
    r2 = np.array([r[0], r[2]])
    c2 = np.array([c[0], c[2]])
    line = r2 - s2
    L = float(np.linalg.norm(line))
    if L < 1e-9:
        return 0.0
    cross = float(line[0] * (c2[1] - s2[1]) - line[1] * (c2[0] - s2[0]))
    return abs(cross) / L


def main(n_pairs: int = 300, seed: int = 0):
    data_path = Path(
        os.environ.get(
            "ICL_DUAL_PROBE_PATH",
            "/Users/mhough/Workspace/brain-fwi/data/icl-dual-probe-2023",
        )
    )
    syn = load_icl_dual_probe(data_path, mode="synthetic")
    exp = load_icl_dual_probe(data_path, mode="experimental")
    try:
        positions = syn["transducer_positions_m"]
        pairs = syn["src_recv_pairs"]
        dt = syn["dt"]
        centre = positions.mean(axis=0)

        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(positions.shape[0] * 0 + pairs.shape[0],
                                  size=n_pairs, replace=False))
        syn_chunk = np.asarray(syn["traces"][idx, :]).astype(np.float64)
        exp_chunk = np.asarray(exp["traces"][idx, :]).astype(np.float64)

        fnyq = 0.5 / dt
        sos = butter(4, [2e5 / fnyq, 1e6 / fnyq], btype="bandpass", output="sos")

        baselines, perps, corrs = [], [], []
        for i, ti in enumerate(idx):
            s, r = pairs[ti]
            sp, rp = positions[s], positions[r]
            d = float(np.linalg.norm(sp - rp))
            if d < 0.05:
                continue
            syn_b = sosfiltfilt(sos, syn_chunk[i])
            exp_b = sosfiltfilt(sos, exp_chunk[i])
            syn_n = syn_b / (np.linalg.norm(syn_b) + 1e-12)
            exp_n = exp_b / (np.linalg.norm(exp_b) + 1e-12)
            baselines.append(d)
            perps.append(_perp_distance(sp, rp, centre))
            corrs.append(float(np.dot(syn_n, exp_n)))

        baselines = np.array(baselines)
        perps = np.array(perps)
        corrs = np.array(corrs)

        print(f"{len(baselines)} pairs analysed (seed={seed}, n={n_pairs} requested)")
        print(f"baseline range:        {baselines.min()*1e3:.0f}-{baselines.max()*1e3:.0f} mm")
        print(f"perp from ring centre: {perps.min()*1e3:.0f}-{perps.max()*1e3:.0f} mm")
        print(f"\nMedian syn-vs-exp correlation: {np.median(corrs):+.3f}")
        print(f"Fraction with corr > 0.95:     {(corrs > 0.95).mean()*100:.1f} %")
        print(f"Fraction with corr < 0.50:     {(corrs < 0.50).mean()*100:.1f} %")

        print("\nStratified by perpendicular distance from ring centre:")
        print(f"  {'perp bin (mm)':<16s}{'n':>5s}{'median corr':>15s}")
        for lo, hi in zip([0, 20, 40, 60, 80], [20, 40, 60, 80, 200]):
            m = (perps * 1e3 >= lo) & (perps * 1e3 < hi)
            if m.sum() == 0:
                continue
            print(f"  [{lo:>3d}, {hi:>3d})    {m.sum():>5d}   {np.median(corrs[m]):>+12.3f}")

        print("\nStratified by baseline:")
        print(f"  {'baseline (mm)':<16s}{'n':>5s}{'median corr':>15s}")
        for lo, hi in zip([0, 50, 100, 150], [50, 100, 150, 300]):
            m = (baselines * 1e3 >= lo) & (baselines * 1e3 < hi)
            if m.sum() == 0:
                continue
            print(f"  [{lo:>3d}, {hi:>3d})    {m.sum():>5d}   {np.median(corrs[m]):>+12.3f}")

    finally:
        syn["_h5_file"].close()
        exp["_h5_file"].close()


if __name__ == "__main__":
    main()

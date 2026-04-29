"""Loader contract for the ICL dual-probe transcranial USCT dataset.

Reference: Robins TC, Cueto C, Cudeiro J, et al. (2023). Dual-probe
transcranial full-waveform inversion: a brain phantom feasibility
study. Ultrasound Med. Biol. 49(10):2302-2315.
DOI: 10.17632/rbx3ybd5zx.1 (Mendeley Data, CC BY 4.0)

Tests are skipped if the dataset is not present locally. The default
search path is /data/datasets/icl-dual-probe-2023/. Set the env var
ICL_DUAL_PROBE_PATH to override.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

DEFAULT_PATH = Path(
    os.environ.get(
        "ICL_DUAL_PROBE_PATH",
        "/data/datasets/icl-dual-probe-2023",
    )
)

REQUIRED_FILES = [
    "02_Synthetic_USCT_Brain.mat",
    "elementList.txt",
    "elementRelas.txt",
    "signal_filtered.mat",
]


def _data_present() -> bool:
    return all((DEFAULT_PATH / f).exists() for f in REQUIRED_FILES)


needs_data = pytest.mark.skipif(
    not _data_present(),
    reason=f"ICL dual-probe data not at {DEFAULT_PATH}",
)


def test_loader_module_is_importable():
    """``load_icl_dual_probe`` is exported from the data module."""
    from brain_fwi.data.icl_dual_probe import load_icl_dual_probe
    assert callable(load_icl_dual_probe)


@needs_data
def test_load_synthetic_returns_required_fields():
    """``load_icl_dual_probe(path, mode='synthetic')`` returns a dict
    with the fields the FWI pipeline needs.

    Note: ``true_velocity`` is intentionally absent — the synthetic
    ground-truth file ``TrueVp.mat`` is missing from the published
    archive (see docs/datasets/icl-dual-probe-2023.md).
    """
    from brain_fwi.data.icl_dual_probe import load_icl_dual_probe

    sample = load_icl_dual_probe(DEFAULT_PATH, mode="synthetic")
    try:
        expected = {
            "traces",
            "transducer_positions_m",
            "source_signal",
            "dt",
            "src_recv_pairs",
        }
        missing = expected - set(sample.keys())
        assert not missing, f"missing keys: {missing}"

        assert sample["traces"].shape == (442368, 3328)
        assert sample["transducer_positions_m"].shape == (3072, 3)
        assert sample["src_recv_pairs"].shape == (442368, 2)
        assert sample["source_signal"].shape[1] == 384
        assert sample["src_recv_pairs"].min() >= 0
        assert sample["src_recv_pairs"].max() < 3072
        assert sample["dt"] == pytest.approx(44e-9, rel=1e-3)
    finally:
        sample["_h5_file"].close()


@needs_data
def test_synthetic_geometry_is_planar_xz():
    """Element table places all 3072 elements in the xz plane (y=0)
    — confirms the paper's 2.5D imaging geometry."""
    from brain_fwi.data.icl_dual_probe import load_icl_dual_probe

    sample = load_icl_dual_probe(DEFAULT_PATH, mode="synthetic")
    try:
        positions = sample["transducer_positions_m"]
        assert np.allclose(positions[:, 1], 0.0), \
            "Expected all elements in xz plane (paper §2.1, 2.5D geometry)"
    finally:
        sample["_h5_file"].close()


@needs_data
def test_synthetic_first_arrival_speed_is_physical():
    """Linear fit of first-arrival time vs src-rcv distance recovers
    a path-averaged speed of sound in [1450, 1600] m/s.

    This jointly validates ``dt``, the trace orientation, and the
    src/rcv index mapping. If ``dt`` were the wrong value (e.g. 50 ns
    from the simulation step) the fit would give c ≈ 1734 m/s, which
    is non-physical for water + brain phantom.
    """
    from brain_fwi.data.icl_dual_probe import load_icl_dual_probe

    sample = load_icl_dual_probe(DEFAULT_PATH, mode="synthetic")
    try:
        traces = sample["traces"]
        positions = sample["transducer_positions_m"]
        pairs = sample["src_recv_pairs"]
        dt = sample["dt"]

        rng = np.random.default_rng(0)
        idx_unsorted = rng.choice(traces.shape[0], size=200, replace=False)
        order = np.argsort(idx_unsorted)
        chunk_sorted = np.asarray(traces[idx_unsorted[order], :])
        chunk = np.empty_like(chunk_sorted)
        chunk[order] = chunk_sorted

        src = pairs[idx_unsorted, 0]
        rcv = pairs[idx_unsorted, 1]
        geom_dist = np.linalg.norm(positions[src] - positions[rcv], axis=1)

        abs_chunk = np.abs(chunk)
        thresh = 0.05 * abs_chunk.max(axis=1, keepdims=True)
        above = abs_chunk > thresh
        first_arrival = np.where(
            above.any(axis=1), above.argmax(axis=1), -1
        )

        fit = (geom_dist > 0.10) & (first_arrival > 0)
        assert fit.sum() > 50, \
            f"too few long-baseline pairs in random sample: {fit.sum()}"

        slope, intercept = np.polyfit(
            geom_dist[fit], first_arrival[fit] * dt, 1
        )
        c_est = 1.0 / slope
        assert 1450.0 <= c_est <= 1600.0, \
            f"path-averaged speed of sound {c_est:.0f} m/s outside [1450,1600]"

        residual = first_arrival[fit] * dt - (slope * geom_dist[fit] + intercept)
        assert residual.std() < 1e-6, \
            f"first-arrival residual std {residual.std()*1e9:.0f} ns too large"
    finally:
        sample["_h5_file"].close()


@needs_data
def test_experimental_loads_with_dataset_root_key():
    """The experimental file uses root key ``dataset`` (vs the
    synthetic file's ``WaterShot`` copy-paste bug). Loader auto-detects
    either."""
    from brain_fwi.data.icl_dual_probe import load_icl_dual_probe

    sample = load_icl_dual_probe(DEFAULT_PATH, mode="experimental")
    try:
        assert sample["traces"].shape == (442368, 3328)
        assert sample["traces"].dtype == np.float64
    finally:
        sample["_h5_file"].close()

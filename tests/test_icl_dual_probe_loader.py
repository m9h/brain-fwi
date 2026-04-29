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

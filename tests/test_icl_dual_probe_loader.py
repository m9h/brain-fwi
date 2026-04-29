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
    "TrueVp.mat",
    "elementList.txt",
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
    with the fields the FWI pipeline needs."""
    from brain_fwi.data.icl_dual_probe import load_icl_dual_probe

    sample = load_icl_dual_probe(DEFAULT_PATH, mode="synthetic")
    expected = {
        "traces",
        "transducer_positions_m",
        "source_signal",
        "true_velocity",
        "dt",
        "src_recv_pairs",
    }
    missing = expected - set(sample.keys())
    assert not missing, f"missing keys: {missing}"

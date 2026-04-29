"""Loader for the ICL dual-probe transcranial USCT dataset.

Reference: Robins TC, Cueto C, Cudeiro J, et al. (2023). Dual-probe
transcranial full-waveform inversion: a brain phantom feasibility
study. Ultrasound Med. Biol. 49(10):2302-2315.
DOI: 10.17632/rbx3ybd5zx.1 (Mendeley Data, CC BY 4.0).

Geometry (from §2.1 of the paper):
- 2.5D imaging plane
- dual P4-1 cardiac probes, 24 elements each
- 16 rotation positions about the target → 384 total source elements
- ring array reconstruction over a water tank
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Union


def load_icl_dual_probe(
    path: Union[str, Path],
    *,
    mode: Literal["synthetic", "experimental"] = "synthetic",
) -> Dict:
    """Load one acquisition mode of the ICL dual-probe dataset.

    Args:
        path: directory containing the unzipped Mendeley archive.
        mode: ``"synthetic"`` reads ``02_Synthetic_USCT_Brain.mat``,
            ``"experimental"`` reads ``01_Experimental_USCT_Brain.mat``.

    Returns:
        Dict with keys ``traces``, ``transducer_positions_m``,
        ``source_signal``, ``true_velocity`` (synthetic only),
        ``dt``, ``src_recv_pairs``.
    """
    raise NotImplementedError(
        "load_icl_dual_probe body fills in once the dataset's .mat "
        "structure is inspected on disk"
    )

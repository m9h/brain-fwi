"""Loader for the ICL dual-probe transcranial USCT dataset.

Reference: Robins TC, Cueto C, Cudeiro J, et al. (2023). Dual-probe
transcranial full-waveform inversion: a brain phantom feasibility
study. Ultrasound Med. Biol. 49(10):2302-2315.
DOI: 10.17632/rbx3ybd5zx.1 (Mendeley Data, CC BY 4.0).

Geometry (from §2.1 of the paper):
- 2.5D imaging plane
- dual P4-1 cardiac probes, 24 elements each
- 16 rotation positions about the target → 384 unique source positions
- each source pings 1152 of 3072 receiver elements (442368 traces total)
- ring array reconstruction over a water tank

Quirks of the published archive — see ``docs/datasets/icl-dual-probe-2023.md``
for the full audit. Briefly:
1. The Mendeley description lists ``TrueVp.mat`` (synthetic ground-truth
   velocity) but the file was never uploaded — the API enumerates only 11
   files. We therefore do **not** return ``true_velocity``; downstream
   callers must obtain it elsewhere or omit ground-truth comparison.
2. ``02_Synthetic_USCT_Brain.mat`` stores its trace matrix under the root
   key ``WaterShot`` (a copy-paste bug from the authors' upload script).
   The experimental files use ``dataset``. The loader auto-detects.
3. ``dt`` is not stated in either the paper or the archive. We use
   ``11 / 250e6 s`` ≈ 44 ns, matching the Verasonics native rate
   (250 MHz / 11) and reproducing the ring-array round-trip time within
   0.2 % of the geometry implied by ``elementList.txt``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Union

import h5py
import numpy as np
import scipy.io

DT_SECONDS: float = 11.0 / 250e6
"""Per-sample time step of the saved traces, in seconds.

Inferred — the paper and dataset omit this. Verasonics systems clock
ADC sampling at 250 MHz / N; N=11 → 22.727 MHz. Cross-check: ring radius
≈ 0.11 m (from ``elementList.txt``) gives a max round-trip of 0.22 m /
1500 m s⁻¹ ≈ 146.7 µs, while 3328 samples × 44 ns = 146.4 µs.
"""

_TRACE_FILES = {
    "synthetic": "02_Synthetic_USCT_Brain.mat",
    "experimental": "01_Experimental_USCT_Brain.mat",
}


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
        A dict with keys:

        - ``traces`` : ``h5py.Dataset`` of shape ``(n_traces, n_time)``
          ``= (442368, 3328)``. Lazily backed by the open .mat file —
          do NOT close ``_h5_file`` until you are done indexing.
        - ``transducer_positions_m`` : ``(3072, 3)`` float64 array of
          ``(x, y, z)`` element positions in metres.
        - ``src_recv_pairs`` : ``(n_traces, 2)`` int64 array of
          ``(src_idx, recv_idx)``, **0-based** indices into
          ``transducer_positions_m``.
        - ``source_signal`` : ``(n_signal_samples, 384)`` float64 array
          of source wavelets (one per unique source position). Note
          ``n_signal_samples = 2730 < n_time = 3328`` — pad as needed.
        - ``dt`` : float, seconds per sample (see :data:`DT_SECONDS`).
        - ``_h5_file`` : the open ``h5py.File`` backing ``traces``.
          Caller is responsible for ``.close()`` when done.

    The contract intentionally omits ``true_velocity``; the synthetic
    ground-truth model is missing from the published archive.
    """
    if mode not in _TRACE_FILES:
        raise ValueError(f"mode must be 'synthetic' or 'experimental', got {mode!r}")

    root = Path(path)
    trace_path = root / _TRACE_FILES[mode]
    h5 = h5py.File(trace_path, "r")
    keys = list(h5.keys())
    if len(keys) != 1:
        h5.close()
        raise ValueError(
            f"{trace_path.name}: expected one root dataset, got {keys}"
        )
    traces = h5[keys[0]]

    positions_table = np.loadtxt(
        root / "elementList.txt", delimiter=",", skiprows=1
    )
    transducer_positions_m = positions_table[:, 1:4].astype(np.float64)

    relas = np.loadtxt(
        root / "elementRelas.txt", delimiter=",", skiprows=1, dtype=np.int64
    )
    src_recv_pairs = relas[:, 1:3] - 1

    signal_mat = scipy.io.loadmat(root / "signal_filtered.mat")
    source_signal = np.asarray(signal_mat["signal"], dtype=np.float64)

    return {
        "traces": traces,
        "transducer_positions_m": transducer_positions_m,
        "src_recv_pairs": src_recv_pairs,
        "source_signal": source_signal,
        "dt": DT_SECONDS,
        "_h5_file": h5,
    }

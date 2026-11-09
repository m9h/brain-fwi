"""Tests for the parallel-Modal-runner aug-range partitioning.

The launcher in `scripts/modal_gen_phase0.py` slices a global aug range
into N contiguous per-rank slices. Each rank writes samples in
``[aug_start_i, aug_end_i)`` to its own ``part_NN/`` directory; the
merge step concatenates them. Two failure modes the tests below guard:

* **Overlap**: two ranks claim the same aug index → duplicate sample IDs
  → merge fails or, worse, silently picks one and discards the other.
* **Gaps**: an index falls between two ranks → merged dataset has a
  hole, but the manifest still claims contiguous coverage.

Both modes silently corrupt downstream training data, so we test the
partition arithmetic directly rather than running Modal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Importable script — see test_phase0_at_scale.py for the same trick.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _compute_rank_slices(
    aug_start: int, aug_end: int, n_ranks: int,
) -> list[tuple[int, int, int]]:
    """Reproduce the slicing logic in modal_gen_phase0.main."""
    total = aug_end - aug_start
    if n_ranks <= 0 or n_ranks > total:
        n_ranks = max(1, min(n_ranks, total))
    base = total // n_ranks
    extra = total % n_ranks
    rank_args = []
    cursor = aug_start
    for rank in range(n_ranks):
        slice_size = base + (1 if rank < extra else 0)
        rank_args.append((rank, cursor, cursor + slice_size))
        cursor += slice_size
    return rank_args


# ---------------------------------------------------------------------------
# Coverage: every aug index hits exactly one rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("aug_start, aug_end, n_ranks", [
    (0, 256, 8),     # production-style case
    (0, 1024, 8),    # the original v1b layout
    (0, 100, 4),     # n_augments default
    (50, 200, 8),    # non-zero start
    (0, 7, 8),       # ranks > samples — clamps to 7
    (0, 8, 8),       # exactly one sample per rank
    (0, 9, 8),       # one extra goes to rank 0
])
def test_slices_cover_range_with_no_overlap_or_gap(aug_start, aug_end, n_ranks):
    slices = _compute_rank_slices(aug_start, aug_end, n_ranks)

    covered: set[int] = set()
    for _, s, e in slices:
        for k in range(s, e):
            assert k not in covered, (
                f"index {k} is covered by more than one rank in {slices!r}"
            )
            covered.add(k)

    expected = set(range(aug_start, aug_end))
    assert covered == expected, (
        f"slices missed indices {expected - covered} or covered "
        f"out-of-range {covered - expected}"
    )


def test_slices_are_contiguous_and_increasing():
    slices = _compute_rank_slices(0, 256, 8)
    for i, (_, s, e) in enumerate(slices):
        assert s < e, f"rank {i} has empty or inverted slice [{s}, {e})"
        if i + 1 < len(slices):
            next_s = slices[i + 1][1]
            assert e == next_s, (
                f"rank {i} ends at {e} but rank {i+1} starts at {next_s} — "
                f"gap or overlap"
            )


def test_slices_balance_close_to_uniform():
    """Within rounding, ranks should get sizes within 1 of each other."""
    slices = _compute_rank_slices(0, 1000, 8)
    sizes = [e - s for _, s, e in slices]
    assert max(sizes) - min(sizes) <= 1, (
        f"rank sizes {sizes} differ by more than 1; load-balance broken"
    )


# ---------------------------------------------------------------------------
# Sample-ID uniqueness across ranks (would have caught a v1b/v2 commingle)
# ---------------------------------------------------------------------------


def _expected_sample_id(subj: int, aug: int, phantom: str = "mida") -> str:
    """Mirror scripts/gen_phase0.py:344 sample_id format."""
    return f"{phantom}_{subj:03d}_{aug:03d}"


def test_sample_ids_unique_across_ranks():
    """No two ranks ever produce the same sample ID for the same subject."""
    slices = _compute_rank_slices(0, 256, 8)
    ids: set[str] = set()
    for _, s, e in slices:
        for aug in range(s, e):
            sid = _expected_sample_id(0, aug)
            assert sid not in ids, f"duplicate sample id {sid} across ranks"
            ids.add(sid)
    assert len(ids) == 256


def test_sample_ids_well_formed():
    """Sanity on the ID format we rely on for downstream debugging."""
    sid = _expected_sample_id(0, 17)
    assert sid == "mida_000_017", f"sample_id format drift: {sid!r}"

"""TDD spec for ``merge_phase0_parts``: combine N rank-output dirs
from a parallel ``gen_phase0`` run into one canonical dataset.

Each part dir has its own ``shards/00000.h5`` (numbered from 0) and
``manifest.json``. Merge:
  - copies all shards into the destination's ``shards/`` with non-
    colliding global indices
  - writes a unified manifest covering the union of completed ids
  - rejects duplicate sample ids across parts (would mean overlapping
    --aug-start/--aug-end ranges, a launcher bug)
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest


def _make_part(root: Path, version: str, sample_ids: list, shard_size: int = 100):
    """Synthesise a minimal valid part-dir with one shard."""
    from brain_fwi.data.sharded_writer import ShardedWriter

    w = ShardedWriter(
        root, shard_size=shard_size, version=version,
        metadata={"phantom": "test", "grid_size": 16},
    )
    for sid in sample_ids:
        w.write({
            "sample_id": sid,
            "sound_speed_voxel": np.zeros((4, 4, 4), dtype=np.float32),
        })
    w.close()


def test_merge_two_disjoint_parts_into_one_dataset(tmp_path: Path):
    """Two parts with disjoint sample ids merge into a single dataset
    whose manifest is the union and whose shards copy across."""
    from brain_fwi.data.merge import merge_phase0_parts

    p0 = tmp_path / "part_0"
    p1 = tmp_path / "part_1"
    out = tmp_path / "merged"
    _make_part(p0, "phase0_v1", ["mida_000_000", "mida_000_001"])
    _make_part(p1, "phase0_v1", ["mida_000_002", "mida_000_003"])

    merge_phase0_parts([p0, p1], out)

    manifest = json.loads((out / "manifest.json").read_text())
    assert sorted(manifest["completed"]) == [
        "mida_000_000", "mida_000_001", "mida_000_002", "mida_000_003",
    ]
    shards = sorted((out / "shards").glob("*.h5"))
    assert len(shards) >= 1, "no shards copied"
    found_ids = set()
    for s in shards:
        with h5py.File(s, "r") as f:
            found_ids.update(f.keys())
    assert found_ids == set(manifest["completed"])


def test_merge_rejects_duplicate_sample_ids(tmp_path: Path):
    """Overlapping sample ids across parts ⇒ ValueError. This catches
    the launcher bug of overlapping aug ranges."""
    from brain_fwi.data.merge import merge_phase0_parts

    p0 = tmp_path / "part_0"
    p1 = tmp_path / "part_1"
    _make_part(p0, "phase0_v1", ["mida_000_000", "mida_000_001"])
    _make_part(p1, "phase0_v1", ["mida_000_001", "mida_000_002"])

    with pytest.raises(ValueError, match="duplicate"):
        merge_phase0_parts([p0, p1], tmp_path / "merged")


def test_merge_rejects_version_mismatch(tmp_path: Path):
    """Parts with different ``version`` strings must not merge silently."""
    from brain_fwi.data.merge import merge_phase0_parts

    p0 = tmp_path / "part_0"
    p1 = tmp_path / "part_1"
    _make_part(p0, "phase0_v1", ["mida_000_000"])
    _make_part(p1, "phase0_v2", ["mida_000_001"])

    with pytest.raises(ValueError, match="version"):
        merge_phase0_parts([p0, p1], tmp_path / "merged")

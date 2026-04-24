"""Tests for ShardedReader: random access + iteration over Phase-0 shards.

Captures the contract:

- ``len(reader)`` equals manifest.completed count.
- ``reader[i]`` (int) and ``reader[sample_id]`` both return a sample dict.
- Iteration preserves manifest.completed order (reproducible across runs).
- ``fields=[...]`` limits returned keys (for memory-cheap iteration).
- Crosses shard boundaries transparently.
- Raises on missing id / version mismatch.

Kept decoupled from ``ShardedWriter`` at the contract level, but uses
it as the fixture factory since it's the canonical producer of this
on-disk layout.
"""

from pathlib import Path

import numpy as np
import pytest

from brain_fwi.data import ShardedWriter
from brain_fwi.data.sharded_reader import ShardedReader


def _make_sample(sample_id: str, value: float = 0.0) -> dict:
    return {
        "sample_id": sample_id,
        "subject_id": "synth_000",
        "seed": abs(hash(sample_id)) % 10_000,
        "sound_speed_voxel": np.full((4, 4, 4), value, dtype=np.float32),
        "observed_data": np.full((8, 16), value, dtype=np.float32),
    }


@pytest.fixture
def small_dataset(tmp_path: Path) -> Path:
    with ShardedWriter(tmp_path, shard_size=3) as w:
        for i in range(7):  # 7 samples → 3 shards (3+3+1)
            w.write(_make_sample(f"s{i:02d}", value=float(i)))
    return tmp_path


class TestLenAndIteration:
    def test_len_matches_manifest(self, small_dataset: Path):
        r = ShardedReader(small_dataset)
        assert len(r) == 7

    def test_iter_preserves_manifest_order(self, small_dataset: Path):
        r = ShardedReader(small_dataset)
        ids = [s["sample_id"] for s in r]
        assert ids == [f"s{i:02d}" for i in range(7)]

    def test_iter_yields_arrays_and_attrs(self, small_dataset: Path):
        r = ShardedReader(small_dataset)
        first = next(iter(r))
        assert first["sample_id"] == "s00"
        assert first["subject_id"] == "synth_000"
        assert first["sound_speed_voxel"].shape == (4, 4, 4)
        np.testing.assert_allclose(first["sound_speed_voxel"], 0.0)


class TestRandomAccess:
    def test_by_int_index(self, small_dataset: Path):
        r = ShardedReader(small_dataset)
        s = r[3]
        assert s["sample_id"] == "s03"
        np.testing.assert_allclose(s["sound_speed_voxel"], 3.0)

    def test_by_sample_id(self, small_dataset: Path):
        r = ShardedReader(small_dataset)
        s = r["s05"]
        assert s["sample_id"] == "s05"
        np.testing.assert_allclose(s["sound_speed_voxel"], 5.0)

    def test_crosses_shard_boundary(self, small_dataset: Path):
        """Fixture has shards 0..2..3, 3..5, 6 — walk across each boundary."""
        r = ShardedReader(small_dataset)
        for i in range(7):
            assert r[i]["sample_id"] == f"s{i:02d}"

    def test_missing_sample_id_raises(self, small_dataset: Path):
        r = ShardedReader(small_dataset)
        with pytest.raises(KeyError, match="not in"):
            r["no_such_id"]

    def test_int_out_of_range_raises(self, small_dataset: Path):
        r = ShardedReader(small_dataset)
        with pytest.raises(IndexError):
            r[99]

    def test_negative_int_indexing(self, small_dataset: Path):
        r = ShardedReader(small_dataset)
        assert r[-1]["sample_id"] == "s06"


class TestFieldSelection:
    def test_fields_limits_returned_keys(self, small_dataset: Path):
        r = ShardedReader(small_dataset, fields=["sound_speed_voxel"])
        s = r[0]
        assert "sound_speed_voxel" in s
        assert "observed_data" not in s
        # sample_id always included as an identity anchor
        assert s["sample_id"] == "s00"

    def test_fields_empty_returns_only_id(self, small_dataset: Path):
        r = ShardedReader(small_dataset, fields=[])
        s = r[0]
        assert s == {"sample_id": "s00"}


class TestMetadata:
    def test_exposes_manifest_metadata(self, small_dataset: Path):
        r = ShardedReader(small_dataset)
        assert r.metadata["version"] == "phase0_v1"
        assert r.metadata["shard_size"] == 3

    def test_sample_ids_property(self, small_dataset: Path):
        r = ShardedReader(small_dataset)
        assert r.sample_ids == [f"s{i:02d}" for i in range(7)]

    def test_version_mismatch_raises(self, small_dataset: Path):
        with pytest.raises(ValueError, match="version"):
            ShardedReader(small_dataset, expected_version="phase0_v2")


class TestEmptyDataset:
    def test_empty_dataset(self, tmp_path: Path):
        ShardedWriter(tmp_path).close()  # writes manifest, no samples
        r = ShardedReader(tmp_path)
        assert len(r) == 0
        assert list(r) == []
        assert r.sample_ids == []


class TestMissingRoot:
    def test_missing_manifest_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="manifest"):
            ShardedReader(tmp_path / "nonexistent")

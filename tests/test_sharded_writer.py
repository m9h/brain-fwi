"""Tests for ShardedWriter: round-trip, resume, shard boundaries."""

import json
from pathlib import Path

import numpy as np
import pytest

from brain_fwi.data import ShardedWriter, load_sample


def _make_sample(sample_id: str, vol_shape=(8, 8, 8)) -> dict:
    rng = np.random.default_rng(abs(hash(sample_id)) % (2**32))
    return {
        "sample_id": sample_id,
        "subject_id": "brainweb_04",
        "seed": 0,
        "dx": 0.002,
        "sound_speed_voxel": rng.standard_normal(vol_shape).astype(np.float16),
        "density_voxel": rng.standard_normal(vol_shape).astype(np.float16),
        "transducer_positions": rng.standard_normal((16, 3)).astype(np.float32),
        "jwave_sha": "abc1234",
    }


class TestRoundTrip:
    def test_write_then_load(self, tmp_path: Path):
        with ShardedWriter(tmp_path, shard_size=10) as w:
            w.write(_make_sample("s0"))
        got = load_sample(tmp_path, "s0")
        assert got["sample_id"] == "s0"
        assert got["subject_id"] == "brainweb_04"
        assert got["sound_speed_voxel"].shape == (8, 8, 8)
        assert got["transducer_positions"].shape == (16, 3)

    def test_manifest_tracks_completed(self, tmp_path: Path):
        with ShardedWriter(tmp_path, shard_size=10) as w:
            w.write(_make_sample("a"))
            w.write(_make_sample("b"))
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["completed"] == ["a", "b"]
        assert manifest["version"] == "phase0_v1"


class TestResume:
    def test_existing_manifest_repopulates_state(self, tmp_path: Path):
        with ShardedWriter(tmp_path, shard_size=10) as w:
            w.write(_make_sample("s0"))
            w.write(_make_sample("s1"))

        with ShardedWriter(tmp_path, shard_size=10) as w2:
            assert w2.n_completed == 2
            assert w2.is_complete("s0")
            assert w2.is_complete("s1")
            assert not w2.is_complete("s2")

    def test_write_is_idempotent_for_completed(self, tmp_path: Path):
        with ShardedWriter(tmp_path, shard_size=10) as w:
            w.write(_make_sample("s0"))
            w.write(_make_sample("s0"))  # no-op
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["completed"].count("s0") == 1

    def test_version_mismatch_raises(self, tmp_path: Path):
        ShardedWriter(tmp_path, version="phase0_v1").close()
        with pytest.raises(ValueError, match="manifest version"):
            ShardedWriter(tmp_path, version="phase0_v2")

    def test_shard_size_mismatch_raises(self, tmp_path: Path):
        ShardedWriter(tmp_path, shard_size=100).close()
        with pytest.raises(ValueError, match="shard_size"):
            ShardedWriter(tmp_path, shard_size=50)


class TestShardBoundaries:
    def test_samples_split_across_shards(self, tmp_path: Path):
        shard_size = 3
        ids = [f"s{i}" for i in range(7)]
        with ShardedWriter(tmp_path, shard_size=shard_size) as w:
            for sid in ids:
                w.write(_make_sample(sid))

        shards = sorted((tmp_path / "shards").iterdir())
        # 7 samples / 3 per shard → 3 shards (3, 3, 1)
        assert len(shards) == 3
        assert shards[0].name == "00000.h5"
        assert shards[1].name == "00001.h5"
        assert shards[2].name == "00002.h5"

        # round-trip across shard boundaries
        for sid in ids:
            got = load_sample(tmp_path, sid)
            assert got["sample_id"] == sid


class TestMissingSampleId:
    def test_write_without_sample_id_raises(self, tmp_path: Path):
        with ShardedWriter(tmp_path) as w:
            with pytest.raises(KeyError, match="sample_id"):
                w.write({"subject_id": "x"})

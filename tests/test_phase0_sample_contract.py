"""Contract tests for Phase-0 sample dicts — what later phases require.

The Phase-0 pipeline writes sample dicts to HDF5 shards. Phase-2 (NPE
training, ``inference/dataprep.py``) and Phase-4 (FNO surrogate,
``surrogate/train.py``) consume those samples. If the dict-shape, dtype,
or value-range contract drifts silently, downstream training fails or
(worse) trains on garbage.

The bugs we found in v1b/v2-morning all "ran clean" — the dataset
serialised, the manifest looked fine, but the actual c-fields were
useless. These tests turn that into a CI failure.

Tests are gated on having a smoke-test shard available. To produce one
locally::

    uv run python scripts/gen_phase0.py --out /tmp/phase0_unit_smoke \\
        --phantom synthetic --grid-size 32 --dx 0.004 \\
        --n-subjects 1 --n-augments 2 --n-elements 32 \\
        --siren-pretrain-steps 100 --freq 5e4

CI / Modal smoke-tests should always populate at least one of the
known shard locations below.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import pytest


_SHARD_CANDIDATES = (
    Path("/tmp/phase0_smoke3/merged"),
    Path("/tmp/phase0_smoke/merged"),
    Path("/tmp/phase0_unit_smoke"),
)


def _find_shard_root() -> Path | None:
    for p in _SHARD_CANDIDATES:
        if (p / "manifest.json").exists():
            return p
    return None


def _load_sample(root: Path, sample_id: str) -> Dict[str, Any]:
    """Load a sample from a sharded dataset.

    ShardedWriter stores array-valued fields as HDF5 datasets and
    scalar / string / dict-valued fields as group attributes — both
    must be merged into a single dict to faithfully represent what
    downstream code receives via ``load_sample`` / ``ShardedReader``.
    """
    shard_dir = root / "shards"
    for f in sorted(shard_dir.glob("*.h5")):
        with h5py.File(f, "r") as h:
            if sample_id in h:
                g = h[sample_id]
                out: Dict[str, Any] = {k: np.asarray(g[k]) for k in g.keys()}
                for k, v in g.attrs.items():
                    out[k] = v
                return out
    raise FileNotFoundError(f"sample {sample_id} not found in any shard under {shard_dir}")


def _load_first_sample(root: Path) -> Dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    return _load_sample(root, manifest["completed"][0])


# ---------------------------------------------------------------------------
# Fixtures — a sample dict produced by the production pipeline.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shard_root() -> Path:
    root = _find_shard_root()
    if root is None:
        pytest.skip(
            "no Phase-0 smoke shard on disk; run a smoke-test first "
            "(see module docstring)"
        )
    return root


@pytest.fixture(scope="module")
def sample(shard_root: Path) -> Dict[str, Any]:
    return _load_first_sample(shard_root)


@pytest.fixture(scope="module")
def manifest(shard_root: Path) -> Dict[str, Any]:
    return json.loads((shard_root / "manifest.json").read_text())


# ---------------------------------------------------------------------------
# Field-presence: catches silent schema drift
# ---------------------------------------------------------------------------


# Fields gen_phase0.generate_sample writes. Drift here = downstream
# dataloaders crash with KeyError.
_PHASE0_REQUIRED_KEYS = frozenset({
    "sample_id", "subject_id", "subject_idx", "aug_idx", "seed",
    "dx", "freq_hz", "dt", "grid_shape",
    "tissue_labels", "sound_speed_voxel", "density_voxel",
    "transducer_positions", "sensor_positions",
    "source_signal", "observed_data",
    "siren_weights_bytes", "siren_arch",
})


# Fields Phase-2 NPE specifically requires.
_PHASE2_REQUIRED_KEYS = frozenset({
    "sample_id", "siren_weights_bytes", "siren_arch", "observed_data",
})


# Fields Phase-4 FNO specifically requires.
_PHASE4_REQUIRED_KEYS = frozenset({
    "sample_id", "sound_speed_voxel", "observed_data",
})


class TestSampleSchema:
    """Catches schema drift before downstream consumers do."""

    def test_phase0_writer_keys_present(self, sample):
        missing = _PHASE0_REQUIRED_KEYS - set(sample.keys())
        assert not missing, f"sample missing Phase-0 writer fields: {sorted(missing)}"

    def test_phase2_required_keys_present(self, sample):
        missing = _PHASE2_REQUIRED_KEYS - set(sample.keys())
        assert not missing, f"sample missing Phase-2 fields: {sorted(missing)}"

    def test_phase4_required_keys_present(self, sample):
        missing = _PHASE4_REQUIRED_KEYS - set(sample.keys())
        assert not missing, f"sample missing Phase-4 fields: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Shape / dtype: catches silent type-or-rank drift
# ---------------------------------------------------------------------------


class TestSampleShapesAndDtypes:

    def test_grid_shape_3d(self, sample):
        gs = np.asarray(sample["grid_shape"]).ravel()
        assert gs.shape == (3,), f"grid_shape rank mismatch: {gs.shape}"
        assert (gs > 0).all(), f"grid_shape has non-positive entries: {gs}"

    def test_volumes_are_3d_and_match_grid_shape(self, sample):
        gs = tuple(int(x) for x in np.asarray(sample["grid_shape"]).ravel())
        for k in ("tissue_labels", "sound_speed_voxel", "density_voxel"):
            arr = np.asarray(sample[k])
            assert arr.shape == gs, (
                f"{k} shape {arr.shape} != grid_shape {gs}"
            )

    def test_observed_data_is_3d(self, sample):
        obs = np.asarray(sample["observed_data"])
        assert obs.ndim == 3, (
            f"observed_data must be (N_src, N_t, N_recv); got rank {obs.ndim} "
            f"with shape {obs.shape}"
        )

    def test_observed_data_dtype_is_float32(self, sample):
        # observed_data is explicitly kept float32 in gen_phase0 because
        # pressure magnitudes overflow float16 at production frequencies.
        obs = np.asarray(sample["observed_data"])
        assert obs.dtype == np.float32, (
            f"observed_data dtype {obs.dtype} != float32 — Phase-2 summary "
            f"path expects float32; float16 will silently overflow"
        )

    def test_transducer_geometry_consistent(self, sample):
        """N_src and N_recv come from the helmet positions — must match obs."""
        pos = np.asarray(sample["transducer_positions"])
        sens = np.asarray(sample["sensor_positions"])
        obs = np.asarray(sample["observed_data"])
        n_src = obs.shape[0]
        n_recv = obs.shape[2]
        assert pos.shape[0] == n_src, (
            f"transducer_positions has {pos.shape[0]} rows but observed_data "
            f"has {n_src} sources"
        )
        assert sens.shape[0] == n_recv, (
            f"sensor_positions has {sens.shape[0]} rows but observed_data "
            f"has {n_recv} receivers"
        )


# ---------------------------------------------------------------------------
# Value-range: catches silent corruption a clean-shape sample can still hide
# ---------------------------------------------------------------------------


class TestSampleValueRanges:

    def test_sound_speed_in_physical_range(self, sample):
        c = np.asarray(sample["sound_speed_voxel"]).astype(np.float32)
        assert np.all(np.isfinite(c)), "sound_speed has NaN or inf"
        # Air ~343, water ~1500, brain ~1560, cortical bone ~2800.
        # Allow generous slack for jitter overshoot.
        assert c.min() >= 200.0, f"c.min()={c.min():.1f} below physical range"
        assert c.max() <= 4500.0, f"c.max()={c.max():.1f} above physical range"

    def test_density_in_physical_range(self, sample):
        rho = np.asarray(sample["density_voxel"]).astype(np.float32)
        assert np.all(np.isfinite(rho)), "density has NaN or inf"
        # Air 1.225, fat ~950, water 1000, bone ~1850. Generous slack.
        assert rho.min() >= 0.5, f"rho.min()={rho.min():.1f} below range"
        assert rho.max() <= 2500.0, f"rho.max()={rho.max():.1f} above range"

    def test_observed_data_finite_and_nonzero(self, sample):
        obs = np.asarray(sample["observed_data"])
        assert np.all(np.isfinite(obs)), "observed_data has NaN or inf"
        peak = float(np.max(np.abs(obs)))
        assert peak > 0.0, (
            "observed_data is identically zero — the forward sim returned "
            "no signal (likely a transducer geometry or source-signal bug)"
        )

    def test_observed_data_signal_to_noise(self, sample):
        """A well-formed forward record has a clear peak above the floor.

        If the field is dominated by numerical noise the peak/median ratio
        will be O(1); for a real wave field it should be > 100x.
        """
        obs = np.asarray(sample["observed_data"])
        # Use median |obs| over all (src, t, recv) entries as a noise floor
        # proxy; very robust to outliers.
        peak = float(np.max(np.abs(obs)))
        floor = float(np.median(np.abs(obs)))
        if floor <= 0:
            pytest.fail("observed_data median absolute value is zero")
        assert peak / floor > 100.0, (
            f"peak/median = {peak/floor:.1f} — observed_data may be all "
            f"numerical noise rather than a forward-sim record"
        )

    def test_tissue_labels_in_valid_range(self, sample):
        labels = np.asarray(sample["tissue_labels"])
        # uint8 already constrains to [0, 255]. MIDA goes to 116, BrainWeb
        # to ~11. Lesion injection uses BrainWeb's label 8. Anything > 116
        # is unexpected.
        assert int(labels.max()) <= 116, (
            f"tissue_labels max {int(labels.max())} > 116 — unknown label"
        )
        assert int(labels.min()) >= 0, (
            f"tissue_labels min {int(labels.min())} < 0"
        )


# ---------------------------------------------------------------------------
# SIREN decoding: Phase-2's whole job
# ---------------------------------------------------------------------------


class TestSirenContract:
    """If the SIREN can't reconstruct c, Phase-2 NPE has nothing to learn."""

    def test_siren_arch_has_required_keys(self, sample):
        from brain_fwi.inference.dataprep import _parse_arch
        arch = _parse_arch(sample["siren_arch"])
        for k in ("in_dim", "hidden_dim", "n_hidden", "out_dim", "omega_0"):
            assert k in arch, f"siren_arch missing {k!r}: {arch}"

    def test_siren_weights_bytes_nonempty(self, sample):
        w = np.asarray(sample["siren_weights_bytes"], dtype=np.uint8)
        assert w.size > 0, "siren_weights_bytes is empty"

    def test_siren_decodes_to_c_within_tolerance(self, sample):
        """End-to-end Phase-2 contract: round-trip SIREN → c-field.

        Allows up to 8% mean relative error — the production pipeline
        uses 400 SIREN pretrain steps and reports ~4%.
        """
        from brain_fwi.inference.dataprep import siren_from_sample
        from brain_fwi.inversion.param_field import SIRENField

        siren = siren_from_sample(sample)

        c_true = np.asarray(sample["sound_speed_voxel"]).astype(np.float32)
        # The dataprep loader returns the raw SIREN MLP. Production
        # wraps it in SIRENField to map MLP output -> [c_min, c_max] in
        # m/s (gen_phase0 uses 1400/3200). Phase-2 NPE works on raw
        # weights, but any consumer that wants a c-field reconstruction
        # has to do this wrap; if the wrap fails or the bounds are
        # wrong, training-time decoders silently produce garbage.
        field = SIRENField(siren=siren, grid_shape=tuple(c_true.shape))
        siren_v = np.asarray(field.to_velocity(1400.0, 3200.0))
        rel_err = float(
            np.mean(np.abs(siren_v - c_true)) / max(np.mean(c_true), 1e-9)
        )
        assert rel_err < 0.08, (
            f"SIREN reconstruction rel-err {rel_err:.2%} > 8% — Phase-2 "
            f"NPE will not have a usable c-field representation"
        )


# ---------------------------------------------------------------------------
# Cross-sample uniqueness: the manifest level
# ---------------------------------------------------------------------------


class TestManifestCrossSampleContract:
    """Catches the "all 1024 samples are byte-identical" failure mode."""

    def test_completed_ids_unique(self, manifest):
        ids = manifest["completed"]
        assert len(ids) == len(set(ids)), (
            f"manifest has duplicate sample ids — merge produced overlap"
        )

    def test_metadata_block_well_formed(self, manifest):
        md = manifest.get("metadata", {})
        for k in ("phantom", "grid_size", "dx", "freq_hz", "n_elements"):
            assert k in md, f"manifest.metadata missing {k!r}"
        assert md["grid_size"] > 0
        assert md["dx"] > 0
        assert md["freq_hz"] > 0


@pytest.mark.skipif(
    _find_shard_root() is None
    or len(json.loads(((_find_shard_root() or Path("/dev/null")) / "manifest.json").read_text()).get("completed", [])) < 2,
    reason="need >=2 samples in shard to test cross-sample diversity",
)
class TestCrossSampleDiversity:
    """Two consecutive samples must actually differ — both geometrically
    (caught the warp no-op) and acoustically (caught the property bug)."""

    @pytest.fixture(scope="class")
    def two_samples(self, shard_root):
        manifest = json.loads((shard_root / "manifest.json").read_text())
        ids = manifest["completed"][:2]
        return [_load_sample(shard_root, sid) for sid in ids]

    def test_warped_labels_differ(self, two_samples):
        """The warp no-op bug had this fail with 0 differing voxels."""
        a, b = two_samples
        n_diff = int((a["tissue_labels"] != b["tissue_labels"]).sum())
        assert n_diff > 0, (
            "two samples have byte-identical tissue_labels — deformation "
            "warp is a no-op (the v1b/v2-morning bug)"
        )

    def test_sound_speed_fields_differ(self, two_samples):
        a, b = two_samples
        n_diff = int((a["sound_speed_voxel"] != b["sound_speed_voxel"]).sum())
        # Even with the warp working, jitter alone makes ~50%+ of voxels
        # differ; the threshold is loose to tolerate small augment configs.
        assert n_diff > a["sound_speed_voxel"].size * 0.10, (
            f"only {n_diff} voxels differ between samples — augmentation "
            f"is producing near-identical data"
        )

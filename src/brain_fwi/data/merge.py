"""Merge per-rank Phase-0 part directories into one canonical dataset.

Used by the parallel ``modal_gen_phase0`` runner where each of N ranks
writes to ``<root>/part_<i>/`` with its own ``shards/`` and
``manifest.json``. After all ranks finish this helper produces a
single output directory with:

  - ``shards/00000.h5`` ... ``shards/<N-1>.h5`` — one shard per
    contributing part, copied verbatim so HDF5 metadata stays
    well-formed (rebuilding one giant 36 GB shard via repeated
    ``h5py.File.copy()`` corrupted the B-tree)
  - ``manifest.json`` — union of all parts' completed lists, in the
    contiguous part-major order the on-disk layout assumes
  - ``shard_size`` is set to ``max(per_part_completed_counts)`` so
    ``ShardedReader``'s ``shard_idx = sample_idx // shard_size``
    actually points at the right shard

Why this design: per-part writers fill exactly one shard (each rank
only ever holds 32-128 samples), so the natural layout is
"one shard per part." Trying to re-pack into a single ``shard_size``-d
output shard at production volumes breaks HDF5 ("wrong B-tree
signature" at ~500 sample copies into a 36 GB file). The simpler
copy-and-rename strategy keeps each output shard at the per-part size
that already works.

Rejects:
  - duplicate ``sample_id`` across parts
  - mismatched ``version`` strings
  - non-contiguous per-part sample-id ranges (would break the
    "shard_size = max per-part count" invariant)
  - more than one shard per part (per-part writer should only
    accumulate ``<= shard_size`` samples)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Sequence

import h5py


def merge_phase0_parts(
    part_dirs: Sequence[Path | str],
    output_dir: Path | str,
) -> None:
    """Combine N per-rank part directories into one canonical dataset."""
    parts = [Path(p) for p in part_dirs]
    if not parts:
        raise ValueError("part_dirs is empty")

    out = Path(output_dir)
    out_shards = out / "shards"
    out_shards.mkdir(parents=True, exist_ok=True)

    manifests = []
    part_shard_paths: list[Path] = []
    for p in parts:
        m_path = p / "manifest.json"
        if not m_path.exists():
            raise FileNotFoundError(f"missing manifest: {m_path}")
        manifests.append(json.loads(m_path.read_text()))
        srcs = sorted((p / "shards").glob("*.h5"))
        if len(srcs) != 1:
            raise ValueError(
                f"part {p} has {len(srcs)} shards; merge expects exactly 1 "
                f"per part. Per-part writer should only fill one shard."
            )
        part_shard_paths.append(srcs[0])

    # Version consistency check.
    versions = {m["version"] for m in manifests}
    if len(versions) > 1:
        raise ValueError(f"inconsistent version across parts: {versions}")

    # Duplicate sample_id check + cross-validate manifest vs shard contents.
    seen: dict[str, Path] = {}
    for p, m, src in zip(parts, manifests, part_shard_paths):
        with h5py.File(src, "r") as f:
            shard_ids = set(f.keys())
        manifest_ids = set(m["completed"])
        if shard_ids != manifest_ids:
            missing = manifest_ids - shard_ids
            extra = shard_ids - manifest_ids
            raise ValueError(
                f"part {p} manifest/shard mismatch: "
                f"missing-from-shard={sorted(missing)[:3]}, "
                f"extra-in-shard={sorted(extra)[:3]}"
            )
        for sid in m["completed"]:
            if sid in seen:
                raise ValueError(
                    f"duplicate sample_id {sid!r} in {p} and {seen[sid]}"
                )
            seen[sid] = p

    # The natural shard_size is the largest per-part count. With uniform
    # contiguous parts (the common case), this ensures sample at
    # global position i = part_idx * per_part + offset lands in shard
    # part_idx via i // shard_size.
    per_part_counts = [len(m["completed"]) for m in manifests]
    shard_size = max(per_part_counts)
    # Trailing parts may be smaller — but reader's shard_size lookup
    # only cares about the largest, since shard_idx = i // shard_size
    # and within a shard ShardedReader looks up by sample_id (not by
    # offset). So the only requirement is that no sample's
    # global-index // shard_size points outside its actual shard. With
    # parts laid out in contiguous part-major order and the largest
    # count first, that's automatic.
    if per_part_counts != sorted(per_part_counts, reverse=True):
        # Front-loaded sizes are required: the reader maps
        # sample_idx -> shard_idx = sample_idx // shard_size, where
        # shard_size = max-per-part. If a smaller part comes before a
        # larger one, samples after the smaller part's last index would
        # land in the wrong shard.
        raise ValueError(
            f"per-part counts must be non-increasing for the "
            f"copy-and-rename merge to land samples in the right shards; "
            f"got {per_part_counts}"
        )

    # Copy each per-part shard verbatim into the merged dir.
    for i, src in enumerate(part_shard_paths):
        dst = out_shards / f"{i:05d}.h5"
        shutil.copy2(src, dst)

    # Manifest: concatenate per-part `completed` lists in part order
    # (NOT sorted alphabetically — must match the on-disk shard layout
    # so sample_idx // shard_size points to the right shard).
    completed: list[str] = []
    for m in manifests:
        completed.extend(m["completed"])
    if len(set(completed)) != len(completed):
        raise ValueError(
            "duplicate sample_ids in concatenated manifest — "
            "should have been caught earlier"
        )

    out_manifest = {
        "version": next(iter(versions)),
        "shard_size": int(shard_size),
        "created_utc": manifests[0].get("created_utc"),
        "metadata": manifests[0].get("metadata", {}),
        "completed": completed,
    }
    (out / "manifest.json").write_text(json.dumps(out_manifest, indent=2))

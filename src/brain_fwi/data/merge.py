"""Merge per-rank Phase-0 part directories into one canonical dataset.

Used by the parallel ``modal_gen_phase0`` runner where each of N ranks
writes to ``<root>/part_<i>/`` with its own ``shards/`` and
``manifest.json``. After all ranks finish this helper produces a
single output directory with:

  - ``shards/00000.h5`` ... ``shards/<K>.h5`` — the merged samples
    re-packed into shards of size ``shard_size``, so that
    ``ShardedReader`` can locate sample ``i`` in shard ``i // shard_size``
  - ``manifest.json`` — union of all parts' completed lists, in the
    deterministic sorted order that the re-packing follows
  - ``metadata`` — taken from the first part (must agree across parts)

Why re-packing matters: per-part writers often only ever write to one
shard (one rank rarely accumulates ``shard_size`` samples), so a naive
shutil.copy2 of per-part shards produces a merged dir whose physical
shard layout disagrees with ``shard_size``. The reader then asks for
sample 94 in shard 0 (94 // 1000) but finds it in shard 1, raising
KeyError. Re-packing during merge keeps ``shard_size`` honest.

Rejects:
  - duplicate ``sample_id`` across parts (means overlapping aug ranges
    in the launcher — a real bug, not a silent dedup case)
  - mismatched ``version`` strings
  - mismatched ``shard_size`` across part manifests
"""

from __future__ import annotations

import json
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
    for p in parts:
        m_path = p / "manifest.json"
        if not m_path.exists():
            raise FileNotFoundError(f"missing manifest: {m_path}")
        manifests.append(json.loads(m_path.read_text()))

    # Version + shard_size consistency check.
    versions = {m["version"] for m in manifests}
    if len(versions) > 1:
        raise ValueError(f"inconsistent version across parts: {versions}")
    shard_sizes = {m["shard_size"] for m in manifests}
    if len(shard_sizes) > 1:
        raise ValueError(
            f"inconsistent shard_size across parts: {shard_sizes}"
        )
    shard_size = next(iter(shard_sizes))

    # Duplicate sample_id check + build sample_id -> source-shard map.
    seen: dict[str, Path] = {}
    sid_to_src: dict[str, Path] = {}
    for p in parts:
        src_shards = sorted((p / "shards").glob("*.h5"))
        for src in src_shards:
            with h5py.File(src, "r") as f:
                for sid in f.keys():
                    if sid in seen:
                        raise ValueError(
                            f"duplicate sample_id {sid!r} in {p} and {seen[sid]}"
                        )
                    seen[sid] = p
                    sid_to_src[sid] = src

    # Cross-check that every manifest entry was actually found on disk.
    manifest_ids: list[str] = []
    for m in manifests:
        manifest_ids.extend(m["completed"])
    missing_on_disk = [sid for sid in manifest_ids if sid not in sid_to_src]
    if missing_on_disk:
        raise ValueError(
            f"{len(missing_on_disk)} manifest sample(s) have no shard: "
            f"{missing_on_disk[:5]}{' ...' if len(missing_on_disk) > 5 else ''}"
        )

    # Sorted, deduped global order — must match the order ShardedReader
    # iterates so shard_idx = idx // shard_size lines up with the file.
    completed = sorted(set(manifest_ids))

    # Re-pack into shard_size-sized chunks. Cross-shard reads are
    # sequential within each output shard, so we open each source once
    # per output shard rather than once per sample.
    n_shards = (len(completed) + shard_size - 1) // shard_size
    for shard_idx in range(n_shards):
        chunk = completed[shard_idx * shard_size:(shard_idx + 1) * shard_size]
        dst = out_shards / f"{shard_idx:05d}.h5"
        # Group samples by their source file so we open each at most once.
        by_src: dict[Path, list[str]] = {}
        for sid in chunk:
            by_src.setdefault(sid_to_src[sid], []).append(sid)
        with h5py.File(dst, "w") as fout:
            for src, sids in by_src.items():
                with h5py.File(src, "r") as fin:
                    for sid in sids:
                        fin.copy(sid, fout)

    out_manifest = {
        "version": next(iter(versions)),
        "shard_size": shard_size,
        "created_utc": manifests[0].get("created_utc"),
        "metadata": manifests[0].get("metadata", {}),
        "completed": completed,
    }
    (out / "manifest.json").write_text(json.dumps(out_manifest, indent=2))

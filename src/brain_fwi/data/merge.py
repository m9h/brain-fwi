"""Merge per-rank Phase-0 part directories into one canonical dataset.

Used by the parallel ``modal_gen_phase0`` runner where each of N ranks
writes to ``<root>/part_<i>/`` with its own ``shards/`` and
``manifest.json``. After all ranks finish this helper produces a
single output directory with:

  - ``shards/00000.h5`` ... ``shards/<N>.h5`` — original shards copied
    over with non-colliding global indices
  - ``manifest.json`` — union of all parts' completed lists
  - ``metadata`` — taken from the first part (must agree across parts)

Rejects:
  - duplicate ``sample_id`` across parts (means overlapping aug ranges
    in the launcher — a real bug, not a silent dedup case)
  - mismatched ``version`` strings
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Sequence


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

    # Duplicate sample_id check.
    seen: dict[str, Path] = {}
    for p, m in zip(parts, manifests):
        for sid in m["completed"]:
            if sid in seen:
                raise ValueError(
                    f"duplicate sample_id {sid!r} in {p} and {seen[sid]}"
                )
            seen[sid] = p

    # Copy shards with renumbered global indices.
    next_idx = 0
    for p in parts:
        src_shards = sorted((p / "shards").glob("*.h5"))
        for src in src_shards:
            dst = out_shards / f"{next_idx:05d}.h5"
            shutil.copy2(src, dst)
            next_idx += 1

    # Union manifest.
    completed = []
    for m in manifests:
        completed.extend(m["completed"])
    completed = sorted(set(completed))  # dedupe in case of resumes within a part

    out_manifest = {
        "version": next(iter(versions)),
        "shard_size": next(iter(shard_sizes)),
        "created_utc": manifests[0].get("created_utc"),
        "metadata": manifests[0].get("metadata", {}),
        "completed": completed,
    }
    (out / "manifest.json").write_text(json.dumps(out_manifest, indent=2))

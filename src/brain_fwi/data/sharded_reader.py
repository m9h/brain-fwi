"""Read-side counterpart to :class:`ShardedWriter`.

Iterates or randomly-accesses Phase-0 dataset samples produced by the
writer. Memory-cheap options:

  - ``fields=[...]`` — only load the listed HDF5 arrays/attrs per sample.
  - Random access is O(1): the manifest gives a stable sample-id →
    (shard_idx, within_shard_idx) map on construction.
  - Iteration order matches ``manifest.completed`` order exactly, so
    deterministic shuffling / train-test splits live in the caller.

Consumers that want batching can compose :class:`ShardedReader` with
``jax.tree.map`` or their framework's DataLoader-equivalent; this
module stays framework-free on purpose.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union

import h5py
import numpy as np


class ShardedReader:
    """Random-access / iterable reader for Phase-0 datasets."""

    def __init__(
        self,
        root: Union[str, os.PathLike],
        fields: Optional[Sequence[str]] = None,
        expected_version: Optional[str] = None,
    ):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found at {manifest_path}")

        self._manifest: Dict[str, Any] = json.loads(manifest_path.read_text())
        if expected_version is not None and self._manifest.get("version") != expected_version:
            raise ValueError(
                f"manifest version {self._manifest.get('version')!r} "
                f"!= expected {expected_version!r}"
            )

        self._shard_size = int(self._manifest["shard_size"])
        self._sample_ids: List[str] = list(self._manifest["completed"])
        self._id_to_index = {sid: i for i, sid in enumerate(self._sample_ids)}

        # Field filter: None → all, [] → sample_id only.
        self._fields: Optional[List[str]] = (
            None if fields is None else list(fields)
        )

    # -- metadata --------------------------------------------------------

    @property
    def metadata(self) -> Mapping[str, Any]:
        return dict(self._manifest)

    @property
    def sample_ids(self) -> List[str]:
        return list(self._sample_ids)

    def __len__(self) -> int:
        return len(self._sample_ids)

    # -- access ----------------------------------------------------------

    def __getitem__(self, key: Union[int, str]) -> Dict[str, Any]:
        if isinstance(key, str):
            if key not in self._id_to_index:
                raise KeyError(f"sample {key!r} not in manifest")
            idx = self._id_to_index[key]
        elif isinstance(key, (int, np.integer)):
            idx = int(key)
            if idx < 0:
                idx += len(self._sample_ids)
            if idx < 0 or idx >= len(self._sample_ids):
                raise IndexError(
                    f"index {key} out of range for {len(self._sample_ids)} samples"
                )
        else:
            raise TypeError(f"key must be int or str, got {type(key).__name__}")

        return self._load(idx)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for idx in range(len(self._sample_ids)):
            yield self._load(idx)

    # -- internals -------------------------------------------------------

    def _load(self, idx: int) -> Dict[str, Any]:
        sample_id = self._sample_ids[idx]
        shard_idx = idx // self._shard_size
        shard_path = self.root / "shards" / f"{shard_idx:05d}.h5"

        out: Dict[str, Any] = {"sample_id": sample_id}

        with h5py.File(str(shard_path), "r") as f:
            grp = f[sample_id]

            if self._fields is None:
                # All fields: attrs + datasets.
                for k, v in grp.attrs.items():
                    out[k] = v
                for k in grp.keys():
                    out[k] = np.asarray(grp[k])
            else:
                # Filtered: only requested fields, plus sample_id.
                for key in self._fields:
                    if key == "sample_id":
                        continue
                    if key in grp.attrs:
                        out[key] = grp.attrs[key]
                    elif key in grp:
                        out[key] = np.asarray(grp[key])
                    else:
                        # Silent-skip missing fields so callers can declare
                        # a superset schema across dataset versions.
                        continue

        return out

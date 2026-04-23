"""Resumable sharded HDF5 writer for the Phase-0 training dataset.

Layout::

    root/
      manifest.json              # completed sample_ids + dataset metadata
      shards/
        00000.h5                 # samples 0 .. shard_size-1
        00001.h5                 # samples shard_size .. 2*shard_size-1
        ...

Each HDF5 file contains one top-level group per sample, keyed by
``sample_id``. Sample payload keys and dtypes are whatever the caller
hands to :meth:`ShardedWriter.write`. Arrays go under the group, scalars
and strings into the group ``attrs``. Dataset-level metadata lives in
``manifest.json``.

The writer is resumable: if ``manifest.json`` already exists at
construction time, its ``completed`` list is read and
:meth:`is_complete` returns True for those sample_ids. Callers should
skip resimulation for completed samples.

HDF5 rather than Zarr here because h5py is already a core dep and Phase-0
I/O is not on the critical path. Migration to Zarr (see
``docs/design/data_pipeline.md``) is planned once storage becomes the
dominant cost.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import h5py
import numpy as np


class ShardedWriter:
    """Append-only sharded dataset writer with manifest-backed resume."""

    def __init__(
        self,
        root: os.PathLike,
        shard_size: int = 1000,
        version: str = "phase0_v1",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.root = Path(root)
        self.shard_size = int(shard_size)
        self.version = version
        self.shards_dir = self.root / "shards"
        self.manifest_path = self.root / "manifest.json"

        self.shards_dir.mkdir(parents=True, exist_ok=True)

        if self.manifest_path.exists():
            self._manifest = json.loads(self.manifest_path.read_text())
            if self._manifest.get("version") != version:
                raise ValueError(
                    f"manifest version {self._manifest.get('version')!r} "
                    f"!= requested {version!r} at {self.manifest_path}"
                )
            if self._manifest.get("shard_size") != self.shard_size:
                raise ValueError(
                    f"manifest shard_size {self._manifest.get('shard_size')} "
                    f"!= requested {self.shard_size}"
                )
        else:
            self._manifest = {
                "version": version,
                "shard_size": self.shard_size,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "metadata": metadata or {},
                "completed": [],
            }
            self._flush_manifest()

        self._completed = set(self._manifest["completed"])

    @property
    def n_completed(self) -> int:
        return len(self._completed)

    def is_complete(self, sample_id: str) -> bool:
        return sample_id in self._completed

    def completed_ids(self) -> Iterable[str]:
        return iter(self._completed)

    def write(self, sample: Dict[str, Any]) -> None:
        """Write one sample. Must contain a ``sample_id`` string key.

        Array values go into HDF5 datasets, scalar/string values become
        group attributes. Dict values are JSON-encoded into an attribute.
        """
        if "sample_id" not in sample:
            raise KeyError("sample dict must include 'sample_id'")
        sample_id = str(sample["sample_id"])
        if sample_id in self._completed:
            return  # idempotent

        shard_idx = self.n_completed // self.shard_size
        shard_path = self.shards_dir / f"{shard_idx:05d}.h5"

        with h5py.File(shard_path, "a") as f:
            if sample_id in f:
                # partial write from a crashed previous run — overwrite
                del f[sample_id]
            grp = f.create_group(sample_id)
            for key, value in sample.items():
                if key == "sample_id":
                    grp.attrs[key] = sample_id
                    continue
                if isinstance(value, (str, bytes)):
                    grp.attrs[key] = value
                elif isinstance(value, (int, float, bool, np.integer, np.floating)):
                    grp.attrs[key] = value
                elif isinstance(value, dict):
                    grp.attrs[key] = json.dumps(value)
                elif isinstance(value, np.ndarray):
                    _write_array(grp, key, value)
                else:
                    arr = np.asarray(value)
                    _write_array(grp, key, arr)

        self._completed.add(sample_id)
        self._manifest["completed"].append(sample_id)
        self._flush_manifest()

    def close(self) -> None:
        self._flush_manifest()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _flush_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._manifest, indent=2, sort_keys=False))
        tmp.replace(self.manifest_path)


def _write_array(group: h5py.Group, key: str, arr: np.ndarray) -> None:
    """Write an array with chunking + gzip for anything larger than a few KB."""
    if arr.size <= 1024:
        group.create_dataset(key, data=arr)
    else:
        group.create_dataset(
            key,
            data=arr,
            chunks=True,
            compression="gzip",
            compression_opts=4,
        )


def load_sample(root: os.PathLike, sample_id: str) -> Dict[str, Any]:
    """Read one sample back by ``sample_id``. Scans manifest for shard index."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    shard_size = manifest["shard_size"]
    try:
        idx = manifest["completed"].index(sample_id)
    except ValueError:
        raise KeyError(f"sample {sample_id!r} not in manifest")
    shard_idx = idx // shard_size

    shard_path = root / "shards" / f"{shard_idx:05d}.h5"
    out: Dict[str, Any] = {}
    with h5py.File(shard_path, "r") as f:
        grp = f[sample_id]
        for k, v in grp.attrs.items():
            out[k] = v
        for k in grp.keys():
            out[k] = np.asarray(grp[k])
    return out

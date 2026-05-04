"""Modal runner: parallel Phase-0 dataset generation across N A100s.

Splits the global ``aug_idx`` range into N contiguous slices, runs one
Modal A100 per slice via ``function.map()``, then merges the per-rank
output dirs into one canonical dataset via ``brain_fwi.data.merge``.

Why parallel:
    Each sample is independent (same MIDA phantom, different aug_idx
    seed). Per-sample wall on A100 is ~4 min (vs 6.85 min on the GB10).
    With 8 ranks → 1024 samples in ~1.4 h instead of ~64 h sequential.

One-time setup (run from a machine with the modal CLI configured)::

    modal volume create brain-fwi-phase0
    modal run scripts/modal_gen_phase0.py --n-augments 1024 --n-ranks 8

Resume after the existing 465 samples on DGX (the typical use):

    1. Upload the existing manifest+shards as part_dgx/ to the work volume::

        modal volume put brain-fwi-phase0 \\
            /data/datasets/brain-fwi/phase0_v1b_mida_96 \\
            /output/phase0_v1b_mida_96/part_dgx

    2. Run with --aug-start=465 --aug-end=1024 (skip the already-done range)::

        modal run scripts/modal_gen_phase0.py \\
            --n-augments 1024 --n-ranks 8 \\
            --aug-start 465 --version phase0_v1b \\
            --siren-hidden 64 --siren-layers 3

    3. Pull back the merged result::

        modal volume get brain-fwi-phase0 \\
            /output/phase0_v1b_mida_96/merged ./
"""

from __future__ import annotations

import time

import modal

app = modal.App("brain-fwi-gen-phase0")

# Defaults match docs/design/data_pipeline.md §5 and the slurm runner.
PHANTOM = "mida"
GRID_SIZE = 96
DX_M = 0.002
N_ELEMENTS = 128
N_SUBJECTS = 1
N_AUGMENTS = 100
FREQ_HZ = 5.0e5
SIREN_HIDDEN = 128
SIREN_LAYERS = 3
SIREN_PRETRAIN_STEPS = 400
DATASET_VERSION = "phase0_v1"

GIT_BRANCH = "feature/parallel-modal-phase0"
CACHE_BUST = "2026-05-04-phase0-modal-parallel-v2-commit-loop"

MIDA_VOL_NAME = "mida-data"
WORK_VOL_NAME = "brain-fwi-phase0"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "unzip")
    .pip_install("uv")
    .env({"BRAIN_FWI_CACHE_BUST": CACHE_BUST})
    .run_commands(
        f"git clone --depth 1 --branch {GIT_BRANCH} "
        f"https://github.com/m9h/brain-fwi.git /opt/brain-fwi",
        "cd /opt/brain-fwi && uv pip install --system -e '.[cuda12]'",
    )
)

mida_vol = modal.Volume.from_name(MIDA_VOL_NAME, create_if_missing=False)
work_vol = modal.Volume.from_name(WORK_VOL_NAME, create_if_missing=True)


def _ensure_mida_path() -> str:
    """Resolve the MIDA NIfTI path on the mounted volume, unzipping if needed."""
    import glob
    import subprocess
    import sys

    candidates = (
        glob.glob("/mida/MIDA_v1.0/MIDA_v1_voxels/MIDA_v1.nii*")
        + glob.glob("/mida/extracted/**/MIDA_v1.nii*", recursive=True)
        + glob.glob("/mida/**/MIDA_v1.nii*", recursive=True)
    )
    if not candidates:
        zips = glob.glob("/mida/*.zip")
        if zips:
            print(f"Unzipping {zips[0]} ...")
            subprocess.run(
                ["unzip", "-qo", zips[0], "-d", "/mida/extracted"],
                check=True,
            )
            candidates = glob.glob(
                "/mida/extracted/**/MIDA_v1.nii*", recursive=True,
            )
    if not candidates:
        sys.exit("MIDA NIfTI not found on /mida volume.")
    return candidates[0]


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    memory=32 * 1024,
    volumes={"/mida": mida_vol, "/work": work_vol},
)
def generate_range(
    rank: int,
    aug_start: int,
    aug_end: int,
    *,
    phantom: str = PHANTOM,
    grid_size: int = GRID_SIZE,
    dx: float = DX_M,
    n_elements: int = N_ELEMENTS,
    n_subjects: int = N_SUBJECTS,
    n_augments: int = N_AUGMENTS,
    freq_hz: float = FREQ_HZ,
    siren_hidden: int = SIREN_HIDDEN,
    siren_layers: int = SIREN_LAYERS,
    siren_pretrain_steps: int = SIREN_PRETRAIN_STEPS,
    version: str = DATASET_VERSION,
):
    """One rank: produce samples in ``[aug_start, aug_end)`` into part_<rank>/."""
    import os
    import subprocess

    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

    base = f"/work/output/{version}_{phantom}_{grid_size}"
    part_dir = f"{base}/part_{rank:02d}"
    os.makedirs(part_dir, exist_ok=True)
    print(f"[rank {rank}] aug=[{aug_start}, {aug_end}) → {part_dir}")

    args = [
        "python", "-u", "/opt/brain-fwi/scripts/gen_phase0.py",
        "--out", part_dir,
        "--phantom", phantom,
        "--grid-size", str(grid_size),
        "--dx", str(dx),
        "--n-elements", str(n_elements),
        "--n-subjects", str(n_subjects),
        "--n-augments", str(n_augments),
        "--aug-start", str(aug_start),
        "--aug-end", str(aug_end),
        "--freq", str(freq_hz),
        "--siren-hidden", str(siren_hidden),
        "--siren-layers", str(siren_layers),
        "--siren-pretrain-steps", str(siren_pretrain_steps),
        "--version", version,
    ]
    if phantom == "mida":
        args += ["--mida-path", _ensure_mida_path()]

    # Run gen_phase0 as a non-blocking subprocess so we can periodically
    # commit the Modal volume. Without this, a mid-run failure (preemption,
    # OOM, network blip, container kill) loses every sample produced
    # since the rank started — the manifest + shards live in container-
    # local storage until commit() runs.
    import threading
    t0 = time.time()
    proc = subprocess.Popen(args, cwd="/opt/brain-fwi")

    stop_event = threading.Event()
    def _commit_loop():
        while not stop_event.wait(timeout=300):  # every 5 min
            try:
                print(f"[rank {rank}] periodic volume commit...", flush=True)
                work_vol.commit()
            except Exception as e:
                print(f"[rank {rank}] periodic commit failed: {e}", flush=True)
    t = threading.Thread(target=_commit_loop, daemon=True)
    t.start()

    ret = proc.wait()
    stop_event.set()
    t.join(timeout=10)
    if ret != 0:
        # Final commit even on failure so partial work is durable.
        try:
            work_vol.commit()
        except Exception:
            pass
        raise RuntimeError(f"gen_phase0.py exited with code {ret}")

    wallclock = time.time() - t0
    work_vol.commit()

    import json
    with open(f"{part_dir}/manifest.json") as f:
        manifest = json.load(f)
    n_done = len(manifest.get("completed", []))

    print(f"[rank {rank}] {n_done} samples in {wallclock/60:.1f} min")
    return {"rank": rank, "part_dir": part_dir, "n_samples": n_done,
            "wallclock_s": wallclock}


@app.function(
    image=image,
    timeout=60 * 60,
    memory=8 * 1024,
    volumes={"/work": work_vol},
)
def merge_parts(part_dirs: list, output_dir: str):
    """Concat per-rank dirs into one canonical dataset on the volume."""
    from brain_fwi.data.merge import merge_phase0_parts

    print(f"merging {len(part_dirs)} parts → {output_dir}")
    merge_phase0_parts(part_dirs, output_dir)
    work_vol.commit()

    import json
    with open(f"{output_dir}/manifest.json") as f:
        manifest = json.load(f)
    n_done = len(manifest["completed"])
    print(f"merged dataset: {n_done} samples at {output_dir}")
    return {"output_dir": output_dir, "n_samples": n_done}


@app.local_entrypoint()
def main(
    phantom: str = PHANTOM,
    grid_size: int = GRID_SIZE,
    n_subjects: int = N_SUBJECTS,
    n_augments: int = N_AUGMENTS,
    aug_start: int = 0,
    aug_end: int = -1,                  # -1 = n_augments
    n_ranks: int = 8,
    siren_hidden: int = SIREN_HIDDEN,
    siren_layers: int = SIREN_LAYERS,
    version: str = DATASET_VERSION,
    existing_parts: str = "",           # comma-sep names already on the volume
):
    if aug_end < 0:
        aug_end = n_augments
    if not (0 <= aug_start < aug_end <= n_augments):
        raise SystemExit(
            f"invalid aug range [{aug_start}, {aug_end}) for "
            f"n_augments={n_augments}"
        )

    total = aug_end - aug_start
    if n_ranks <= 0 or n_ranks > total:
        n_ranks = max(1, min(n_ranks, total))

    # Contiguous slicing — leftover samples spread across the early ranks.
    base = total // n_ranks
    extra = total % n_ranks
    rank_args = []
    cursor = aug_start
    for rank in range(n_ranks):
        slice_size = base + (1 if rank < extra else 0)
        rank_args.append((rank, cursor, cursor + slice_size))
        cursor += slice_size

    print("=" * 70)
    print(f"  Phase-0 parallel gen on {n_ranks}× A100-80GB")
    print(f"  Phantom: {phantom}, grid: {grid_size}^3")
    print(f"  Aug range: [{aug_start}, {aug_end}) = {total} samples")
    print(f"  SIREN: hidden={siren_hidden}, layers={siren_layers}")
    for rank, s, e in rank_args:
        print(f"    rank {rank}: aug=[{s}, {e}) ({e-s} samples)")
    print("=" * 70)

    t0 = time.time()
    results = list(
        generate_range.starmap(
            rank_args,
            kwargs=dict(
                phantom=phantom,
                grid_size=grid_size,
                n_subjects=n_subjects,
                n_augments=n_augments,
                siren_hidden=siren_hidden,
                siren_layers=siren_layers,
                version=version,
            ),
        )
    )
    gen_wall = time.time() - t0
    total_samples = sum(r["n_samples"] for r in results)
    print(f"\nAll ranks done. {total_samples} samples in {gen_wall/60:.1f} min wall")

    base_dir = f"/work/output/{version}_{phantom}_{grid_size}"
    part_dirs = [r["part_dir"] for r in results]
    if existing_parts:
        for name in existing_parts.split(","):
            name = name.strip()
            if name:
                part_dirs.append(f"{base_dir}/{name}")
    merge_result = merge_parts.remote(
        part_dirs=part_dirs,
        output_dir=f"{base_dir}/merged",
    )

    print(f"\nFinal dataset: {merge_result['n_samples']} samples")
    print(f"Pull back with:")
    print(f"  modal volume get {WORK_VOL_NAME} {merge_result['output_dir']} ./")

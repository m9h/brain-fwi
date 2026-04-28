"""Modal runner: Phase-0 dataset generation on A100.

Outsources Phase-0 (c, d) pair generation off DGX so the local GPU
stays free for fMRI / sMRI work. Produces the dataset the Phase-2 NPE
training (`scripts/train_npe_on_phase0.py`) and Phase-4 FNO surrogate
training need to train on real data instead of toys — directly
unblocks the §7.4 surrogate-vs-jWave NPE-equivalence gate that the
stepback identified as the only Phase-4 metric that scientifically
matters.

Why Modal here:
    - Generation is embarrassingly parallel across (subject × aug)
      indices but our current driver (`scripts/gen_phase0.py`) writes
      one shard at a time. A single Modal A100 doing 1 subject × 100
      augs of MIDA at 96^3 should take ~8 h based on the design-doc
      compute budget (~100 s/sample at 96^3).
    - `ShardedWriter`'s manifest-based resume means we can stop a run
      and pick up where we left off — important for a long Modal job
      that might bump the function timeout.
    - Output volume can be downloaded back to DGX for Phase-2 NPE
      training, or consumed straight on Modal by a follow-up runner.

One-time setup (run from a machine with the modal CLI configured)::

    # MIDA NIfTI volume (already exists from modal_mida_256.py setup;
    # if not:)
    #   modal volume create mida-data
    #   modal volume put mida-data /path/to/MIDAv1-0.zip /MIDAv1-0.zip

    modal volume create brain-fwi-phase0
    modal run scripts/modal_gen_phase0.py

Pull the shards back to DGX::

    modal volume get brain-fwi-phase0 /output/phase0_v1_mida_96 ./
"""

from __future__ import annotations

import time

import modal

app = modal.App("brain-fwi-gen-phase0")

# Defaults match docs/design/data_pipeline.md §5 and the existing
# slurm_gen_phase0.sh. Override via the local entrypoint args.
PHANTOM = "mida"
GRID_SIZE = 96
DX_M = 0.002
N_ELEMENTS = 128
N_SUBJECTS = 1
N_AUGMENTS = 100
FREQ_HZ = 5.0e5
SIREN_PRETRAIN_STEPS = 400
DATASET_VERSION = "phase0_v1"

GIT_BRANCH = "main"
CACHE_BUST = "2026-04-25-phase0-modal-v1"

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


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,    # 24 h — Modal max per-fn budget. Use the RunPod variant if you need longer.
    memory=32 * 1024,
    volumes={
        "/mida": mida_vol,
        "/work": work_vol,
    },
)
def generate(
    phantom: str = PHANTOM,
    grid_size: int = GRID_SIZE,
    dx: float = DX_M,
    n_elements: int = N_ELEMENTS,
    n_subjects: int = N_SUBJECTS,
    n_augments: int = N_AUGMENTS,
    freq_hz: float = FREQ_HZ,
    siren_pretrain_steps: int = SIREN_PRETRAIN_STEPS,
    version: str = DATASET_VERSION,
):
    import glob
    import os
    import subprocess
    import sys

    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

    out_dir = f"/work/output/{version}_{phantom}_{grid_size}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    args = [
        "python", "-u", "/opt/brain-fwi/scripts/gen_phase0.py",
        "--out", out_dir,
        "--phantom", phantom,
        "--grid-size", str(grid_size),
        "--dx", str(dx),
        "--n-elements", str(n_elements),
        "--n-subjects", str(n_subjects),
        "--n-augments", str(n_augments),
        "--freq", str(freq_hz),
        "--siren-pretrain-steps", str(siren_pretrain_steps),
        "--version", version,
    ]

    # MIDA path resolution mirrors modal_mida_192_voxel_fwi.py
    if phantom == "mida":
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
        args += ["--mida-path", candidates[0]]
        print(f"Using MIDA: {candidates[0]}")

    # ShardedWriter's manifest resume kicks in automatically if a previous
    # run wrote partial shards to the same out_dir.
    manifest_path = f"{out_dir}/manifest.json"
    if os.path.exists(manifest_path):
        import json
        with open(manifest_path) as f:
            manifest = json.load(f)
        already = len(manifest.get("completed", []))
        print(f"Resume: {already} sample(s) already in {out_dir}; "
              f"target {n_subjects * n_augments} total.")

    print(f"\nLaunching: {' '.join(args)}\n")
    t0 = time.time()
    subprocess.run(args, check=True, cwd="/opt/brain-fwi")
    wallclock = time.time() - t0

    # Persist outputs
    work_vol.commit()

    # Final manifest stats
    with open(manifest_path) as f:
        import json
        manifest = json.load(f)
    n_done = len(manifest.get("completed", []))
    print(f"\nWall time: {wallclock/60:.1f} min")
    print(f"Samples in manifest: {n_done}")
    print(f"Output volume path:  {out_dir}")
    return {"out_dir": out_dir, "n_samples": n_done, "wallclock_s": wallclock}


@app.local_entrypoint()
def main(
    phantom: str = PHANTOM,
    grid_size: int = GRID_SIZE,
    n_subjects: int = N_SUBJECTS,
    n_augments: int = N_AUGMENTS,
    version: str = DATASET_VERSION,
):
    print("=" * 64)
    print(f"  Phase-0 dataset gen on Modal A100-80GB")
    print(f"  Phantom: {phantom}, grid: {grid_size}^3, "
          f"target: {n_subjects} x {n_augments} = {n_subjects * n_augments}")
    print("=" * 64)
    result = generate.remote(
        phantom=phantom,
        grid_size=grid_size,
        n_subjects=n_subjects,
        n_augments=n_augments,
        version=version,
    )
    print(f"\nGenerated {result['n_samples']} samples in {result['wallclock_s']/60:.1f} min")
    print(f"Pull back with:")
    print(f"  modal volume get {WORK_VOL_NAME} {result['out_dir']} ./")

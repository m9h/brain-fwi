"""Modal runner: MIDA 192^3 voxel FWI with checkpoint resume.

Outsources the production MIDA 192^3 reconstruction off the DGX Spark
so the local GPU stays available for fMRI / sMRI work. Resumes from a
Band-1 checkpoint produced by an earlier on-cluster run (job 959 today,
2026-04-25) so the ~10 h of Band-1 forward simulation + FWI is not
re-spent on Modal.

Why Modal not RunPod for this run:
    - Production j-Wave at 192^3 is memory-bandwidth-bound (Phase 4
      readiness benchmark, PR #16). A100-80GB is the right tier; H100
      gives a marginal gain at 2x cost.
    - Modal's volume model fits the resume-from-checkpoint workflow:
      one volume for inputs (MIDA NIfTI + Band-1 ckpt), one for outputs
      (final h5 + figures). No object-storage round-trip.
    - We already have working Modal infra (modal_validate_siren.py,
      modal_extended_fwi.py); copying the pattern keeps reviewer
      surface area small.

Wall budget: bands 2 + 3 at 30 iters each ~= 5 h on A100 (band 1 took
~5 h on DGX); plus the script re-runs the ~42 min forward simulation
on entry. Total ~6 h actual, so the 24 h Modal timeout below has 4x
margin. If a future config bumps total runtime past 24 h, fall back
to the RunPod pattern in `scripts/run_runpod_siren_validation.py` —
RunPod pods have no per-function timeout.

One-time setup (run from a machine with the modal CLI configured)::

    # 1. Mida NIfTI — already on the mida-data volume from earlier runs
    #    (modal_mida_256.py uploaded it). If for some reason it's not
    #    there:
    #      modal volume create mida-data
    #      modal volume put mida-data <local>/MIDAv1-0.zip /MIDAv1-0.zip

    # 2. Band-1 checkpoint produced by on-cluster job 959. From DGX:
    #      rsync the file to your Modal-CLI machine first, then:
    modal volume create brain-fwi-mida-192-resume
    modal volume put brain-fwi-mida-192-resume \\
        /data/datasets/brain-fwi/checkpoints/192_mida_voxel/fwi_checkpoint.h5 \\
        /checkpoints/192_mida_voxel/fwi_checkpoint.h5

Run::

    modal run scripts/modal_mida_192_voxel_fwi.py

Result lands on the ``brain-fwi-mida-192-resume`` volume at
``/output/brain_usct_192_mida_voxel_modal.h5`` and ``.png``. Pull
back via::

    modal volume get brain-fwi-mida-192-resume \\
        /output/brain_usct_192_mida_voxel_modal.h5 ./
"""

from __future__ import annotations

import time

import modal

app = modal.App("brain-fwi-mida-192-voxel")

# --- knobs --------------------------------------------------------------
GRID_SIZE = 192
DX_M = 0.001
N_ELEMENTS = 256
ITERS_PER_BAND = 30
SHOTS_PER_ITER = 8
GIT_BRANCH = "main"
CACHE_BUST = "2026-04-25-mida-192-resume-v1"

# Volumes
MIDA_VOL_NAME = "mida-data"
WORK_VOL_NAME = "brain-fwi-mida-192-resume"

# Container image — same recipe used by modal_validate_siren.py +
# explicit nibabel + h5py + scipy for MIDA NIfTI loading and
# checkpoint serialisation.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
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
    timeout=24 * 60 * 60,                # 24h — Modal max per-fn budget for bands 2+3
    memory=32 * 1024,                    # 32 GiB RAM
    volumes={
        "/mida": mida_vol,
        "/work": work_vol,
    },
)
def run_fwi():
    import os
    import subprocess
    import sys

    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

    # Resolve MIDA NIfTI path on the volume. modal_mida_256.py either
    # placed it directly or as a zip under /mida — try both.
    import glob
    mida_candidates = (
        glob.glob("/mida/MIDA_v1.0/MIDA_v1_voxels/MIDA_v1.nii*")
        + glob.glob("/mida/extracted/**/MIDA_v1.nii*", recursive=True)
        + glob.glob("/mida/**/MIDA_v1.nii*", recursive=True)
    )
    if not mida_candidates:
        # Try unzipping if a zip is on the volume
        zips = glob.glob("/mida/*.zip")
        if zips:
            print(f"Unzipping {zips[0]} ...")
            subprocess.run(
                ["unzip", "-qo", zips[0], "-d", "/mida/extracted"],
                check=True,
            )
            mida_candidates = glob.glob(
                "/mida/extracted/**/MIDA_v1.nii*", recursive=True,
            )
    if not mida_candidates:
        listing = subprocess.run(
            ["find", "/mida", "-maxdepth", "4"], capture_output=True, text=True,
        )
        sys.exit(
            "Could not find MIDA_v1.nii on /mida volume.\n"
            f"Volume listing:\n{listing.stdout}"
        )
    mida_path = mida_candidates[0]
    print(f"Using MIDA: {mida_path}")

    # Verify the resume checkpoint is in place
    ckpt_dir = "/work/checkpoints/192_mida_voxel"
    ckpt_file = f"{ckpt_dir}/fwi_checkpoint.h5"
    if not os.path.exists(ckpt_file):
        sys.exit(
            f"No checkpoint at {ckpt_file}. "
            "Upload via: modal volume put brain-fwi-mida-192-resume "
            "/data/datasets/brain-fwi/checkpoints/192_mida_voxel/"
            "fwi_checkpoint.h5 /checkpoints/192_mida_voxel/fwi_checkpoint.h5"
        )
    sz_mb = os.path.getsize(ckpt_file) / 1e6
    print(f"Resume checkpoint: {ckpt_file} ({sz_mb:.1f} MB)")

    out_dir = "/work/output"
    os.makedirs(out_dir, exist_ok=True)
    out_h5 = f"{out_dir}/brain_usct_192_mida_voxel_modal.h5"
    out_png = f"{out_dir}/brain_usct_192_mida_voxel_modal.png"

    args = [
        "python", "-u", "/opt/brain-fwi/run_full_usct.py",
        "--grid-size", str(GRID_SIZE),
        "--dx", str(DX_M),
        "--n-elements", str(N_ELEMENTS),
        "--iters", str(ITERS_PER_BAND),
        "--shots", str(SHOTS_PER_ITER),
        "--phantom", "mida",
        "--mida-path", mida_path,
        "--parameterization", "voxel",
        "--checkpoint-dir", ckpt_dir,
        "--output", out_h5,
        "--figures", out_png,
    ]
    print(f"Launching: {' '.join(args)}")

    t0 = time.time()
    subprocess.run(args, check=True, cwd="/opt/brain-fwi")
    wallclock = time.time() - t0

    # Persist outputs to the volume so the local-side `modal volume get`
    # finds them.
    work_vol.commit()
    print(f"\nWall time: {wallclock/60:.1f} min")
    print(f"Output H5:  {out_h5}")
    print(f"Output PNG: {out_png}")
    return out_h5, out_png, wallclock


@app.local_entrypoint()
def main():
    print("=" * 64)
    print("  MIDA 192^3 voxel FWI on Modal A100-80GB (resume from Band 1)")
    print("=" * 64)
    out_h5, out_png, secs = run_fwi.remote()
    print(f"\nDone in {secs/60:.1f} min")
    print(f"Pull results back with:")
    print(f"  modal volume get {WORK_VOL_NAME} {out_h5} ./")
    print(f"  modal volume get {WORK_VOL_NAME} {out_png} ./")

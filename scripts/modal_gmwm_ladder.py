"""Modal runner for the 3-D GM/WM frequency ladder (`scripts/gmwm_velocity_3d.py`).

Why Modal for stage 3 (and beam for stage 2): stage 3 (500 kHz, N=176) peaks ~55 GB
(stage 2 at N=144 needed the 32 GB RTX5090; peak scales ~N^3.5), so it needs an 80 GB
card. beam's 80 GB cards are on-demand only and require $25 of separate "managed compute"
credits; Modal has A100-80GB serverless with a 24 h per-function cap, and stage 3 is
~1 h (stage 2 took 22.6 min on an RTX5090). Pattern copied from modal_mida_192_voxel_fwi.py
(cuda13 = the stack proven on Modal; cuda12 faulted XID-31 at 192^3; cuda_malloc_async).

Outputs land in results/gmwm_velocity_3d/ under the runner's own names
(summary_<TAG>.json, c_rec_<TAG>.npy, gmwm3d_<TAG>.png) + beam-style launch log, and are
also committed to Modal volume `brain-fwi-results` at /gmwm_velocity_3d/ (pull back with
`modal volume get brain-fwi-results /gmwm_velocity_3d/<file> ./`).

    .venv/bin/modal run scripts/modal_gmwm_ladder.py --stage 3
    .venv/bin/modal run scripts/modal_gmwm_ladder.py --stage 3 --iters 40 --shots-per-iter 8 --tag it40s8
    .venv/bin/modal run --detach scripts/modal_gmwm_ladder.py --stage 3     # survives local exit
"""
from __future__ import annotations
import os
import modal

app = modal.App("brain-fwi-gmwm-ladder")

GIT_BRANCH = os.environ.get("BFWI_MODAL_BRANCH", "feature/diffusion-prior-fwi")
GPU = os.environ.get("BFWI_MODAL_GPU", "A100-80GB")
# bump (or set BFWI_MODAL_CACHE_BUST) after every push: Modal reuses the image, i.e. the OLD clone
CACHE_BUST = os.environ.get("BFWI_MODAL_CACHE_BUST", "2026-10-23-gmwm-ladder-v2-N-override")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .env({"BRAIN_FWI_CACHE_BUST": CACHE_BUST})
    .run_commands(
        f"git clone --depth 1 --branch {GIT_BRANCH} https://github.com/m9h/brain-fwi.git /opt/brain-fwi",
        "cd /opt/brain-fwi && uv pip install --system -e '.[cuda13]'",
    )
)
vol = modal.Volume.from_name("brain-fwi-results", create_if_missing=True)


@app.function(image=image, gpu=GPU, timeout=6 * 60 * 60, memory=32 * 1024, volumes={"/vol": vol})
def run(stage: int, phase: str, iters: int, shots_per_iter: int, n_src: int, n_rec: int,
        noise_db: float | None, tag: str, n_grid: int | None = None, smooth: float = 1.0,
        invert_all: bool = False):
    import glob, os, shutil, subprocess, sys, time
    os.environ.update({"JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                       "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9", "TF_GPU_ALLOCATOR": "cuda_malloc_async",
                       "PYTHONUNBUFFERED": "1"})
    os.chdir("/opt/brain-fwi")
    cmd = ["python", "-u", "scripts/gmwm_velocity_3d.py", "--stage", str(stage), "--phase", phase,
           "--iters", str(iters), "--shots-per-iter", str(shots_per_iter),
           "--n-src", str(n_src), "--n-rec", str(n_rec), "--smooth", str(smooth)]
    if invert_all:
        cmd += ["--invert-all"]
    if n_grid:
        cmd += ["--N", str(n_grid)]            # ppw test: stage bands on a finer grid
    if noise_db is not None:
        cmd += ["--noise-db", str(noise_db)]
    if tag:
        cmd += ["--tag", tag]
    print("$", " ".join(cmd), flush=True)
    t0 = time.time(); log = []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True); log.append(line)
    rc = p.wait()
    TAG = f"s{stage}{phase}" + (f"_{tag}" if tag else "")
    vdir = "/vol/gmwm_velocity_3d"; os.makedirs(vdir, exist_ok=True)
    files = {}
    for f in glob.glob(f"results/gmwm_velocity_3d/*_{TAG}.*"):
        files[os.path.basename(f)] = open(f, "rb").read()
        shutil.copy(f, vdir)
    open(f"{vdir}/modal_{TAG}.log", "w").write("".join(log))
    vol.commit()
    return {"rc": rc, "tag": TAG, "log": "".join(log), "files": files, "wall_min": (time.time() - t0) / 60}


@app.local_entrypoint()
def main(stage: int = 3, phase: str = "B", iters: int = 20, shots_per_iter: int = 4,
         n_src: int = 16, n_rec: int = 128, noise_db: float = None, tag: str = "", n: int = None,
         smooth: float = 1.0, invert_all: bool = False):
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "gmwm_velocity_3d")
    os.makedirs(OUT, exist_ok=True)
    print(f"=== Modal GM/WM ladder: stage {stage} phase {phase} N={n or 'stage default'} smooth={smooth} "
          f"invert_all={invert_all} on {GPU} ({GIT_BRANCH}) ===", flush=True)
    res = run.remote(stage, phase, iters, shots_per_iter, n_src, n_rec, noise_db, tag, n, smooth, invert_all)
    for name, data in res["files"].items():
        with open(os.path.join(OUT, name), "wb") as fh:
            fh.write(data)
        print(f"saved results/gmwm_velocity_3d/{name} ({len(data)} bytes)", flush=True)
    with open(os.path.join(OUT, f"modal_{res['tag']}.log"), "w") as fh:
        fh.write(res["log"])
    tail = [l for l in res["log"].splitlines() if any(k in l for k in ("AUC", "VERDICT", "WM median", "residual", "FWI done"))]
    print("\n".join(tail[-8:]), flush=True)
    print(f"remote rc={res['rc']}  wall {res['wall_min']:.1f} min", flush=True)

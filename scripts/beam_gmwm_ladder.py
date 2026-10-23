"""Run the 3-D GM/WM frequency ladder (`scripts/gmwm_velocity_3d.py`) on beam.cloud.

Stage 1 (300 kHz, N=112) is done on the GB10 (Runs 1-4, `scope_3d_gmwm_run.md`); the
3 mm cortical ribbon is still resolution-limited there (AUC 0.846 at 2.60 mm). Stages
2-3 are what the ladder exists for, and they are multi-hour -> off-GB10 (queue-priority
rule) -> beam. Peak memory (measured, segmented checkpointing): N=144 ~17 GB (fits a
24 GB RTX4090), N=176 ~32.4 GB (does NOT fit any serverless card; needs A100-80/H100).

Outputs land in results/gmwm_velocity_3d/ under the SAME names the runner writes locally
(summary_<TAG>.json, c_rec_<TAG>.npy, gmwm3d_<TAG>.png) so run*_aggregate.py and the
docs work unchanged. Run from scripts/ (see .beamignore whitelist):

    cd scripts && ../.venv-beam/bin/python beam_gmwm_ladder.py --stage 2
    BFWI_BEAM_GPU=RTX5090 ../.venv-beam/bin/python beam_gmwm_ladder.py --stage 2 --tag rtx5090

Entrypoint is pure-python (.venv-beam has only the beam client): bytes in, bytes out.
"""
import argparse, os
from beam import function, Image, Volume

# outputs are ALSO copied here so a multi-hour result survives the local client dying
# (the CLI only has `beam volume list/create/delete`; read it back with a tiny @function
# that mounts the same Volume and returns the bytes, or via the beam web dashboard)
VOL = Volume(name="brain-fwi-results", mount_path="/vol")

GIT_BRANCH = os.environ.get("BFWI_BEAM_BRANCH", "feature/diffusion-prior-fwi")
GPU = os.environ.get("BFWI_BEAM_GPU", "RTX5090")           # serverless: RTX5090 (32 GB) / RTX4090 / A10G (24 GB)
# stage 3 (N=176, ~55 GB peak) needs on-demand hardware: `beam machine reserve --gpu A100-80 --ttl 3h`
# creates pool "ondemand-A100-80"; then BFWI_BEAM_GPU=A100-80 BFWI_BEAM_POOL=ondemand-A100-80
POOL = os.environ.get("BFWI_BEAM_POOL") or None
_pool_kw = {"pool": POOL} if POOL else {}
image = (
    Image(python_version="python3.11")
    .add_commands([
        "apt-get update && apt-get install -y git build-essential",
        f"git clone --depth 1 --branch {GIT_BRANCH} https://github.com/m9h/brain-fwi.git /opt/brain-fwi",
        "cd /opt/brain-fwi && pip install -e '.[cuda12]'",
    ])
    # stage 2 OOM'd on a 24 GB RTX4090 (one 14.4 GiB checkpoint-segment block on ~13 GB resident,
    # MEM_FRACTION=0.9 -> 21.6 GB cap). Needs the 32 GB RTX5090; async allocator cuts fragmentation.
    .with_envs(["JAX_PLATFORMS=cuda", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION=0.95", "TF_GPU_ALLOCATOR=cuda_malloc_async",
                "PYTHONUNBUFFERED=1"])
)


# defaults are cpu=0.125 / 128 MiB (jax import alone OOMs, exit 137) and timeout=3600; -1 = no cap
@function(gpu=GPU, image=image, cpu=4, memory="16Gi", timeout=-1, volumes=[VOL], **_pool_kw)
def run(stage, phase, iters, shots_per_iter, n_src, n_rec, noise_db, tag, n_grid=None):
    import subprocess, sys, json, glob, os, shutil
    os.chdir("/opt/brain-fwi")
    cmd = [sys.executable, "-u", "scripts/gmwm_velocity_3d.py", "--stage", str(stage), "--phase", phase,
           "--iters", str(iters), "--shots-per-iter", str(shots_per_iter),
           "--n-src", str(n_src), "--n-rec", str(n_rec)]
    if n_grid:
        cmd += ["--N", str(n_grid)]            # ppw test: stage bands on a finer grid
    if noise_db is not None:
        cmd += ["--noise-db", str(noise_db)]
    if tag:
        cmd += ["--tag", tag]
    print("$", " ".join(cmd), flush=True)
    log = []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:                      # stream to beam logs AND keep for the caller
        print(line, end="", flush=True); log.append(line)
    rc = p.wait()
    TAG = f"s{stage}{phase}" + (f"_{tag}" if tag else "")
    files = {}
    vdir = "/vol/gmwm_velocity_3d"; os.makedirs(vdir, exist_ok=True)
    for f in glob.glob(f"results/gmwm_velocity_3d/*_{TAG}.*"):
        files[os.path.basename(f)] = open(f, "rb").read()
        shutil.copy(f, vdir)
    open(f"{vdir}/beam_{TAG}.log", "w").write("".join(log))
    return {"rc": rc, "tag": TAG, "log": "".join(log), "files": files}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2, 3], default=2)
    ap.add_argument("--phase", choices=["A", "B"], default="B")      # A=no skull, B=transcranial
    ap.add_argument("--iters", type=int, default=20)                 # Run 1 defaults, for comparability
    ap.add_argument("--shots-per-iter", type=int, default=4)
    ap.add_argument("--n-src", type=int, default=16)
    ap.add_argument("--n-rec", type=int, default=128)
    ap.add_argument("--noise-db", type=float, default=None)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--N", type=int, default=None, help="grid override (ppw test); see gmwm_velocity_3d.py")
    a = ap.parse_args()
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "gmwm_velocity_3d")
    os.makedirs(OUT, exist_ok=True)
    print(f"=== beam GM/WM ladder: stage {a.stage} phase {a.phase} on {GPU} ({GIT_BRANCH}) ===", flush=True)
    # NO retry loop: on a multi-hour run the local gRPC stream can drop ("Unhandled GRPC error:
    # StatusCode.INTERNAL", client exits) while the REMOTE task keeps running — re-invoking
    # would launch a duplicate. Recovery = `beam task list` to watch it, then
    # `beam_fetch_results.py --tag <TAG>` to pull the files from the Volume.
    res = run.remote(a.stage, a.phase, a.iters, a.shots_per_iter, a.n_src, a.n_rec, a.noise_db, a.tag, a.N)
    if not res:
        raise SystemExit("no result returned (stream lost?) — task may still be running: "
                         "`beam task list`, then `beam_fetch_results.py --tag <TAG>`")
    for name, data in res["files"].items():
        with open(os.path.join(OUT, name), "wb") as fh:
            fh.write(data)
        print(f"saved results/gmwm_velocity_3d/{name} ({len(data)} bytes)", flush=True)
    with open(os.path.join(OUT, f"beam_{res['tag']}.log"), "w") as fh:
        fh.write(res["log"])
    tail = [l for l in res["log"].splitlines() if any(k in l for k in ("AUC", "VERDICT", "WM median", "residual"))]
    print("\n".join(tail[-8:]), flush=True)
    print(f"remote rc={res['rc']}", flush=True)
    raise SystemExit(res["rc"])

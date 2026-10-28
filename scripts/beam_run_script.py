"""Generic beam runner: execute any `scripts/<name>.py` of brain-fwi on a serverless GPU.

Generalises beam_gmwm_ladder.py so each experiment script does not need its own wrapper.
The remote clones GIT_BRANCH (push first!), runs the script from /opt/brain-fwi, streams its
log, and returns every file in `--outdir` whose name contains `--tag` (also copied to the
`brain-fwi-results` Volume; `beam_fetch_results.py --subdir <outdir-basename> --tag <tag>`
pulls them back if the gRPC stream drops). Run from scripts/ (see .beamignore):

    cd scripts && ../.venv-beam/bin/python beam_run_script.py \
        --script gmwm_unknown_skull.py --outdir results/gmwm_unknown_skull --tag pose007_free \
        -- --pose-vox 0.07 --pose-deg 0.1 --interface free --smooth 2.0 --tag pose007_free

Everything after `--` is passed verbatim to the script. GPU via BFWI_BEAM_GPU (default RTX5090).
"""
import argparse, os, sys
from beam import function, Image, Volume

GIT_BRANCH = os.environ.get("BFWI_BEAM_BRANCH", "feature/diffusion-prior-fwi")
GPU = os.environ.get("BFWI_BEAM_GPU", "RTX5090")
CACHE_BUST = os.environ.get("BFWI_BEAM_CACHE_BUST", "2026-10-24-v4-generic-runner")
VOL = Volume(name="brain-fwi-results", mount_path="/vol")
image = (
    Image(python_version="python3.11")
    .add_commands([
        "apt-get update && apt-get install -y git build-essential",
        f"echo cache-bust {CACHE_BUST}",
        f"git clone --depth 1 --branch {GIT_BRANCH} https://github.com/m9h/brain-fwi.git /opt/brain-fwi",
        "cd /opt/brain-fwi && pip install -e '.[cuda12]'",
    ])
    .with_envs(["JAX_PLATFORMS=cuda", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION=0.95", "TF_GPU_ALLOCATOR=cuda_malloc_async",
                "PYTHONUNBUFFERED=1"])
)


# defaults are cpu=0.125 / 128 MiB (jax import alone OOMs, exit 137) and timeout=3600; -1 = no cap
@function(gpu=GPU, image=image, cpu=4, memory="16Gi", timeout=-1, volumes=[VOL])
def run(script, script_args, outdir, tag):
    import glob, os, shutil, subprocess, sys, time
    os.chdir("/opt/brain-fwi")
    cmd = [sys.executable, "-u", f"scripts/{script}", *script_args]
    print("$", " ".join(cmd), flush=True)
    t0 = time.time(); log = []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True); log.append(line)
    rc = p.wait()
    vdir = f"/vol/{os.path.basename(outdir.rstrip('/'))}"; os.makedirs(vdir, exist_ok=True)
    files = {}
    for f in glob.glob(f"{outdir}/*{tag}*"):
        files[os.path.basename(f)] = open(f, "rb").read(); shutil.copy(f, vdir)
    open(f"{vdir}/beam_{tag}.log", "w").write("".join(log))
    return {"rc": rc, "log": "".join(log), "files": files, "wall_min": (time.time() - t0) / 60}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="file name under scripts/")
    ap.add_argument("--outdir", required=True, help="results dir the script writes to (repo-relative)")
    ap.add_argument("--tag", required=True, help="substring selecting the output files to ship back")
    a, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", a.outdir)
    os.makedirs(OUT, exist_ok=True)
    print(f"=== beam {a.script} {' '.join(rest)} on {GPU} ({GIT_BRANCH}) ===", flush=True)
    # no retry: a dropped stream leaves the REMOTE running; re-invoking would duplicate it
    res = run.remote(a.script, rest, a.outdir, a.tag)
    if not res:
        raise SystemExit("no result (stream lost?) — `beam task list`, then beam_fetch_results.py")
    for name, data in res["files"].items():
        with open(os.path.join(OUT, name), "wb") as fh:
            fh.write(data)
        print(f"saved {a.outdir}/{name} ({len(data)} bytes)", flush=True)
    with open(os.path.join(OUT, f"beam_{a.tag}.log"), "w") as fh:
        fh.write(res["log"])
    tail = [l for l in res["log"].splitlines() if any(k in l for k in ("AUC", "VERDICT", "median", "residual", "done"))]
    print("\n".join(tail[-8:]), flush=True)
    print(f"remote rc={res['rc']}  wall {res['wall_min']:.1f} min", flush=True)
    raise SystemExit(res["rc"])

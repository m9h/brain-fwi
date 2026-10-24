"""Generic Modal runner: execute `scripts/<name>.py` of brain-fwi on an A100-80GB, fanned out.

Modal twin of beam_run_script.py (beam's RTX5090 capacity and build workers were unreliable
on 2026-10-23/24). Takes a JSON list of argument lists and runs them IN PARALLEL with
`starmap`, one container each. Outputs matching --tag are returned and also committed to the
Modal volume `brain-fwi-results` under /<outdir-basename>/.

    .venv/bin/modal run --detach scripts/modal_run_script.py \
        --script gmwm_unknown_skull.py --outdir results/gmwm_unknown_skull \
        --jobs '[["--pose-vox","0.07","--pose-deg","0.1","--interface","free","--smooth","2.0","--tag","pose007_free_sm2"], ...]'

Each job's --tag (the value after "--tag" in its arg list) selects the files shipped back.
Push first: the image clones GIT_BRANCH. Bump BFWI_MODAL_CACHE_BUST after every push.
"""
from __future__ import annotations
import json, os
import modal

app = modal.App("brain-fwi-run-script")
GIT_BRANCH = os.environ.get("BFWI_MODAL_BRANCH", "feature/diffusion-prior-fwi")
GPU = os.environ.get("BFWI_MODAL_GPU", "A100-80GB")
CACHE_BUST = os.environ.get("BFWI_MODAL_CACHE_BUST", "2026-10-24-run-script-v1")

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
def run(script: str, script_args: list[str], outdir: str):
    import glob, os, shutil, subprocess, sys, time
    os.environ.update({"JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                       "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9", "TF_GPU_ALLOCATOR": "cuda_malloc_async",
                       "PYTHONUNBUFFERED": "1"})
    os.chdir("/opt/brain-fwi")
    tag = script_args[script_args.index("--tag") + 1] if "--tag" in script_args else "run"
    cmd = ["python", "-u", f"scripts/{script}", *script_args]
    print(f"[{tag}] $", " ".join(cmd), flush=True)
    t0 = time.time(); log = []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(f"[{tag}] {line}", end="", flush=True); log.append(line)
    rc = p.wait()
    vdir = f"/vol/{os.path.basename(outdir.rstrip('/'))}"; os.makedirs(vdir, exist_ok=True)
    files = {}
    for f in glob.glob(f"{outdir}/*{tag}*"):
        files[os.path.basename(f)] = open(f, "rb").read(); shutil.copy(f, vdir)
    open(f"{vdir}/modal_{tag}.log", "w").write("".join(log))
    vol.commit()
    return {"tag": tag, "rc": rc, "log": "".join(log), "files": files, "wall_min": (time.time() - t0) / 60}


@app.local_entrypoint()
def main(script: str, outdir: str, jobs: str):
    job_list = json.loads(jobs)
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", outdir)
    os.makedirs(OUT, exist_ok=True)
    print(f"=== Modal {script}: {len(job_list)} jobs in parallel on {GPU} ({GIT_BRANCH}) ===", flush=True)
    for res in run.starmap([(script, j, outdir) for j in job_list]):
        for name, data in res["files"].items():
            with open(os.path.join(OUT, name), "wb") as fh:
                fh.write(data)
        with open(os.path.join(OUT, f"modal_{res['tag']}.log"), "w") as fh:
            fh.write(res["log"])
        tail = [l for l in res["log"].splitlines() if any(k in l for k in ("AUC", "VERDICT", "median"))]
        print(f"--- {res['tag']}  rc={res['rc']}  {res['wall_min']:.1f} min  ({len(res['files'])} files)", flush=True)
        print("\n".join(tail[-5:]), flush=True)

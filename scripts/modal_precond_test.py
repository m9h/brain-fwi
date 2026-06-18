"""Modal: does preconditioning fix 3D skull recovery from water? (synthetic 96^3)

Runs run_full_usct.py twice — precondition OFF (the job-984 setting) vs ON —
on the synthetic head phantom from a water start, and compares % skull
contrast recovered. Synthetic (not MIDA) so it needs no dataset and avoids
GB10 contention; the conditioning effect is phantom-agnostic.

    modal run scripts/modal_precond_test.py
"""
from pathlib import Path
import modal

app = modal.App("brain-fwi-precond-test")
GIT_BRANCH = "feature/parallel-modal-phase0"      # base; has run_full_usct.py + deps
CACHE_BUST = "2026-06-15-precond-v1"
REPO_ROOT = Path(__file__).resolve().parent.parent

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
    # overlay local working tree: run_full_usct.py (with --precondition) + src
    .add_local_dir(str(REPO_ROOT / "src" / "brain_fwi"), "/opt/brain-fwi/src/brain_fwi",
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file(str(REPO_ROOT / "run_full_usct.py"), "/opt/brain-fwi/run_full_usct.py")
)
vol = modal.Volume.from_name("brain-fwi-precond", create_if_missing=True)


def _run(precond: bool):
    import os, subprocess
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    tag = "on" if precond else "off"
    out = f"/results/synth64_precond_{tag}.h5"
    # 64^3: below the 128^3 gradient-checkpoint threshold the backward is the
    # full-trajectory (memory-heavy) path, so 96^3 timed out at 2h. 64^3 keeps
    # that backward cheap and still exhibits the near-transducer gradient
    # dominance that preconditioning targets.
    # FEW elements (32): the per-source FWI backward compile is what blew past
    # the 2 h cap before (96 elements → 96 compiles), NOT the 64^3 grid itself.
    cmd = ["python", "-u", "/opt/brain-fwi/run_full_usct.py",
           "--phantom", "synthetic", "--grid-size", "64", "--dx", "0.0025",
           "--n-elements", "32", "--iters", "12", "--shots", "4",
           "--output", out, "--figures", f"/results/synth64_precond_{tag}.png"]
    if precond:
        cmd.append("--precondition")
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd="/opt/brain-fwi")
    vol.commit()
    return out


@app.function(image=image, gpu="A100-40GB", timeout=2 * 3600, volumes={"/results": vol})
def run_off():
    return _run(False)


@app.function(image=image, gpu="A100-40GB", timeout=2 * 3600, volumes={"/results": vol})
def run_on():
    return _run(True)


@app.function(image=image, timeout=600, volumes={"/results": vol})
def compare(off_h5, on_h5):
    import h5py, numpy as np

    def skull(p):
        f = h5py.File(p, "r")
        ct = np.array(f["velocity_true"]); cr = np.array(f["velocity_recon"]); f.close()
        m = ct > 2200
        ts, rs = float(ct[m].mean()), float(cr[m].mean())
        return 100.0 * (rs - 1500.0) / (ts - 1500.0), ts, rs

    o, n = skull(off_h5), skull(on_h5)
    print("\n=== synthetic 96^3, from water — % skull contrast recovered ===")
    print(f"  precond OFF (984 setting): {o[0]:5.1f}%   (true {o[1]:.0f} -> recon {o[2]:.0f})")
    print(f"  precond ON               : {n[0]:5.1f}%   (true {n[1]:.0f} -> recon {n[2]:.0f})")
    print(f"  delta = {n[0]-o[0]:+.1f} pts")


@app.local_entrypoint()
def main():
    off_call = run_off.spawn()
    on_call = run_on.spawn()
    off_h5 = off_call.get()
    on_h5 = on_call.get()
    compare.remote(off_h5, on_h5)

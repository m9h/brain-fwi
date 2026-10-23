"""Does beam's host driver support jax[cuda13]? (3-min gate before porting the 192^3 run.)

The MIDA 192^3 run faulted deterministically on Modal's jax[cuda12] (XID-31, 3 GPUs) but ran
fine on the GB10's jax[cuda13] (job 959). beam has 80GB cards (A100_80/H100). So replicate the
GB10's PROVEN cuda13 stack on beam -- IF beam's host driver is new enough for CUDA 13 (the wheels
bundle CUDA libs but the DRIVER is on beam's host). This builds the .[cuda13] image and just checks
jax initializes the GPU + a real matmul + a tiny j-Wave-style FFT. Pass -> port the full 192^3.
"""
import os
from beam import function, Image

# RTX4090/RTX5090/A10G are the serverless cards (2026-10); H100 is on-demand-only now.
GPU = os.environ.get("BFWI_BEAM_GPU", "RTX4090")
GIT_BRANCH = os.environ.get("BFWI_BEAM_BRANCH", "feature/diffusion-prior-fwi")
CUDA_EXTRA = os.environ.get("BFWI_BEAM_CUDA", "cuda13")               # cuda12 = the stack the FWI scripts use
image = (
    Image(python_version="python3.11")
    .add_commands([
        "apt-get update && apt-get install -y git build-essential",
        f"git clone --depth 1 --branch {GIT_BRANCH} https://github.com/m9h/brain-fwi.git /opt/brain-fwi",
        f"cd /opt/brain-fwi && pip install -e '.[{CUDA_EXTRA}]'",
    ])
    .with_envs(["JAX_PLATFORMS=cuda", "XLA_PYTHON_CLIENT_PREALLOCATE=false"])
)


# beam defaults are cpu=0.125 / memory=128 MiB: importing jax+CUDA libs alone OOMs (exit 137)
@function(gpu=GPU, image=image, cpu=4, memory="8Gi", timeout=1800)
def smoke():
    import sys; sys.path.insert(0, "/opt/brain-fwi")
    import jax, jax.numpy as jnp, jaxlib
    out = {"jax": jax.__version__, "jaxlib": jaxlib.__version__}
    try:
        out["devices"] = str(jax.devices())
        x = jnp.ones((2048, 2048)); out["matmul"] = float((x @ x).sum())     # real GPU compute
        out["fft_ok"] = float(jnp.abs(jnp.fft.fft(jnp.ones(4096))).sum())    # FFT path j-Wave uses
        # also confirm j-Wave imports + a tiny forward builds on cuda13
        from brain_fwi.simulation.forward import build_domain, build_medium
        build_medium(build_domain((32, 32, 32), 1.5e-3), 1500.0, 1000.0, pml_size=6)
        out["jwave_import"] = "ok"
        # the FWI gradient path imports this; absent = pyproject pinned a pre-fix jwave commit
        import jwave.acoustics.time_varying as tv
        out["absorber_fix"] = hasattr(tv, "absorbing_pressure_from_density")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return out


@function(gpu=GPU, image=image, cpu=1, memory="2Gi")
def _noop():
    return "ok"


if __name__ == "__main__":
    print(f"=== beam jax[{CUDA_EXTRA}] smoke test on {GPU} ({GIT_BRANCH}) ===", flush=True)
    res = None
    for attempt in range(3):
        try:
            res = smoke.remote()
        except Exception as e:
            print(f"attempt {attempt+1}/3 raised: {type(e).__name__}: {str(e)[:200]}", flush=True)
        if res:
            break
    print("RESULT:", res)
    if res and "error" not in res and res.get("jwave_import") == "ok":
        print(f"=> CUDA13 WORKS ON BEAM ({GPU}): jax {res.get('jax')}, devices {res.get('devices')} -> port the 192^3 run")
    else:
        print(f"=> {CUDA_EXTRA} FAILED on beam ({GPU}). exit 137 / 'out-of-memory' = container RAM "
              "(raise memory= on @function), otherwise driver/init error -> fall back to GB10")

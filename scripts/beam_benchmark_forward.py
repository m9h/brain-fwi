"""Beam.cloud micro-benchmark: per-forward-sim wall-clock for j-Wave on a cloud GPU.

The decisive test of whether beam.cloud can be our FWI venue. Modal's jax[cuda12]
+ j-Wave was pathologically slow (and capped at 2 h, which killed FWI runs); beam
has no documented timeout and on-demand A100/H100. This mirrors
scripts/modal_benchmark_forward.py: clone the repo, install .[cuda12], time the
first (JIT) call and warm-cache calls of a homogeneous-water forward sim across
grid sizes. Compare the warm median against GB10 to decide if beam is viable.

    .venv-beam/bin/python scripts/beam_benchmark_forward.py            # A100-80
    BFWI_BEAM_GPU=H100 .venv-beam/bin/python scripts/beam_benchmark_forward.py
"""
import os
from beam import function, Image

GIT_BRANCH = "feature/diffusion-prior-fwi"
# This beam account's serverless @function supports A10G / RTX4090 (24 GB) only;
# A100/H100/L40S are on-demand-pod only. 24 GB is plenty for j-Wave forward + the
# checkpointed FWI backward at <=96^3.
GPU = os.environ.get("BFWI_BEAM_GPU", "A10G")
GRIDS = [(48, 0.004), (64, 0.003), (96, 0.002)]
N_TIMED = 5

image = (
    Image(python_version="python3.11")
    .add_commands([
        "apt-get update && apt-get install -y git build-essential",
        f"git clone --depth 1 --branch {GIT_BRANCH} https://github.com/m9h/brain-fwi.git /opt/brain-fwi",
        "cd /opt/brain-fwi && pip install -e '.[cuda12]'",
    ])
    .with_envs(["JAX_PLATFORMS=cuda", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION=0.85"])
)


@function(gpu=GPU, image=image, timeout=1800)
def bench():
    import sys, time, json
    sys.path.insert(0, "/opt/brain-fwi")
    import subprocess
    subprocess.run(["nvidia-smi", "-L"], check=False)
    import jax, jax.numpy as jnp, numpy as np
    from brain_fwi.simulation.forward import (
        _build_source_signal, build_domain, build_medium, build_time_axis, simulate_shot_sensors)
    print(f"JAX devices: {jax.devices()}", flush=True)

    summary = []
    for grid_size, dx in GRIDS:
        gs = (grid_size,) * 3
        c = jnp.full(gs, 1500.0, jnp.float32); rho = jnp.full(gs, 1000.0, jnp.float32)
        domain = build_domain(gs, dx); medium = build_medium(domain, c, rho, pml_size=20)
        ta = build_time_axis(medium, cfl=0.3, t_end=30e-6); dt = float(ta.dt)
        n = int(float(ta.t_end) / dt); sig = _build_source_signal(3e5, dt, n)
        cx = grid_size // 2; src = (cx, cx, cx)
        sens = tuple(np.array([cx], dtype=np.int32) for _ in range(3))

        t0 = time.time()
        out = simulate_shot_sensors(medium, ta, src, sens, sig, dt)
        out.block_until_ready()
        jit_s = time.time() - t0
        times = []
        for _ in range(N_TIMED):
            t0 = time.time()
            out = simulate_shot_sensors(medium, ta, src, sens, sig, dt)
            out.block_until_ready()
            times.append(time.time() - t0)
        med = float(np.median(times))
        entry = {"grid": grid_size, "dx": dx, "steps": n, "jit_s": round(jit_s, 2),
                 "warm_median_s": round(med, 3), "warm_min_s": round(float(np.min(times)), 3)}
        summary.append(entry)
        print(f"  {grid_size}^3 ({n} steps): JIT {jit_s:.1f}s | warm median {med:.3f}s", flush=True)
    return summary


if __name__ == "__main__":
    import json
    print(f"=== beam j-Wave forward benchmark on {GPU} ===", flush=True)
    res = bench.remote()
    print("\nRESULT:", json.dumps(res, indent=2))

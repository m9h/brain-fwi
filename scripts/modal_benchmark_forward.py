"""Modal micro-benchmark: per-forward-sim wall-clock on A100.

Previous SIREN-vs-voxel runs hit the 1 h timeout — root cause was that
our per-gradient-step cost assumption was ~5× optimistic for Modal's
A100. This script measures the actual forward-simulation throughput
across grid sizes so future compute budgets are grounded in measurement,
not estimate.

Output (persisted to the ``brain-fwi-validation`` volume):

    /results/bench_fwd_<GRID>.json   per-grid timings
    /results/bench_fwd_summary.json  concatenated across grids

Each entry records first-call (JIT-included) time and median of N
warm-cache calls. Use these numbers to size the ``timeout=`` argument
for any FWI Modal run via::

    per_step ≈ 2-3 × median_forward_s
    total    ≈ (N_elements + 3 × iters × shots) × per_step + JIT + overhead

Usage::

    modal run --detach scripts/modal_benchmark_forward.py
"""

import modal

app = modal.App("brain-fwi-forward-benchmark")

# Grid sizes we care about, with matching dx so the physical domain stays
# in the same ballpark (~20 cm span). 128^3 is the stretch goal — skip
# if the A100 40GB can't hold the intermediate fields.
GRIDS = [
    (32, 0.006),
    (48, 0.004),
    (64, 0.003),
    (96, 0.002),
    (128, 0.0015),
]

# Number of timed calls per grid. First call includes JIT; subsequent
# calls are warm-cache and give the steady-state throughput.
N_TIMED_CALLS = 5

CACHE_BUST = "2026-04-24-bench-v1"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .env({"BRAIN_FWI_CACHE_BUST": CACHE_BUST})
    .run_commands(
        "git clone --depth 1 https://github.com/m9h/brain-fwi.git /opt/brain-fwi",
        "cd /opt/brain-fwi && uv pip install --system -e '.[cuda12]'",
    )
)

results_vol = modal.Volume.from_name(
    "brain-fwi-validation", create_if_missing=True,
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=30 * 60,  # 30 min — plenty for ~25 timed sims across 5 grids
    volumes={"/results": results_vol},
)
def run_benchmark():
    import json
    import os
    import subprocess
    import time

    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

    subprocess.run(["nvidia-smi", "-L"], check=True)

    import jax
    import jax.numpy as jnp
    import numpy as np

    from brain_fwi.simulation.forward import (
        _build_source_signal,
        build_domain,
        build_medium,
        build_time_axis,
        simulate_shot_sensors,
    )

    print(f"JAX devices: {jax.devices()}")

    summary = []
    for grid_size, dx in GRIDS:
        print(f"\n{'=' * 60}")
        print(f"  Grid {grid_size}^3, dx={dx*1e3:.1f} mm")
        print(f"{'=' * 60}")

        grid_shape = (grid_size, grid_size, grid_size)

        # Homogeneous water medium — simplest forward setup.
        c = jnp.full(grid_shape, 1500.0, dtype=jnp.float32)
        rho = jnp.full(grid_shape, 1000.0, dtype=jnp.float32)
        domain = build_domain(grid_shape, dx)
        medium = build_medium(domain, c, rho, pml_size=20)
        time_axis = build_time_axis(medium, cfl=0.3, t_end=30e-6)
        dt = float(time_axis.dt)
        n_samples = int(float(time_axis.t_end) / dt)
        signal = _build_source_signal(3e5, dt, n_samples)

        cx = grid_size // 2
        src_pos = (cx, cx, cx)
        sensor_positions_grid = tuple(
            np.array([cx], dtype=np.int32) for _ in range(3)
        )

        # First call: JIT + tracing
        t0 = time.time()
        out = simulate_shot_sensors(
            medium, time_axis, src_pos, sensor_positions_grid, signal, dt,
        )
        out.block_until_ready() if hasattr(out, "block_until_ready") else None
        jit_time = time.time() - t0
        print(f"  first call (JIT+trace): {jit_time:.2f}s")

        # Warm-cache timed calls
        times = []
        for i in range(N_TIMED_CALLS):
            t0 = time.time()
            out = simulate_shot_sensors(
                medium, time_axis, src_pos, sensor_positions_grid, signal, dt,
            )
            out.block_until_ready() if hasattr(out, "block_until_ready") else None
            times.append(time.time() - t0)
        times = np.array(times)

        entry = {
            "grid_size": grid_size,
            "dx_m": dx,
            "n_timesteps": n_samples,
            "jit_s": round(jit_time, 2),
            "mean_s": round(float(times.mean()), 3),
            "median_s": round(float(np.median(times)), 3),
            "stddev_s": round(float(times.std()), 3),
            "n_timed_calls": N_TIMED_CALLS,
        }
        summary.append(entry)

        detail_path = f"/results/bench_fwd_{grid_size:03d}.json"
        with open(detail_path, "w") as f:
            json.dump({**entry, "raw_times_s": [round(float(t), 3) for t in times]}, f, indent=2)
        results_vol.commit()

        print(f"  warm-cache mean: {entry['mean_s']:.3f}s  "
              f"median: {entry['median_s']:.3f}s  "
              f"stddev: {entry['stddev_s']:.3f}s")

    # Final summary
    summary_path = "/results/bench_fwd_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"runs": summary, "n_timed_calls": N_TIMED_CALLS}, f, indent=2)
    results_vol.commit()

    print("\n" + "=" * 60)
    print("  Forward-sim benchmark summary")
    print("=" * 60)
    print(f"  {'grid':>6}  {'jit':>8}  {'mean':>8}  {'median':>8}")
    for e in summary:
        print(f"  {e['grid_size']:>4}³  {e['jit_s']:>7.2f}s  "
              f"{e['mean_s']:>7.3f}s  {e['median_s']:>7.3f}s")
    print(f"\n  Written: {summary_path}")


@app.local_entrypoint()
def main():
    run_benchmark.remote()

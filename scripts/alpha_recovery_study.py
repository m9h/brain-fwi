"""Phase-6 alpha-recovery characterization (issue #48).

Question: does stronger data (more iters / multi-band / higher top frequency)
push the recovered attenuation fraction past the tissue-archetype MIDPOINT (50%
of truth)? Below the midpoint the manifold prior snaps alpha toward water and
hurts; above it the prior and the constitutive (CANN/KK) coupling become usable.

This is a CHARACTERIZATION script. It does not modify library code. It reuses the
exact phantom + harness from tests/test_multiparameter_fwi.py and sweeps the data
richness, reporting the in-blob recovered alpha fraction (in-blob mean / true 6.0)
for each setting.

Run:  PYTHONUNBUFFERED=1 uv run python scripts/alpha_recovery_study.py
"""
from __future__ import annotations

import time
import numpy as np
import jax.numpy as jnp
import jax.random as jr

from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, simulate_shot_sensors,
    _build_source_signal,
)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi

TRUE_ALPHA = 6.0


# --- phantom + ring copied verbatim from tests/test_multiparameter_fwi.py ---
def _alpha_blob_phantom(n=32):
    grid = (n, n, n)
    dx = 5e-4
    y = 1.1
    c = jnp.full(grid, 1500.0, jnp.float32)
    rho = jnp.full(grid, 1000.0, jnp.float32)
    zz, yy, xx = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    r = np.sqrt((zz - n / 2) ** 2 + (yy - n / 2) ** 2 + (xx - n / 2) ** 2)
    blob = (r <= n * 0.18)
    alpha = jnp.asarray(np.where(blob, TRUE_ALPHA, 0.0), dtype=jnp.float32)
    return grid, dx, y, c, rho, alpha, blob


def _ring(n, radius_frac=0.42):
    cx = cy = cz = n // 2
    rad = n * radius_frac
    angs = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    xs = np.clip(np.round(cx + rad * np.cos(angs)).astype(int), 0, n - 1)
    ys = np.clip(np.round(cy + rad * np.sin(angs)).astype(int), 0, n - 1)
    zs = np.full_like(xs, cz)
    return xs, ys, zs


def build_setup(n=32, f0=300e3, c_max=1600.0, pml=8):
    """Return everything needed for a run: observed data + geometry + signal."""
    grid, dx, y, c, rho, alpha_true, blob = _alpha_blob_phantom(n)
    t_end = 1.9 * (n * dx) / 1500.0

    dom = build_domain(grid, dx)
    ref_med = build_medium(dom, c_max, 1000.0, pml_size=pml)
    ta = build_time_axis(ref_med, cfl=0.3, t_end=t_end)
    dt = float(ta.dt); nt = int(ta.Nt)
    sig = _build_source_signal(f0, dt, nt)

    xs, ys, zs = _ring(n)
    recv = (xs, ys, zs)
    src_positions = [(int(xs[i]), int(ys[i]), int(zs[i])) for i in range(0, 16, 2)]

    observed = []
    for sp in src_positions:
        med = build_medium(dom, c, rho, pml_size=pml,
                           attenuation=alpha_true, alpha_power=y)
        observed.append(simulate_shot_sensors(med, ta, sp, recv, sig, dt))
    observed = jnp.stack(observed)

    return dict(grid=grid, dx=dx, y=y, c=c, rho=rho, alpha_true=alpha_true,
                blob=np.asarray(blob), recv=recv, src_positions=src_positions,
                observed=observed, sig=sig, dt=dt, t_end=t_end,
                c_max=c_max, pml=pml, nt=nt)


def run_case(name, setup, freq_bands, n_iters):
    s = setup
    roi = jnp.asarray(np.ones(s["grid"], np.float32))
    cfg = FWIConfig(
        freq_bands=freq_bands, n_iters_per_band=n_iters,
        shots_per_iter=len(s["src_positions"]),
        learning_rate=12.0, c_min=1400.0, c_max=s["c_max"], pml_size=s["pml"],
        cfl=0.3, gradient_smooth_sigma=1.0, alpha_power=s["y"],
        invert_attenuation=True, attenuation_lr=2.0, attenuation_max=12.0,
        attenuation_mask=roi, verbose=False,
    )
    t0 = time.time()
    res = run_fwi(s["observed"], s["c"], s["rho"], s["dx"],
                  s["src_positions"], s["recv"], s["sig"], s["dt"], s["t_end"],
                  config=cfg, key=jr.PRNGKey(0))
    dt_run = time.time() - t0
    a_rec = np.asarray(res.attenuation)
    blob = s["blob"]
    in_blob = float(a_rec[blob].mean())
    out_blob = float(a_rec[~blob].mean())
    frac = in_blob / TRUE_ALPHA
    total_iters = n_iters * len(freq_bands)
    bands_str = str([(int(a / 1e3), int(b / 1e3)) for a, b in freq_bands])
    print(f"[{name:34s}] bands={bands_str:40s} "
          f"iters/band={n_iters:2d} total={total_iters:3d} | "
          f"in-blob={in_blob:5.3f} frac={frac:5.3f} out={out_blob:6.3f} "
          f"| {dt_run:5.1f}s", flush=True)
    return dict(name=name, frac=frac, in_blob=in_blob, out_blob=out_blob,
                total_iters=total_iters)


def main():
    print("=== Phase-6 alpha recovery study (issue #48) ===", flush=True)
    print(f"true alpha in blob = {TRUE_ALPHA}; midpoint threshold = 50% (frac 0.5)\n",
          flush=True)

    results = []

    # --- observed at f0=300 kHz (baseline source content) ---
    print("Building setup A (f0=300 kHz)...", flush=True)
    setA = build_setup(f0=300e3)
    print(f"  grid={setA['grid']} dt={setA['dt']:.3e} Nt={setA['nt']}\n", flush=True)

    # 1. Baseline (exact test config)
    results.append(run_case("baseline single-band 24it",
                            setA, [(150e3, 300e3)], 24))
    # 2a. more iterations
    results.append(run_case("single-band 48it",
                            setA, [(150e3, 300e3)], 48))
    results.append(run_case("single-band 96it",
                            setA, [(150e3, 300e3)], 96))
    # 2b. multi-band low->high (24 iters/band)
    results.append(run_case("multiband 50-300 x24",
                            setA, [(50e3, 100e3), (100e3, 200e3), (200e3, 300e3)], 24))
    # multiband matched to ~baseline total iters
    results.append(run_case("multiband 50-300 x16",
                            setA, [(50e3, 100e3), (100e3, 200e3), (200e3, 300e3)], 16))

    # --- observed at f0=500 kHz (richer high-freq content) ---
    print("\nBuilding setup B (f0=500 kHz)...", flush=True)
    setB = build_setup(f0=500e3)

    # 2c. higher top frequency (ppw >= 6 at 500 kHz, safe on 0.5mm grid)
    results.append(run_case("hi-f multiband 50-500 x24",
                            setB, [(50e3, 150e3), (150e3, 300e3), (300e3, 500e3)], 24))
    results.append(run_case("hi-f single-band 300-500 x48",
                            setB, [(300e3, 500e3)], 48))

    # --- observed at f0=700 kHz (ppw~4.3, near practical PSTD floor) ---
    print("\nBuilding setup C (f0=700 kHz, ppw~4.3)...", flush=True)
    setC = build_setup(f0=700e3)
    results.append(run_case("hi-f multiband 100-700 x24",
                            setC, [(100e3, 300e3), (300e3, 500e3), (500e3, 700e3)], 24))

    print("\n=== SUMMARY (recovered alpha fraction vs midpoint 0.50) ===", flush=True)
    print(f"{'setting':36s} {'total_it':>8s} {'in-blob':>8s} {'frac':>6s} {'>50%?':>6s}",
          flush=True)
    for r in results:
        cross = "YES" if r["frac"] >= 0.5 else "no"
        print(f"{r['name']:36s} {r['total_iters']:8d} {r['in_blob']:8.3f} "
              f"{r['frac']:6.3f} {cross:>6s}", flush=True)


if __name__ == "__main__":
    main()

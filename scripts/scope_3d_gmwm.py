"""Scoping benchmark for the 3-D GM/WM velocity run.

Measures REAL forward+adjoint (gradient) wall-clock and peak memory per shot on
this GPU at several grid sizes, over a fixed physical FOV (so dx -> dt -> nt scale
together and the measured exponent is the true cost law), then extrapolates to the
grids needed for 300-500 kHz imaging frequency.

Cost should scale ~N^4 (N^3 voxels x N time steps at fixed FOV & CFL).
"""
from __future__ import annotations
import time, json, os
import numpy as np
import jax, jax.numpy as jnp

from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, simulate_shot_sensors,
    _build_source_signal)

FOV = 0.180                      # 180 mm head
C_MAX = 2900.0                   # skull CFL bound
OUT = "results/gmwm_velocity_resolution"; os.makedirs(OUT, exist_ok=True)

def bench(N, freq=300e3, n_rec=128):
    dx = FOV / N
    dom = build_domain((N, N, N), dx)
    c = jnp.full((N, N, N), 1560.0, jnp.float32)
    rho = jnp.full((N, N, N), 1040.0, jnp.float32)
    ref = build_medium(dom, C_MAX, 1000.0, pml_size=10)
    t_end = 1.6 * FOV / 1500.0
    ta = build_time_axis(ref, cfl=0.3, t_end=t_end)
    dt, nt = float(ta.dt), int(ta.Nt)
    med = build_medium(dom, c, rho, pml_size=10)
    sig = _build_source_signal(freq, dt, nt)
    cx = N // 2; R = int(0.44 * N)
    a = np.linspace(0, 2*np.pi, n_rec, endpoint=False)
    recv = ((cx + R*np.cos(a)).astype(int), (cx + R*np.sin(a)).astype(int),
            np.full(n_rec, cx, int))
    src = (cx - R, cx, cx)

    # run_fwi auto-enables segmented gradient checkpointing at N>=128; match that
    # so the measured cost law reflects the path the real inversion takes.
    use_ckpt = N >= 128
    def loss(cf):
        m = build_medium(dom, cf, rho, pml_size=10)
        tr = simulate_shot_sensors(m, ta, src, recv, sig, dt, checkpointed=use_ckpt)
        return jnp.sum(tr ** 2)
    g = jax.jit(jax.grad(loss))
    t0 = time.time(); _ = g(c).block_until_ready(); t_compile = time.time() - t0
    t0 = time.time(); _ = g(c).block_until_ready(); t_grad = time.time() - t0
    try:
        st = jax.local_devices()[0].memory_stats() or {}
        peak = st.get("peak_bytes_in_use", 0) / 1e9
    except Exception:
        peak = float("nan")
    return dict(N=N, dx_mm=dx*1e3, nt=nt, t_grad_s=t_grad, t_compile_s=t_compile,
                peak_GB=peak, voxels_M=N**3/1e6, ckpt=use_ckpt)

rows = []
for N in (128, 160, 176):
    try:
        r = bench(N)
        rows.append(r)
        print(f"N={r['N']:4d} dx={r['dx_mm']:.2f}mm nt={r['nt']:5d} "
              f"grad {r['t_grad_s']:7.2f}s peak {r['peak_GB']:6.2f}GB "
              f"({r['voxels_M']:.2f} Mvox)", flush=True)
    except Exception as e:
        print(f"N={N} FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
        break

# fit cost exponent  t ~ N^p
if len(rows) >= 2:
    Ns = np.array([r["N"] for r in rows], float)
    ts = np.array([r["t_grad_s"] for r in rows], float)
    p = float(np.polyfit(np.log(Ns), np.log(ts), 1)[0])
    k = float(np.exp(np.polyfit(np.log(Ns), np.log(ts), 1)[1]))
    print(f"\nmeasured cost law: t_grad ~ {k:.3e} * N^{p:.2f}  (theory N^4)", flush=True)

    # frequency -> required grid (>=3 pts per wavelength in brain at 1560 m/s)
    print("\n--- extrapolated 3-D FWI cost (fixed 180mm FOV) ---", flush=True)
    plan = []
    for f, ppw in [(300e3, 3.0), (400e3, 3.0), (500e3, 3.0), (500e3, 4.0)]:
        lam = 1560.0 / f
        dx_req = lam / ppw
        Nreq = int(np.ceil(FOV / dx_req / 16) * 16)     # round to multiple of 16
        t1 = k * Nreq ** p
        for (nsh, nit, nb) in [(8, 20, 2), (16, 25, 2)]:
            tot = t1 * nsh * nit * nb / 3600.0
            plan.append(dict(freq_kHz=f/1e3, ppw=ppw, N=Nreq, dx_mm=FOV/Nreq*1e3,
                             res_mm=lam/2*1e3, t_grad_s=t1, shots=nsh, iters=nit,
                             bands=nb, hours=tot))
            print(f"  {f/1e3:3.0f}kHz ppw{ppw:.0f} N={Nreq:3d} dx={FOV/Nreq*1e3:.2f}mm "
                  f"res={lam/2*1e3:.2f}mm | grad {t1:6.1f}s x {nsh}sh x {nit}it x {nb}b "
                  f"= {tot:6.1f} h", flush=True)
    json.dump({"bench": rows, "exponent": p, "coeff": k, "plan": plan},
              open(f"{OUT}/scope_3d.json", "w"), indent=2)
    print(f"\nsaved {OUT}/scope_3d.json", flush=True)

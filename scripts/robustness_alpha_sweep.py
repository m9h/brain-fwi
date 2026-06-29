#!/usr/bin/env python
"""Robustness curve: reconstruction error vs error in the assumed skull
absorption, using the MAITE-style harness (brain_fwi.robustness) over real
j-Wave FWI. Scale 0 = ignore absorption (lossless), 1 = true alpha, 2 = double.

Demonstrates the harness on real physics and quantifies how much alpha-prior
error the absorption-aware inversion tolerates. Small grid for speed.
"""
from __future__ import annotations
import importlib.util
import numpy as np
import jax.numpy as jnp
import jax.random as jr

spec = importlib.util.spec_from_file_location("ex06", "examples/06_absorption_aware_fwi_3d.py")
ex06 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex06)
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, generate_observed_data, _build_source_signal)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi
from brain_fwi.robustness import (
    AbsorptionErrorPerturber, robustness_sweep, plot_robustness_curve, brain_roi_rmse)

N = 48
c_true, rho_true, alpha_true, labels, dx = ex06.synthetic_head(N)
roi = ex06.brain_roi(labels)
src_positions, sensor_positions = ex06.make_helmet(labels, dx, n_elem=32, n_src=6)

ref = build_medium(build_domain((N, N, N), dx), ex06.C_MAX, 1000.0, pml_size=8)
t_end = 1.9 * (N * dx) / 1500.0
ta = build_time_axis(ref, cfl=0.3, t_end=t_end); dt = float(ta.dt); nt = int(ta.Nt)
f0 = 100e3; sig = _build_source_signal(f0, dt, nt)
print(f"grid {N}^3, dx={dx*1e3:.1f}mm, Nt={nt}; generating observed (true alpha)...", flush=True)
observed = generate_observed_data(
    jnp.asarray(c_true), jnp.asarray(rho_true), dx, src_positions, sensor_positions, f0,
    pml_size=8, time_axis=ta, source_signal=sig, dt=dt,
    attenuation=jnp.asarray(alpha_true), alpha_power=ex06.Y_POWER, verbose=False)

c_init = c_true.copy(); c_init[roi] = ex06.C_BRAIN0
rho_j = jnp.asarray(rho_true); mask_j = jnp.asarray(roi.astype(np.float32))
sample = dict(alpha_assumed=alpha_true, c_init=c_init, c_true=c_true, roi=roi)


def model(s):
    cfg = FWIConfig(
        freq_bands=[(50e3, 100e3)], n_iters_per_band=10, shots_per_iter=6,
        learning_rate=30.0, c_min=ex06.C_MIN, c_max=ex06.C_MAX, pml_size=8, cfl=0.3,
        gradient_smooth_sigma=1.0, mask=mask_j,
        attenuation=jnp.asarray(s["alpha_assumed"]), alpha_power=ex06.Y_POWER, verbose=False)
    res = run_fwi(observed, jnp.asarray(s["c_init"]), rho_j, dx, src_positions,
                  sensor_positions, sig, dt, t_end, config=cfg, key=jr.PRNGKey(0))
    return np.asarray(res.velocity)


metric = lambda recon, s: brain_roi_rmse(recon, s["c_true"], s["roi"])
strengths = [0.0, 0.5, 1.0, 1.5, 2.0]
print(f"sweeping assumed-alpha scale {strengths} ...", flush=True)
curve = robustness_sweep(model, AbsorptionErrorPerturber, strengths, sample, metric)
for x, r in curve:
    tag = " (lossless)" if x == 0 else (" (TRUE)" if x == 1 else "")
    print(f"  assumed-alpha x{x:>4}: brain RMSE {r:6.2f} m/s{tag}", flush=True)

out = "results/absorption_aware_fwi_3d/robustness_alpha.png"
plot_robustness_curve(
    curve, out, xlabel="assumed skull absorption / true  (0 = ignore absorption)",
    title="FWI robustness to skull-absorption error (MAITE-style sweep, 48³)", true_x=1.0)
print("saved", out, flush=True)
np.savez("results/absorption_aware_fwi_3d/robustness_alpha.npz",
         strengths=strengths, rmse=[r for _, r in curve])

"""Does PURE multi-band attenuation inversion (no c->a coupling, no prior) see
the GM/WM contrast? GM/WM differ ONLY in alpha (0.6 vs 0.9); the coupling is
degenerate and homogenised them in the capstone. Here we invert brain alpha
directly from multi-band data (c known at truth) and check whether the recon
preserves the CORRECT ordering (WM > GM) and any separability."""
import numpy as np, jax.numpy as jnp, jax.random as jr, time
from brain_fwi.phantoms.synthetic import make_gm_wm_contrast_head
from brain_fwi.simulation.forward import (build_domain, build_medium, build_time_axis,
    generate_observed_data, _build_source_signal)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi
from brain_fwi.robustness import gm_wm_separability

N, dx, Y = 64, 2e-3, 1.1
c, rho, alpha, labels, dx, gm, wm = make_gm_wm_contrast_head((N, N, N), dx)
brain = (labels == 2) | (labels == 3); C_MAX = 2900.0
print(f"{N}^3  GM a={alpha[gm].mean():.2f}  WM a={alpha[wm].mean():.2f}  (true ratio "
      f"{alpha[wm].mean()/alpha[gm].mean():.2f})", flush=True)
pml = 8; dom = build_domain((N, N, N), dx); ref = build_medium(dom, C_MAX, 1000.0, pml_size=pml)
ta = build_time_axis(ref, cfl=0.2, t_end=1.9*N*dx/1500.0); dt, nt = float(ta.dt), int(ta.Nt)
gi = (1+np.sqrt(5))/2; k = np.arange(220)
bx, by, bz = [int(np.where(brain)[i].mean()) for i in range(3)]
th = np.arccos(1-2*(k+0.5)/220); ph = 2*np.pi*k/gi; rad = N*0.34
px = np.clip(np.round(bx+rad*np.sin(th)*np.cos(ph)).astype(int), 9, N-10)
py = np.clip(np.round(by+rad*np.sin(th)*np.sin(ph)).astype(int), 9, N-10)
pz = np.clip(np.round(bz+rad*np.cos(th)).astype(int), 9, N-10)
recv = (px, py, pz); si = np.linspace(0, 219, 14).astype(int)
srcs = [(int(px[i]), int(py[i]), int(pz[i])) for i in si]
bands = [(80e3, 150e3), (150e3, 250e3)]; f0 = 250e3
sig = _build_source_signal(f0, dt, nt)
obs = generate_observed_data(jnp.asarray(c), jnp.asarray(rho), dx, srcs, recv, f0, pml_size=pml,
    time_axis=ta, source_signal=sig, dt=dt, attenuation=jnp.asarray(alpha), alpha_power=Y, verbose=False)
# c known at truth (GM/WM share it anyway); invert brain alpha ONLY, NO coupling, NO prior.
a_init = alpha.copy(); a_init[brain] = 0.0
bm = jnp.asarray(brain.astype(np.float32))
cfg = FWIConfig(freq_bands=bands, n_iters_per_band=18, shots_per_iter=10, learning_rate=15.0,
    c_min=1450.0, c_max=C_MAX, pml_size=pml, cfl=0.2, gradient_smooth_sigma=1.0, mask=bm,
    precondition=True, precondition_floor=0.05, invert_attenuation=True,
    attenuation_init=jnp.asarray(a_init), attenuation_lr=1.0, attenuation_max=6.0,
    attenuation_mask=bm, verbose=False)  # NB: no archetypes, no speed coupling
t0 = time.time()
res = run_fwi(obs, jnp.asarray(c), jnp.asarray(rho), dx, srcs, recv, sig, dt, 1.9*N*dx/1500.0,
              config=cfg, key=jr.PRNGKey(0))
a = np.asarray(res.attenuation)
gmm, wmm = a[gm].mean(), a[wm].mean()
print(f"PURE alpha inversion ({time.time()-t0:.0f}s): recon GM a={gmm:.3f}  WM a={wmm:.3f}  "
      f"ordering {'CORRECT (WM>GM)' if wmm>gmm else 'WRONG'}  sep={gm_wm_separability(a,gm,wm):.3f}", flush=True)
print(f"  recovered fraction: GM {gmm/0.6:.2f}  WM {wmm/0.9:.2f}", flush=True)

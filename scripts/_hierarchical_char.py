"""Definitive c-first demo: from a COLD start (uniform c), does the hierarchical
schedule (freeze alpha -> recover c -> release alpha + coupling) recover BOTH c
and alpha, where naive co-inversion-from-iter-0 fails (alpha starves c)?"""
import numpy as np, jax.numpy as jnp, jax.random as jr
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, simulate_shot_sensors, _build_source_signal)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi

n, dx, y = 32, 5e-4, 1.1
grid = (n, n, n)
zz, yy, xx = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
r = np.sqrt((zz - n/2)**2 + (yy - n/2)**2 + (xx - n/2)**2)
blob = r <= n*0.18
c_true = jnp.asarray(np.where(blob, 2000.0, 1500.0), np.float32)
alpha_true = jnp.asarray(np.where(blob, 6.0, 0.0), np.float32)
rho = jnp.full(grid, 1000.0, jnp.float32)
pml, c_max, t_end = 8, 2100.0, 1.9*(n*dx)/1500.0
dom = build_domain(grid, dx); ref = build_medium(dom, c_max, 1000.0, pml_size=pml)
ta = build_time_axis(ref, cfl=0.3, t_end=t_end); dt, nt = float(ta.dt), int(ta.Nt)
sig = _build_source_signal(300e3, dt, nt)
cx = n//2; rad = n*0.42; angs = np.linspace(0, 2*np.pi, 16, endpoint=False)
xs = np.clip(np.round(cx+rad*np.cos(angs)).astype(int), 0, n-1)
ys = np.clip(np.round(cx+rad*np.sin(angs)).astype(int), 0, n-1)
zs = np.full_like(xs, cx); recv = (xs, ys, zs)
srcs = [(int(xs[i]), int(ys[i]), int(zs[i])) for i in range(0, 16, 2)]
obs = jnp.stack([simulate_shot_sensors(
    build_medium(dom, c_true, rho, pml_size=pml, attenuation=alpha_true, alpha_power=y),
    ta, sp, recv, sig, dt) for sp in srcs])
c_init = jnp.full(grid, 1500.0, jnp.float32)   # COLD start
roi = jnp.asarray(np.ones(grid, np.float32))
anchors = (jnp.array([1500.0, 2000.0]), jnp.array([0.0, 6.0]))
base = dict(freq_bands=[(150e3, 300e3)], n_iters_per_band=40, shots_per_iter=len(srcs),
            learning_rate=20.0, c_min=1400.0, c_max=c_max, pml_size=pml, cfl=0.3,
            gradient_smooth_sigma=1.0, alpha_power=y, invert_attenuation=True,
            attenuation_lr=2.0, attenuation_max=12.0, attenuation_mask=roi,
            attenuation_speed_anchors=anchors, attenuation_speed_weight=0.4, verbose=False)
blob = np.asarray(blob); at = np.asarray(alpha_true); ct = np.asarray(c_true)
def rep(tag, res):
    a = np.asarray(res.attenuation); c = np.asarray(res.velocity)
    print(f"{tag}: c_in={c[blob].mean():.0f}(2000) a_in={a[blob].mean():.3f}(6.0) "
          f"a_rmse={np.sqrt(np.mean((a-at)**2)):.3f}", flush=True)
# Naive: co-invert from iter 0 (release_frac=0), coupling on.
r0 = run_fwi(obs, c_init, rho, dx, srcs, recv, sig, dt, t_end,
             config=FWIConfig(attenuation_release_frac=0.0, attenuation_prior_ramp=0.0, **base),
             key=jr.PRNGKey(0))
rep("naive co-invert", r0)
# Hierarchical: freeze alpha first half, then release + couple.
r1 = run_fwi(obs, c_init, rho, dx, srcs, recv, sig, dt, t_end,
             config=FWIConfig(attenuation_release_frac=0.5, attenuation_prior_ramp=0.0, **base),
             key=jr.PRNGKey(0))
rep("hierarchical   ", r1)

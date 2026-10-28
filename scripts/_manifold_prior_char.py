"""Characterization (not a test): does the tissue-α manifold prior help IN-LOOP
in the weak single-band regime? Compares free-voxel α vs manifold-prior α on the
same absorption-blob phantom as tests/test_multiparameter_fwi.py."""
import numpy as np, jax.numpy as jnp, jax.random as jr
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, simulate_shot_sensors, _build_source_signal)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi

n, dx, y = 32, 5e-4, 1.1
grid = (n, n, n)
c = jnp.full(grid, 1500.0, jnp.float32); rho = jnp.full(grid, 1000.0, jnp.float32)
zz, yy, xx = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
r = np.sqrt((zz - n/2)**2 + (yy - n/2)**2 + (xx - n/2)**2)
blob = r <= n*0.18
alpha_true = jnp.asarray(np.where(blob, 6.0, 0.0), np.float32)
pml, c_max, t_end = 8, 1600.0, 1.9*(n*dx)/1500.0
dom = build_domain(grid, dx); ref = build_medium(dom, c_max, 1000.0, pml_size=pml)
ta = build_time_axis(ref, cfl=0.3, t_end=t_end); dt, nt = float(ta.dt), int(ta.Nt)
sig = _build_source_signal(300e3, dt, nt)
cx = cy = cz = n//2; rad = n*0.42; angs = np.linspace(0, 2*np.pi, 16, endpoint=False)
xs = np.clip(np.round(cx+rad*np.cos(angs)).astype(int), 0, n-1)
ys = np.clip(np.round(cy+rad*np.sin(angs)).astype(int), 0, n-1)
zs = np.full_like(xs, cz); recv = (xs, ys, zs)
srcs = [(int(xs[i]), int(ys[i]), int(zs[i])) for i in range(0, 16, 2)]
obs = jnp.stack([simulate_shot_sensors(
    build_medium(dom, c, rho, pml_size=pml, attenuation=alpha_true, alpha_power=y),
    ta, sp, recv, sig, dt) for sp in srcs])
roi = jnp.asarray(np.ones(grid, np.float32))
base = dict(freq_bands=[(150e3, 300e3)], n_iters_per_band=24, shots_per_iter=len(srcs),
            learning_rate=12.0, c_min=1400.0, c_max=c_max, pml_size=pml, cfl=0.3,
            gradient_smooth_sigma=1.0, alpha_power=y, invert_attenuation=True,
            attenuation_lr=2.0, attenuation_max=12.0, attenuation_mask=roi, verbose=False)
blob = np.asarray(blob)
def rep(tag, res):
    a = np.asarray(res.attenuation)
    print(f"{tag}: rmse={np.sqrt(np.mean((a-np.asarray(alpha_true))**2)):.3f} "
          f"in={a[blob].mean():.3f} out={a[~blob].mean():.3f}", flush=True)
r0 = run_fwi(obs, c, rho, dx, srcs, recv, sig, dt, t_end, config=FWIConfig(**base), key=jr.PRNGKey(0))
rep("free-voxel   ", r0)
r1 = run_fwi(obs, c, rho, dx, srcs, recv, sig, dt, t_end,
             config=FWIConfig(attenuation_archetypes=jnp.array([0.0, 6.0]),
                              attenuation_prior_weight=0.3, attenuation_prior_ramp=0.5, **base),
             key=jr.PRNGKey(0))
rep("manifold-prior", r1)

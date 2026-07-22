"""Characterization: does the constitutive c->alpha coupling rescue weak alpha
when c and alpha co-vary (a 'skull-like' blob: high c AND high alpha)?
Compares free-voxel alpha vs coupling on the same phantom."""
import numpy as np, jax.numpy as jnp, jax.random as jr
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, simulate_shot_sensors, _build_source_signal)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi

n, dx, y = 32, 5e-4, 1.1
grid = (n, n, n)
zz, yy, xx = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
r = np.sqrt((zz - n/2)**2 + (yy - n/2)**2 + (xx - n/2)**2)
blob = r <= n*0.18
c_true = jnp.asarray(np.where(blob, 2000.0, 1500.0), np.float32)     # c contrast
alpha_true = jnp.asarray(np.where(blob, 6.0, 0.0), np.float32)       # co-varying alpha
rho = jnp.full(grid, 1000.0, jnp.float32)
pml, c_max, t_end = 8, 2100.0, 1.9*(n*dx)/1500.0
dom = build_domain(grid, dx); ref = build_medium(dom, c_max, 1000.0, pml_size=pml)
ta = build_time_axis(ref, cfl=0.3, t_end=t_end); dt, nt = float(ta.dt), int(ta.Nt)
sig = _build_source_signal(300e3, dt, nt)
cx = cy = cz = n//2; rad = n*0.42; angs = np.linspace(0, 2*np.pi, 16, endpoint=False)
xs = np.clip(np.round(cx+rad*np.cos(angs)).astype(int), 0, n-1)
ys = np.clip(np.round(cy+rad*np.sin(angs)).astype(int), 0, n-1)
zs = np.full_like(xs, cz); recv = (xs, ys, zs)
srcs = [(int(xs[i]), int(ys[i]), int(zs[i])) for i in range(0, 16, 2)]
obs = jnp.stack([simulate_shot_sensors(
    build_medium(dom, c_true, rho, pml_size=pml, attenuation=alpha_true, alpha_power=y),
    ta, sp, recv, sig, dt) for sp in srcs])
# Well-recovered-c regime (what a c-first / hierarchical schedule provides):
# start c at truth so the coupling can be isolated from the c-alpha crosstalk.
c_init = jnp.asarray(np.array(c_true), jnp.float32)
roi = jnp.asarray(np.ones(grid, np.float32))
base = dict(freq_bands=[(150e3, 300e3)], n_iters_per_band=20, shots_per_iter=len(srcs),
            learning_rate=20.0, c_min=1400.0, c_max=c_max, pml_size=pml, cfl=0.3,
            gradient_smooth_sigma=1.0, alpha_power=y, invert_attenuation=True,
            attenuation_lr=2.0, attenuation_max=12.0, attenuation_mask=roi, verbose=False)
blob = np.asarray(blob); at = np.asarray(alpha_true); ct = np.asarray(c_true)
def rep(tag, res):
    a = np.asarray(res.attenuation); c = np.asarray(res.velocity)
    print(f"{tag}: a_rmse={np.sqrt(np.mean((a-at)**2)):.3f} a_in={a[blob].mean():.3f} "
          f"a_out={a[~blob].mean():.3f} | c_in={c[blob].mean():.0f}(true 2000)", flush=True)
r0 = run_fwi(obs, c_init, rho, dx, srcs, recv, sig, dt, t_end, config=FWIConfig(**base), key=jr.PRNGKey(0))
rep("free-voxel", r0)
anchors = (jnp.array([1500.0, 2000.0]), jnp.array([0.0, 6.0]))
r1 = run_fwi(obs, c_init, rho, dx, srcs, recv, sig, dt, t_end,
             config=FWIConfig(attenuation_speed_anchors=anchors, attenuation_speed_weight=0.4,
                              attenuation_prior_ramp=0.4, **base), key=jr.PRNGKey(0))
rep("coupling  ", r1)

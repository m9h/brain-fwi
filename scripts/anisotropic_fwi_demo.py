"""Full-wave anisotropic FWI: recover the attenuation tensor (a_par, a_perp, fibre
angle phi) of a homogeneous anisotropic medium from ring transmission, through the
real j-Wave adjoint. Full-wave counterpart of the ray-tomography sandbox."""
import time, numpy as np, jax, jax.numpy as jnp, optax
from jwave.geometry import Medium
from brain_fwi.simulation.forward import (build_domain, build_time_axis,
    simulate_shot_sensors, _build_source_signal)

N, dx = 40, 5e-4; S = (N, N); pml = 8; Y = 1.1
dom = build_domain(S, dx)
c = jnp.full(S, 1500., jnp.float32); rho = jnp.full(S, 1000., jnp.float32)
cx = N // 2; R = int(N * 0.42)
ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
px = np.clip(np.round(cx + R * np.cos(ang)).astype(int), pml + 1, N - pml - 2)
py = np.clip(np.round(cx + R * np.sin(ang)).astype(int), pml + 1, N - pml - 2)
recv = (px, py); srcs = [(int(px[i]), int(py[i])) for i in range(0, 16, 4)]   # 4 sources

def tfield(a_par, a_perp, phi):
    cc, ss = jnp.cos(phi), jnp.sin(phi)
    return jnp.stack([jnp.full(S, a_par*cc*cc + a_perp*ss*ss),
                      jnp.full(S, a_par*ss*ss + a_perp*cc*cc),
                      jnp.full(S, (a_par - a_perp)*cc*ss)], -1)
def med(a_par, a_perp, phi):
    return Medium(domain=dom, sound_speed=c, density=rho, attenuation=0.0,
                  pml_size=pml, alpha_power=Y, attenuation_tensor=tfield(a_par, a_perp, phi))
ta = build_time_axis(med(1., 1., 0.), cfl=0.2, t_end=1.9*N*dx/1500.)
dt, nt = float(ta.dt), int(ta.Nt); sig = _build_source_signal(300e3, dt, nt)
A_PAR_T, A_PERP_T, PHI_T = 2.0, 8.0, 0.5
obs = [simulate_shot_sensors(med(A_PAR_T, A_PERP_T, PHI_T), ta, sp, recv, sig, dt) for sp in srcs]
print(f"truth a_par={A_PAR_T} a_perp={A_PERP_T} phi={PHI_T:.2f} | {len(srcs)} src, {len(px)} recv, nt={nt}", flush=True)

@jax.jit
def loss_and_grad(p):
    def loss(p):
        m = med(jax.nn.softplus(p[0]), jax.nn.softplus(p[1]), p[2])
        return sum(jnp.mean((simulate_shot_sensors(m, ta, sp, recv, sig, dt) - o) ** 2)
                   for sp, o in zip(srcs, obs))
    return jax.value_and_grad(loss)(p)

params = jnp.array([float(np.log(np.expm1(4.5)))] * 2 + [0.0])   # isotropic start, phi=0
opt = optax.adam(0.15); st = opt.init(params); t0 = time.time()
for it in range(40):
    l, g = loss_and_grad(params)
    u, st = opt.update(g, st); params = optax.apply_updates(params, u)
    if it % 5 == 0 or it == 39:
        ap, ae, ph = float(jax.nn.softplus(params[0])), float(jax.nn.softplus(params[1])), float(params[2])
        print(f"  it{it:2d} loss={float(l):.3e} a_par={ap:.2f} a_perp={ae:.2f} phi={ph:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
ap, ae, ph = float(jax.nn.softplus(params[0])), float(jax.nn.softplus(params[1])), float(params[2])
dphi = np.degrees(abs((ph - PHI_T + np.pi/2) % np.pi - np.pi/2))
print(f"RECOVERED a_par={ap:.2f}(2.0) a_perp={ae:.2f}(8.0) phi={ph:.2f}(0.50) | phi_err={dphi:.1f}deg "
      f"ratio={ae/ap:.1f}(4.0)", flush=True)

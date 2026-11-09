"""Full-wave anisotropic FWI: invert the attenuation tensor (a_par, a_perp, fibre
angle phi) of a homogeneous anisotropic medium from ring transmission, through the
real j-Wave adjoint (the validated anisotropic absorber). The full-wave
counterpart of the ray-tomography sandbox: recovers fibre orientation + anisotropy
from the differentiable wave solver.
"""

from __future__ import annotations

import numpy as np
import jax, jax.numpy as jnp
import optax
import pytest


def _tensor_field(a_par, a_perp, phi, S):
    c, s = jnp.cos(phi), jnp.sin(phi)
    Dxx = a_par * c * c + a_perp * s * s
    Dyy = a_par * s * s + a_perp * c * c
    Dxy = (a_par - a_perp) * c * s
    return jnp.stack([jnp.full(S, Dxx), jnp.full(S, Dyy), jnp.full(S, Dxy)], -1)


@pytest.mark.slow
def test_full_wave_anisotropic_fwi_recovers_tensor():
    from jwave.geometry import Medium
    from brain_fwi.simulation.forward import (
        build_domain, build_time_axis, simulate_shot_sensors, _build_source_signal)
    N, dx = 40, 5e-4; S = (N, N); pml = 8; Y = 1.1
    dom = build_domain(S, dx)
    c = jnp.full(S, 1500.0, jnp.float32); rho = jnp.full(S, 1000.0, jnp.float32)
    cx = N // 2; R = int(N * 0.42)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    px = np.clip(np.round(cx + R * np.cos(ang)).astype(int), pml + 1, N - pml - 2)
    py = np.clip(np.round(cx + R * np.sin(ang)).astype(int), pml + 1, N - pml - 2)
    recv = (px, py)
    srcs = [(int(px[i]), int(py[i])) for i in range(0, 16, 4)]        # 4 sources

    def med(a_par, a_perp, phi):
        return Medium(domain=dom, sound_speed=c, density=rho, attenuation=0.0,
                      pml_size=pml, alpha_power=Y, attenuation_tensor=_tensor_field(a_par, a_perp, phi, S))
    ta = build_time_axis(med(1., 1., 0.), cfl=0.2, t_end=1.9 * N * dx / 1500.)
    dt, nt = float(ta.dt), int(ta.Nt)
    sig = _build_source_signal(300e3, dt, nt)

    A_PAR_T, A_PERP_T, PHI_T = 2.0, 8.0, 0.5              # ground truth
    obs = [simulate_shot_sensors(med(A_PAR_T, A_PERP_T, PHI_T), ta, sp, recv, sig, dt) for sp in srcs]

    @jax.jit
    def loss_and_grad(p):
        def loss(p):
            m = med(jax.nn.softplus(p[0]), jax.nn.softplus(p[1]), p[2])
            return sum(jnp.mean((simulate_shot_sensors(m, ta, sp, recv, sig, dt) - o) ** 2)
                       for sp, o in zip(srcs, obs))
        return jax.value_and_grad(loss)(p)

    # start wrong: isotropic (a_par=a_perp=4.5), phi=0
    params = jnp.array([float(np.log(np.expm1(4.5))), float(np.log(np.expm1(4.5))), 0.0])
    opt = optax.adam(0.15); st = opt.init(params)
    for _ in range(40):
        _, g = loss_and_grad(params)
        u, st = opt.update(g, st); params = optax.apply_updates(params, u)
    a_par, a_perp = float(jax.nn.softplus(params[0])), float(jax.nn.softplus(params[1]))
    phi = float(params[2])
    dphi = abs((phi - PHI_T + np.pi / 2) % np.pi - np.pi / 2)     # mod pi
    assert abs(a_par - A_PAR_T) < 1.5 and abs(a_perp - A_PERP_T) < 2.0, f"a=({a_par:.1f},{a_perp:.1f})"
    assert dphi < np.radians(15), f"phi err {np.degrees(dphi):.1f} deg"
    assert a_perp > 1.8 * a_par, "anisotropy ratio not recovered"

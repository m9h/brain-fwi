"""Single-k, diffraction-free calibration of the anisotropic absorber.

Seeds a pure single-wavevector standing wave ``p0 = cos(k.x)`` in a PERIODIC
domain (pml=0), propagates it, projects the field back onto that mode, and fits
the envelope decay -> spatial attenuation ``alpha`` [dB/cm]. With a single k
there is no diffraction/k-spread, so this isolates the absorber's per-k rate and
compares it to the analytic ``alpha(k_hat) = (k_hat^T D k_hat) * (f/1e6)^y``.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

NP_PER_M_TO_DB_PER_CM = 8.685889638 / 100.0


def measure_single_k_alpha_db(a_par, a_perp, fibre_axis, prop_axis, N, dx, y, m,
                              c0=1500.0, rho0=1000.0, n_periods=24):
    """Measure the single-k attenuation (dB/cm) of the anisotropic absorber and
    return ``(alpha_measured, alpha_analytic)``.

    ``m`` = integer wavenumber index along ``prop_axis`` (k = 2*pi*m/(N*dx));
    fibre along ``fibre_axis`` with eigenvalue ``a_par``, else ``a_perp``.
    """
    from jaxdf import FourierSeries
    from jwave.geometry import Medium
    from jwave.acoustics.time_varying import simulate_wave_propagation
    from brain_fwi.simulation.forward import build_domain, build_time_axis

    S = (N, N, N)
    dom = build_domain(S, dx)
    ev = [a_perp, a_perp, a_perp]; ev[fibre_axis] = a_par
    D = np.zeros((*S, 6), np.float32)
    D[..., 0], D[..., 1], D[..., 2] = ev
    med = Medium(domain=dom, sound_speed=jnp.full(S, c0, jnp.float32),
                 density=jnp.full(S, rho0, jnp.float32), attenuation=0.0,
                 pml_size=0, alpha_power=y, attenuation_tensor=jnp.asarray(D))

    # single-k standing wave cos(k.x), k along prop_axis (periodic on the grid)
    line = np.cos(2.0 * np.pi * m * np.arange(N) / N).astype(np.float32)
    shape = [1, 1, 1]; shape[prop_axis] = N
    coskx = (line.reshape(shape) * np.ones(S, np.float32))
    p0 = FourierSeries(jnp.asarray(coskx[..., None]), dom)

    f = c0 * m / (N * dx)                      # frequency of this k
    t_end = n_periods / f
    ta = build_time_axis(med, cfl=0.2, t_end=t_end); dt = float(ta.dt); nt = int(ta.Nt)

    out = simulate_wave_propagation(med, ta, sources=None, p0=p0, u0=None)
    field = np.asarray(out.on_grid)            # (nt, *S, 1)
    P = field.reshape(field.shape[0], -1)      # (nt, Nvox)
    w = (coskx / np.sum(coskx ** 2)).ravel()
    a_t = P @ w                                # (nt,) single-k modal amplitude ~ cos(wt) exp(-a c t)

    # upper envelope: max |a_t| over sliding half-period windows, then fit ln vs t
    half_T_samples = max(2, int(0.5 / f / dt))
    times, env = [], []
    for s in range(0, nt - half_T_samples, half_T_samples):
        seg = np.abs(a_t[s:s + half_T_samples])
        env.append(seg.max()); times.append((s + half_T_samples / 2) * dt)
    times = np.array(times); env = np.array(env)
    keep = env > env[0] * 0.05                 # fit the clean decaying part
    slope = np.polyfit(times[keep], np.log(env[keep] + 1e-30), 1)[0]   # d ln A / dt = -alpha*c
    alpha_np_per_m = -slope / c0
    alpha_db = alpha_np_per_m * NP_PER_M_TO_DB_PER_CM

    eig = a_par if prop_axis == fibre_axis else a_perp
    analytic_db = eig * (f / 1e6) ** y
    return float(alpha_db), float(analytic_db)

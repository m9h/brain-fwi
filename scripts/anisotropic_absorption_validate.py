"""Plane-wave decay validation for the ANISOTROPIC absorber (jwave
feature/anisotropic-absorption). The k-Wave-substitute: k-Wave has no anisotropic
attenuation, so the analytic directional decay rate is the ground truth.

Method (adapts scripts/absorption_validate.py): propagation is along the long (x)
axis; the fibre is rotated so the SAME propagation direction samples the tensor's
along- vs across-fibre eigenvalue.
  - fibre || x  -> wave k || fibre  -> alpha_along  (eigenvalue a_par, LOW)
  - fibre || y  -> wave k across fibre -> alpha_across (eigenvalue a_perp, HIGH)
Measure alpha via the lossy/lossless amplitude-ratio slope (geometric spreading
cancels); compare to the analytic alpha(k_hat) = (k_hat^T D k_hat) * (f/1e6)^y
[dB/cm]. Confirms sign (alpha_along < alpha_across), directional ratio
(= a_par/a_perp), and magnitude calibration.
"""
import time, numpy as np
import jax, jax.numpy as jnp
from jwave.geometry import Medium
from brain_fwi.simulation.forward import build_domain, build_time_axis, simulate_shot_sensors
from brain_fwi.utils.wavelets import toneburst

S = (160, 24, 24); dx = 2.5e-4; pml = 8; Y = 1.1
A_PAR, A_PERP = 3.0, 9.0                                   # dB/cm/MHz^Y eigenvalues
dom = build_domain(S, dx)
c = jnp.full(S, 1500.0, jnp.float32); rho = jnp.full(S, 1000.0, jnp.float32)
src = (16, 12, 12)
recv_x = np.arange(40, 150, 12)
pg = (np.array(recv_x), np.full(len(recv_x), 12), np.full(len(recv_x), 12))

def tensor(fibre_axis):
    """Uniform tensor with eigenvalue a_par along `fibre_axis`, a_perp elsewhere."""
    ev = [A_PERP, A_PERP, A_PERP]; ev[fibre_axis] = A_PAR
    D = np.zeros((*S, 6), np.float32)
    D[..., 0], D[..., 1], D[..., 2] = ev                  # Dxx, Dyy, Dzz (off-diag 0)
    return jnp.asarray(D)

def medium(tens):
    return Medium(domain=dom, sound_speed=c, density=rho, attenuation=0.0,
                  pml_size=pml, alpha_power=Y, attenuation_tensor=tens)

ta = build_time_axis(medium(None), cfl=0.3, t_end=6e-5); dt = float(ta.dt); nt = int(6e-5 / dt)
def peak_amp(tens, sig):
    tr = np.asarray(simulate_shot_sensors(medium(tens), ta, src, pg, sig, dt, checkpointed=False))
    return np.max(np.abs(tr), axis=0)

print(f"anisotropic absorption validation: a_par={A_PAR} a_perp={A_PERP} dB/cm/MHz^{Y}, "
      f"propagation ||x", flush=True)
t0 = time.time()
L_cm = (recv_x - src[0]) * dx * 100.0; use = L_cm > 0.3
for f0 in [300e3, 500e3]:
    sig = toneburst(f0=f0, dt=dt, n_cycles=6, n_samples=nt)
    a_free = peak_amp(None, sig)
    for label, ax, eig in [("along-fibre (k||fibre)", 0, A_PAR), ("across-fibre (k⊥fibre)", 1, A_PERP)]:
        a_loss = peak_amp(tensor(ax), sig)
        R = np.clip(a_loss / (a_free + 1e-30), 1e-6, 1.0)
        am = np.polyfit(L_cm[use], -20 * np.log10(R[use]), 1)[0]      # dB/cm
        aa = eig * (f0 / 1e6) ** Y                                    # analytic dB/cm
        print(f"  f={f0/1e3:.0f}kHz {label}: alpha_meas {am:6.2f} | analytic {aa:5.2f} dB/cm | "
              f"ratio {am/aa:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    print(flush=True)

# differentiability of the anisotropic loss w.r.t. the tensor
sig = toneburst(f0=500e3, dt=dt, n_cycles=6, n_samples=nt)
obs = simulate_shot_sensors(medium(None), ta, src, pg, sig, dt, checkpointed=False)
def loss(scale):
    D = tensor(0) * scale
    pred = simulate_shot_sensors(medium(D), ta, src, pg, sig, dt, checkpointed=True)
    return jnp.mean((pred - obs) ** 2)
g = float(jax.grad(loss)(1.0))
print(f"d(misfit)/d(tensor scale) = {g:.3e} -> {'DIFFERENTIABLE' if g != 0 and np.isfinite(g) else 'NOT'}",
      flush=True)

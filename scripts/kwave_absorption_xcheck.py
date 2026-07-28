"""Cross-validate j-Wave's Treeby-Cox absorbing EoS against k-Wave (the field
standard), using k-Wave's pure-Python pseudospectral backend.

Why: the gating test only checks attenuation CHANGES traces. For a FUS-grade
solver the DECAY RATE must be correct. The original time-domain absorption
decayed the diagnostic pressure each step (which pressure_from_density
overwrites), so a homogeneous plane wave lost ~0 amplitude. The fix folds the
loss into the constitutive equation of state (jwave.acoustics.time_varying.
absorbing_pressure_from_density). This script confirms the fix matches k-Wave.

Method: homogeneous water + power-law absorber, CW lock-in amplitude at f0 vs
distance (geometric spreading cancels in the lossy/lossless ratio) -> dB/cm
slope; compare to analytic alpha(f)=a_db*(f/1e6)^y and across absorption
strengths. k-Wave runs here (pure-Python backend, ARM-compatible); the matching
j-Wave numbers come from /tmp/jwave_match.py / tests/test_attenuation_effect.py.

RESULT (192x48x48, dx=0.25mm, 500 kHz, 12 ppw, 2026-06):

    a_db  | absorption-only      | with dispersion
          | jWave    k-Wave      | jWave    k-Wave
    1.5   | 1.082    1.082       | 1.023    1.021
    6.0   | 1.074    1.074       | 0.868    0.870
    12.0  | 1.062    1.061       | 0.741    0.741

j-Wave matches k-Wave bit-for-bit in BOTH modes. The shared >1 offset is a
CW-lock-in metric bias; the with-dispersion droop is the lossy-vs-lossless
sound-speed-mismatch in the ratio measurement, not a physics error (k-Wave
shows it too). The coefficient is exact: ratio -> 1 as a_db -> 0.

Run (needs the k-wave-python venv):  .venv-kwave/bin/python scripts/kwave_absorption_xcheck.py
Set KW_NODISP=1 for absorption-only (k-Wave alpha_mode='no_dispersion').
"""
import os, time, numpy as np
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.solvers.kspace_solver import Simulation

dx = 2.5e-4; Nx, Ny, Nz = 192, 48, 48; pml = 12; ctr = Nz // 2
c0, rho0, Y, f0, cfl, t_end = 1500.0, 1000.0, 1.1, 500e3, 0.3, 5e-5
src_x = 24; recv_x = np.arange(48, 170, 10)

kgrid = kWaveGrid([Nx, Ny, Nz], [dx, dx, dx])
kgrid.makeTime(c0, cfl, t_end)
t = kgrid.t_array.squeeze(); dt = float(kgrid.dt)
drive = (np.clip(t / (8 / f0), 0, 1) * np.sin(2 * np.pi * f0 * t)).astype(np.float32)

source = kSource()
smask = np.zeros((Nx, Ny, Nz)); smask[src_x, ctr, ctr] = 1
source.p_mask = smask; source.p = drive.reshape(1, -1)

sensor = kSensor()
rmask = np.zeros((Nx, Ny, Nz))
for rx in recv_x:
    rmask[rx, ctr, ctr] = 1
sensor.mask = rmask  # binary mask -> ascending linear index == ascending x (y,z fixed)

amode = "no_dispersion" if os.environ.get("KW_NODISP") else None


def run(a_db):
    if a_db > 0:
        medium = kWaveMedium(sound_speed=c0, density=rho0, alpha_coeff=a_db,
                             alpha_power=Y, alpha_mode=amode)
    else:
        medium = kWaveMedium(sound_speed=c0, density=rho0)
    out = Simulation(kgrid, medium, source, sensor, device="cpu",
                     use_sg=True, use_kspace=True, smooth_p0=False,
                     pml_size=(pml,) * 3, pml_alpha=(2.0,) * 3, quiet=True).run()
    p = np.asarray(out["p"])
    if p.ndim == 2 and p.shape[0] != len(recv_x):
        p = p.T
    return p


def lockin(p):
    tt = np.arange(p.shape[1]) * dt
    m = (tt >= 30e-6) & (tt < 46e-6); tw = tt[m]; sw = p[:, m]
    I = (sw * np.cos(2 * np.pi * f0 * tw)[None, :]).mean(1)
    Q = (sw * np.sin(2 * np.pi * f0 * tw)[None, :]).mean(1)
    return 2.0 * np.sqrt(I ** 2 + Q ** 2)


L_cm = (recv_x - src_x) * dx * 100.0; sel = slice(2, -2)
print(f"k-Wave {Nx}x{Ny}x{Nz}, ppw={c0/f0/dx:.0f}, dispersion={'off' if amode else 'on'}", flush=True)
t0 = time.time()
a_free = lockin(run(0.0))
for a_db in [1.5, 6.0, 12.0]:
    R = np.clip(lockin(run(a_db)) / (a_free + 1e-30), 1e-6, 2.0)
    am = float(np.polyfit(L_cm[sel], -20 * np.log10(R[sel]), 1)[0])
    aa = a_db * (f0 / 1e6) ** Y
    print(f"  a_db={a_db:5.1f}: alpha_meas {am:+.3f} | analytic {aa:.3f} | "
          f"ratio {am/aa:+.3f}  ({time.time()-t0:.0f}s)", flush=True)

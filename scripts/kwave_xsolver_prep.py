"""Prep for the k-Wave independent-solver cross-check (run with the main .venv).

Computes the helmet geometry (source + sensor grid indices) and the j-Wave
reference observed data on the same medium, saving:
  /tmp/kwave_geom.json  - pg (3 x n_sensors), src (n_shots x 3), cfg
  /tmp/jwave_obs.npy     - j-Wave obs (n_shots, Nt, n_sensors), broadband ricker
  /tmp/jwave_meta.npz    - dt, nsteps, F0, t_end
beam_kwave_gen.py then generates the k-Wave obs on the same geometry; the compare
script aligns + quantifies the solver discrepancy.
"""
import json, numpy as np
import jax.numpy as jnp
from brain_fwi.phantoms.birnbaum import to_velocity, SKULL, C_SKULL
from brain_fwi.simulation.forward import build_domain, build_medium, build_time_axis, generate_observed_data
from brain_fwi.transducers import helmet_array_3d, transducer_positions_to_grid
from brain_fwi.utils.wavelets import ricker_wavelet

S = 96; dx = 1.5e-3; pml = 8; F0 = 80e3; T_END = 1.0e-4; N_SHOTS = 12
crop = np.load("/tmp/subj2_crop_96.npy")
c_true = jnp.asarray(to_velocity(crop, with_skull=True))
rho = jnp.full((S, S, S), 1000.0, jnp.float32)
c = (S // 2) * dx; r = (S // 2 - 4) * dx
pos = helmet_array_3d(n_elements=80, center=(c, c, c), radius_ap=r, radius_lr=r, radius_si=r,
                      standoff=0.0, coverage_angle=3.1, exclude_face=False)
pg = tuple(np.asarray(z) for z in transducer_positions_to_grid(pos, dx, (S, S, S))); ne = len(pg[0])
src = [[int(pg[0][i]), int(pg[1][i]), int(pg[2][i])] for i in np.linspace(0, ne - 1, N_SHOTS).astype(int)]
ta = build_time_axis(build_medium(build_domain((S, S, S), dx), C_SKULL, 1000.0, pml_size=pml), cfl=0.3, t_end=T_END)
dt = float(ta.dt); nsteps = int(T_END / dt); sig = ricker_wavelet(f0=F0, dt=dt, n_samples=nsteps)

# write geometry first so beam_kwave_gen can start while j-Wave obs generates
json.dump({"S": S, "dx": dx, "F0": F0, "t_end": T_END, "pml": pml,
           "pg": [pg[0].astype(int).tolist(), pg[1].astype(int).tolist(), pg[2].astype(int).tolist()],
           "src": src}, open("/tmp/kwave_geom.json", "w"))
print(f"wrote /tmp/kwave_geom.json ({ne} sensors, {N_SHOTS} shots)", flush=True)

obs = np.asarray(generate_observed_data(sound_speed=c_true, density=rho, dx=dx,
    src_positions_grid=[tuple(s) for s in src], sensor_positions_grid=pg, freq=F0,
    pml_size=pml, time_axis=ta, source_signal=sig, dt=dt, verbose=False))
np.save("/tmp/jwave_obs.npy", obs.astype(np.float32))
np.savez("/tmp/jwave_meta.npz", dt=dt, nsteps=nsteps, F0=F0, t_end=T_END)
print(f"prep done: jwave obs {obs.shape}, dt {dt*1e9:.1f}ns, nsteps {nsteps}, {ne} sensors, {N_SHOTS} shots")

"""Generate a starter FNO-surrogate training set on the GB10 (Track-2 / Phase-4).

Self-contained (bypasses the full Phase-0 HDF5 reader): for each training cerebrum
(with skull) at 64^3, run j-Wave forward from a few random helmet sources and save
(velocity, source, traces). Trains CToTraceFNO3D as a forward surrogate; the
cross-solver agreement harness then validates it like any other solver.
PROOF-OF-CONCEPT scale (~hundreds of samples); a useful surrogate needs Phase-0
scale (~10^4, per docs/design/phase4_fno_surrogate.md).
    .venv/bin/python scripts/fno_gen_dataset.py
"""
import os, time, numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", False)
import nibabel as nib
from brain_fwi.phantoms import birnbaum
from brain_fwi.simulation.forward import (build_domain, build_medium, build_time_axis, simulate_shot_sensors)
from brain_fwi.transducers import helmet_array_3d, transducer_positions_to_grid
from brain_fwi.utils.wavelets import ricker_wavelet

S = 64; dx = 2.25e-3; pml = 6; F0 = 70e3; T_END = 9.0e-5
N_SRC = int(os.environ.get("FNO_NSRC", "4")); N_SUBJ = int(os.environ.get("FNO_NSUBJ", "56"))
files = birnbaum.label_files()[:-4][:N_SUBJ]                  # training subjects (hold out subj1-4)
c = (S // 2) * dx; r = (S // 2 - 3) * dx
pos = helmet_array_3d(n_elements=64, center=(c, c, c), radius_ap=r, radius_lr=r, radius_si=r,
                      standoff=0.0, coverage_angle=3.1, exclude_face=False)
pg = tuple(np.asarray(z) for z in transducer_positions_to_grid(pos, dx, (S, S, S))); ne = len(pg[0])
allsrc = [(int(pg[0][i]), int(pg[1][i]), int(pg[2][i])) for i in range(ne)]
rho = jnp.full((S, S, S), 1000.0, jnp.float32)
ta = build_time_axis(build_medium(build_domain((S, S, S), dx), 2800.0, 1000.0, pml_size=pml), cfl=0.3, t_end=T_END)
dt = float(ta.dt); nsteps = int(T_END / dt); sig = ricker_wavelet(f0=F0, dt=dt, n_samples=nsteps)
rng = np.random.default_rng(0)
print(f"S={S}, {ne} receivers, {nsteps} steps, {len(files)} subjects x {N_SRC} sources", flush=True)

vels, srcs, traces = [], [], []
t0 = time.time()
for fi, f in enumerate(files):
    lab = np.asarray(nib.load(f).dataobj).astype(np.int16)
    crop = birnbaum.cerebrum_volume_crop(lab, S=S)
    if crop is None:
        continue
    v = birnbaum.to_velocity(crop, with_skull=True).astype(np.float32)
    cj = jnp.asarray(v)
    sidx = rng.choice(ne, N_SRC, replace=False)
    for si in sidx:
        med = build_medium(build_domain((S, S, S), dx), cj, rho, pml_size=pml)
        tr = np.asarray(simulate_shot_sensors(med, ta, allsrc[int(si)], pg, sig, dt))  # (Nt, n_recv)
        vels.append(v); srcs.append(list(allsrc[int(si)])); traces.append(tr[:nsteps].astype(np.float32))
    if (fi + 1) % 10 == 0:
        print(f"  {fi+1}/{len(files)} subjects, {len(traces)} samples ({time.time()-t0:.0f}s)", flush=True)

vels = np.stack(vels); srcs = np.array(srcs); traces = np.stack(traces)
recv = np.stack([pg[0], pg[1], pg[2]], axis=1)
np.savez("/tmp/fno_dataset_64.npz", sound_speed=vels.astype(np.float16), sources=srcs,
         traces=traces.astype(np.float16), receivers=recv, dx=dx, dt=dt, nsteps=nsteps, S=S)
print(f"saved /tmp/fno_dataset_64.npz: {len(traces)} samples, vel {vels.shape}, traces {traces.shape}, "
      f"{ne} recv, {time.time()-t0:.0f}s", flush=True)

"""Generate k-Wave reference obs on beam (x86) for the independent-solver check.

Reads the geometry (kwave_geom.json) + cerebrum crop (shipped as bytes) from the
entrypoint, runs k-Wave forward per shot on the SAME medium/geometry as j-Wave,
reorders sensors to j-Wave's pg order, and returns the obs (n_shots, Nt, n_sensors)
+ dt. The compare script then aligns to j-Wave's time axis and quantifies the
solver discrepancy. Run:  (cd scripts && ../.venv-beam/bin/python beam_kwave_gen.py)
"""
import json
from beam import function, Image

image = (
    Image(python_version="python3.11")
    .add_commands(["apt-get update && apt-get install -y libgomp1 libgl1 libglib2.0-0",
                   "apt-get install -y libhdf5-103 || apt-get install -y libhdf5-103-1 || apt-get install -y libhdf5-dev",
                   "pip install k-wave-python numpy scipy"])
)


@function(image=image, cpu=16, memory="32Gi", timeout=60 * 60)
def gen(crop_bytes, geom):
    import io, numpy as np
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksource import kSource
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_options import SimulationOptions
    from kwave.options.simulation_execution_options import SimulationExecutionOptions

    S = geom["S"]; dx = geom["dx"]; F0 = geom["F0"]; T_END = geom["t_end"]
    pg = [np.array(geom["pg"][d], dtype=int) for d in range(3)]; ne = len(pg[0])
    src = geom["src"]
    crop = np.load(io.BytesIO(crop_bytes))
    v = np.full((S, S, S), 1500.0, np.float32)
    v[np.isin(crop, (2, 3, 4))] = 1560.0; v[crop == 1] = 1660.0; v[crop == 5] = 2800.0

    # determine dt/Nt once (deterministic from c_max)
    kg0 = kWaveGrid([S, S, S], [dx, dx, dx])
    kg0.makeTime(v, cfl=0.3, t_end=T_END)
    Nt = int(kg0.Nt); dt = float(kg0.dt)
    t = np.arange(Nt) * dt
    tp = np.pi * F0 * (t - 1.5 / F0)
    ricker = ((1.0 - 2.0 * tp ** 2) * np.exp(-tp ** 2)).astype(np.float32)   # ricker @ F0

    smask = np.zeros((S, S, S), dtype=bool); smask[pg[0], pg[1], pg[2]] = True
    # return RAW sensor data (k-Wave mask order); the compare script tries both
    # C- and F-order reorderings against j-Wave (k-Wave's convention is ambiguous).
    eo = SimulationExecutionOptions(is_gpu_simulation=False)
    obs = []
    for k, sp in enumerate(src):
        # rebuild ALL objects fresh per shot: kspaceFirstOrder3D mutates kgrid
        # (PML), so reuse corrupts state. pml_inside=True -> grid stays S^3 (no
        # expansion), matching j-Wave's inside-PML.
        kgrid = kWaveGrid([S, S, S], [dx, dx, dx]); kgrid.setTime(Nt, dt)
        medium = kWaveMedium(sound_speed=v.copy(), density=1000.0 * np.ones((S, S, S)))
        source = kSource(); pm = np.zeros((S, S, S), dtype=bool); pm[sp[0], sp[1], sp[2]] = True
        source.p_mask = pm; source.p = ricker[None, :].copy()
        sensor = kSensor(); sensor.mask = smask.copy()
        so = SimulationOptions(save_to_disk=True, pml_inside=True, pml_size=int(geom["pml"]))
        out = kspaceFirstOrder3D(kgrid=kgrid, source=source, sensor=sensor, medium=medium,
                                 simulation_options=so, execution_options=eo)
        p = out["p"] if isinstance(out, dict) else out          # (n_active, Nt)
        p = np.asarray(p)
        if p.shape[0] != ne and p.shape[1] == ne:
            p = p.T
        obs.append(p.T.astype(np.float32))                       # (Nt, n_sensors) RAW k-Wave order
        print(f"  k-Wave shot {k+1}/{len(src)} done", flush=True)
    # decimate by 4 + float16 to fit beam's 4 MiB return cap (content <=120 kHz,
    # decimated Nyquist ~750 kHz -> no aliasing).
    DECIM = 4
    out_obs = np.stack(obs)[:, ::DECIM, :].astype(np.float16)
    return {"obs": out_obs, "dt": dt * DECIM, "Nt": int(out_obs.shape[1]), "F0": F0}


if __name__ == "__main__":
    import numpy as np
    geom = json.load(open("/tmp/kwave_geom.json"))
    crop_bytes = open("/tmp/subj2_crop_96.npy", "rb").read()
    print("=== k-Wave gen on beam ===", flush=True)
    res = None
    for attempt in range(4):
        try:
            res = gen.remote(crop_bytes, geom)
        except Exception as e:
            print(f"attempt {attempt+1}/4: {type(e).__name__}: {str(e)[:160]}", flush=True)
        if res:
            break
        print(f"attempt {attempt+1}/4 failed; retrying...", flush=True)
    if not res:
        print("FAILED after retries"); raise SystemExit(1)
    np.save("/tmp/kwave_obs.npy", res["obs"])
    np.savez("/tmp/kwave_meta.npz", dt=res["dt"], Nt=res["Nt"], F0=res["F0"])
    print(f"saved /tmp/kwave_obs.npy {res['obs'].shape}, dt {res['dt']*1e9:.1f}ns, Nt {res['Nt']}")

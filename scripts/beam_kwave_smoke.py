"""Gating smoke test: does k-Wave run on beam's x86 workers? (GB10 is ARM -> the
bundled k-Wave binary is Exec-format-incompatible there.) If this prints a sensor
shape, the independent-solver cross-validation via k-Wave-on-beam is viable."""
from beam import function, Image

image = (
    Image(python_version="python3.11")
    .add_commands(["apt-get update && apt-get install -y libgomp1 libgl1 libglib2.0-0",
                   # k-Wave OMP binary needs libhdf5_serial.so.103 (distro-specific soname)
                   "apt-get install -y libhdf5-103 || apt-get install -y libhdf5-103-1 || apt-get install -y libhdf5-dev",
                   "pip install k-wave-python numpy"])
)


@function(image=image, cpu=4, memory="8Gi")
def smoke():
    import numpy as np
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksource import kSource
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_options import SimulationOptions
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    import platform
    N = 32; dx = 1e-3
    kgrid = kWaveGrid([N, N, N], [dx, dx, dx])
    medium = kWaveMedium(sound_speed=1500.0 * np.ones((N, N, N)), density=1000.0 * np.ones((N, N, N)))
    kgrid.makeTime(medium.sound_speed)
    src = kSource(); src.p_mask = np.zeros((N, N, N)); src.p_mask[N//2, N//2, N//2] = 1
    t = np.arange(kgrid.Nt) * kgrid.dt
    src.p = (np.sin(2*np.pi*1e5*t) * np.exp(-((t-5e-6)/2e-6)**2))[None, :]
    sensor = kSensor(); sensor.mask = np.zeros((N, N, N)); sensor.mask[N//2, N//2, 4] = 1
    so = SimulationOptions(save_to_disk=True, pml_inside=False)
    eo = SimulationExecutionOptions(is_gpu_simulation=False)
    import subprocess
    try:
        out = kspaceFirstOrder3D(kgrid=kgrid, source=src, sensor=sensor, medium=medium,
                                 simulation_options=so, execution_options=eo)
    except subprocess.CalledProcessError as e:
        return {"arch": platform.machine(), "error": "CalledProcessError",
                "returncode": e.returncode, "cmd": str(e.cmd)[:300],
                "stdout": (e.output or b"")[-1500:] if isinstance(e.output, bytes) else str(e.output)[-1500:],
                "stderr": (e.stderr or b"")[-1500:] if isinstance(e.stderr, bytes) else str(e.stderr)[-1500:]}
    p = out["p"] if isinstance(out, dict) else out
    return {"arch": platform.machine(), "shape": list(np.asarray(p).shape), "Nt": int(kgrid.Nt)}


if __name__ == "__main__":
    print("=== k-Wave on beam smoke ===", flush=True)
    for attempt in range(4):
        try:
            r = smoke.remote()
            if r:
                print("RESULT:", r); break
        except Exception as e:
            print(f"attempt {attempt+1}/4: {type(e).__name__}: {str(e)[:160]}", flush=True)

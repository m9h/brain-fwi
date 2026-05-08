"""prepare.py — Brain FWI API for AgenticSciML experiments.

Exposes a clean interface for the Engineer agent to run FWI experiments
and report results.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

# Use brain_fwi internals
from brain_fwi.phantoms.properties import map_labels_to_all
from brain_fwi.transducers.helmet import helmet_array_3d, transducer_positions_to_grid
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis,
    simulate_shot_sensors, _build_source_signal,
)
from brain_fwi.inversion.fwi import FWIConfig as InternalFWIConfig, run_fwi

@dataclass
class FWIConfig:
    """Simplified configuration for AgenticSciML discovery."""
    freq_bands: list[tuple[float, float]] = field(default_factory=lambda: [(40e3, 80e3), (80e3, 150e3)])
    n_iters_per_band: int = 15
    shots_per_iter: int = 4
    learning_rate: float = 50.0
    parameterization: str = "voxel"
    siren_lr: float = 1e-3
    siren_pretrain_steps: int = 0
    loss_fn: str = "l2"
    gradient_smooth_sigma: float = 3.0
    mask_type: str = "head"  # "head", "brain", "none"
    dx: float = 0.002
    grid_size: int = 64
    phantom: str = "synthetic"  # "synthetic" or "mida"
    seed: int = 42

@dataclass
class ExperimentResult:
    commit: str
    config: FWIConfig
    brain_rmse: float
    skull_rmse: float
    loss_history: list[float]
    wall_time: float
    dt: float = 0.0
    n_t: int = 0
    status: str = "success"

    @property
    def metrics(self) -> dict[str, float]:
        """Hallucination-protection for AgenticSciML agents."""
        return {
            "brain_rmse": self.brain_rmse,
            "skull_rmse": self.skull_rmse,
            "dt": self.dt,
            "n_t": float(self.n_t),
        }

def get_commit_hash() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()[:8]
    except Exception:
        return "unknown"

def create_phantom(grid_shape, dx, phantom_type="synthetic"):
    from brain_fwi.phantoms.synthetic import make_three_layer_head
    if phantom_type == "synthetic":
        labels = jnp.asarray(make_three_layer_head(grid_shape, dx))
    else:
        # Fallback to synthetic if MIDA not found, for CI/local safety
        labels = jnp.asarray(make_three_layer_head(grid_shape, dx))

    props = map_labels_to_all(labels)
    c = jnp.where(labels == 0, 1500.0, props["sound_speed"])
    rho = jnp.where(labels == 0, 1000.0, props["density"])
    alpha = jnp.where(labels == 0, 0.0, props["attenuation"])
    return labels, c, rho, alpha

def run_fwi_experiment(config: FWIConfig) -> ExperimentResult:
    t0 = time.time()
    grid_shape = (config.grid_size, config.grid_size, config.grid_size)
    dx = config.dx

    # 1. Setup Phantom
    labels, c_true, rho, alpha = create_phantom(grid_shape, dx, config.phantom)

    # 2. Setup Helmet
    cx_m = grid_shape[0] * dx / 2
    cy_m = grid_shape[1] * dx / 2
    cz_m = grid_shape[2] * dx / 2
    positions = helmet_array_3d(
        n_elements=128,
        center=(cx_m, cy_m, cz_m),
        radius_ap=0.09, radius_lr=0.07, radius_si=0.08,
        standoff=0.005, coverage_angle=2.8, exclude_face=True,
    )
    pos_grid = transducer_positions_to_grid(positions, dx, grid_shape)
    src_list = [(int(pos_grid[0][i]), int(pos_grid[1][i]), int(pos_grid[2][i]))
                for i in range(len(pos_grid[0]))]

    # 3. Generate Data (Subset of shots for speed)
    max_freq = max(f for _, f in config.freq_bands)
    domain = build_domain(grid_shape, dx)
    ref_medium = build_medium(domain, 3200.0, 1000.0, pml_size=10)
    time_axis = build_time_axis(ref_medium, cfl=0.3)
    dt = float(time_axis.dt)
    t_end = float(time_axis.t_end)
    n_samples = int(t_end / dt)
    source_signal = _build_source_signal(max_freq, dt, n_samples)

    medium_true = build_medium(domain, c_true, rho, pml_size=10, attenuation=alpha)
    n_data_src = min(len(src_list), config.shots_per_iter * 2) # Reduced for experiment speed

    observed = []
    for i in range(n_data_src):
        d = simulate_shot_sensors(medium_true, time_axis, src_list[i], pos_grid, source_signal, dt)
        observed.append(d)
    observed = jnp.stack(observed, axis=0)

    # 4. Inversion
    c_init = jnp.full(grid_shape, 1500.0, dtype=jnp.float32)

    if config.mask_type == "brain":
        mask = ((labels == 2) | (labels == 3)).astype(jnp.float32)
    elif config.mask_type == "head":
        mask = (labels > 0).astype(jnp.float32)
    else:
        mask = jnp.ones(grid_shape, dtype=jnp.float32)

    internal_config = InternalFWIConfig(
        freq_bands=config.freq_bands,
        n_iters_per_band=config.n_iters_per_band,
        shots_per_iter=config.shots_per_iter,
        learning_rate=config.learning_rate,
        c_min=1400.0,
        c_max=3200.0,
        pml_size=10,
        gradient_smooth_sigma=config.gradient_smooth_sigma,
        loss_fn=config.loss_fn,
        mask=mask,
        parameterization=config.parameterization,
        siren_learning_rate=config.siren_lr,
        siren_pretrain_steps=config.siren_pretrain_steps,
        verbose=False
    )

    fwi_result = run_fwi(
        observed_data=observed,
        initial_velocity=c_init,
        density=rho,
        dx=dx,
        src_positions_grid=src_list[:n_data_src],
        sensor_positions_grid=pos_grid,
        source_signal=source_signal,
        dt=dt,
        t_end=t_end,
        config=internal_config,
    )

    # 5. Metrics
    c_recon = fwi_result.velocity
    brain_mask = (labels == 2) | (labels == 3)
    skull_mask = (labels == 7) | (labels == 11)

    def get_rmse(mask):
        n = jnp.sum(mask)
        if n == 0: return 0.0
        return float(jnp.sqrt(jnp.sum((c_recon - c_true) ** 2 * mask) / n))

    res = ExperimentResult(
        commit=get_commit_hash(),
        config=config,
        brain_rmse=get_rmse(brain_mask),
        skull_rmse=get_rmse(skull_mask),
        loss_history=[float(l) for l in fwi_result.loss_history],
        wall_time=time.time() - t0,
        dt=dt,
        n_t=n_samples
    )
    return res

def print_result(res: ExperimentResult):
    print(f"RESULT|brain_rmse={res.brain_rmse:.4f}|skull_rmse={res.skull_rmse:.4f}|loss={res.loss_history[-1]:.6f}|time={res.wall_time:.1f}|dt={res.dt:.4e}|nt={res.n_t}")

def log_result(res: ExperimentResult):
    results_file = Path("results.tsv")
    header = not results_file.exists()
    with open(results_file, "a") as f:
        if header:
            f.write("commit\tbrain_rmse\tskull_rmse\tfinal_loss\twall_time\tdt\tnt\tconfig\n")
        f.write(f"{res.commit}\t{res.brain_rmse}\t{res.skull_rmse}\t{res.loss_history[-1]}\t{res.wall_time}\t{res.dt}\t{res.n_t}\t{res.config}\n")

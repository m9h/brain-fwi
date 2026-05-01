#!/usr/bin/env python
"""Sweep SIREN architecture vs. θ-dimensionality and fit quality.

Question this answers: how small can the SIREN be and still pass the
``TestSIRENOnCoupledHeadPhantom::test_coupled_phantom_meets_validation_gate``
p95 < 2% validation gate? Smaller SIRENs = smaller θ vectors =
tractable NPE (Phase 2) and diffusion (Phase 3) training.

Why a script, not a test: this is a calibration run, not a contract
check. The test enforces ``fixed arch → gate met``. This sweeps
architectures and prints the Pareto front so we can choose a new
default consciously.

Usage::

    uv run --no-sync python scripts/theta_dim_sweep.py

Approx. 3–5 min on CPU (grid 24³ phantom, 400 pretrain steps per config).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from brain_fwi.inversion.param_field import (
    init_siren_from_velocity,
)
from brain_fwi.phantoms.augment import jittered_properties


C_MIN, C_MAX = 1400.0, 3200.0


# SIREN configurations worth probing. Ordered by theta-dim (ascending),
# so the sweep walks from "definitely too small" up to "current default".
CONFIGS = [
    # hidden, n_hidden, omega_0
    (32, 2, 30.0),
    (32, 3, 30.0),
    (32, 4, 30.0),
    (64, 2, 30.0),
    (64, 3, 30.0),
    (64, 4, 30.0),
    (128, 2, 30.0),
    (128, 3, 30.0),
    (128, 4, 30.0),
]


@dataclass
class Result:
    hidden: int
    n_hidden: int
    omega_0: float
    theta_dim: int
    p95_rel_err: float
    max_rel_err: float
    fit_seconds: float
    passes_gate: bool


def _build_mida_phantom(grid_shape=(32, 32, 32), dx=0.003):
    from brain_fwi.phantoms.mida import make_mida_phantom
    mida_path = "/data/datasets/MIDAv1-0/MIDA_v1.0/MIDA_v1_voxels/MIDA_v1.nii"
    labels, c, rho, alpha = make_mida_phantom(
        mida_path, grid_shape, dx, add_lesion=False, crop_cube=True,
    )
    return labels, c


def sweep(configs=CONFIGS, pretrain_steps: int = 400) -> List[Result]:
    labels, c_target = _build_mida_phantom()
    results: List[Result] = []

    print(f"Phantom: MIDA {tuple(c_target.shape)}, dx=3.0mm")
    print(f"Fit target: p95 rel-err < 2.0%")
    print(f"Pretrain steps per config: {pretrain_steps}")
    print()
    print(f"{'hidden':>6}  {'n_hid':>5}  {'θ-dim':>6}  {'fit s':>6}  "
          f"{'p95':>6}  {'max':>6}  gate")
    print("-" * 60)

    for hidden, n_hidden, omega_0 in configs:
        t0 = time.time()
        field = init_siren_from_velocity(
            c_target,
            c_min=C_MIN, c_max=C_MAX,
            hidden_dim=hidden, n_hidden=n_hidden, omega_0=omega_0,
            pretrain_steps=pretrain_steps, learning_rate=1e-3,
            key=jr.PRNGKey(0), verbose=False,
        )
        fit_time = time.time() - t0

        v = field.to_velocity(C_MIN, C_MAX)
        rel_err = jnp.abs(v - c_target) / c_target
        p95 = float(jnp.percentile(rel_err, 95))
        mx = float(jnp.max(rel_err))

        theta_leaves = jax.tree.leaves(
            eqx.filter(field.siren, eqx.is_inexact_array)
        )
        theta_dim = int(sum(l.size for l in theta_leaves))

        r = Result(
            hidden=hidden, n_hidden=n_hidden, omega_0=omega_0,
            theta_dim=theta_dim, p95_rel_err=p95, max_rel_err=mx,
            fit_seconds=fit_time, passes_gate=p95 < 0.02,
        )
        results.append(r)

        marker = "OK" if r.passes_gate else "--"
        print(f"{hidden:>6}  {n_hidden:>5}  {theta_dim:>6}  {fit_time:>5.1f}s  "
              f"{p95*100:>5.2f}%  {mx*100:>5.2f}%  {marker}")

    print()
    passing = [r for r in results if r.passes_gate]
    if passing:
        smallest = min(passing, key=lambda r: r.theta_dim)
        print(f"Smallest config passing gate: hidden={smallest.hidden}, "
              f"n_hidden={smallest.n_hidden} → θ-dim {smallest.theta_dim}")
        print(f"  (vs default hidden=128, n_hidden=3 → ~{[r.theta_dim for r in results if r.hidden==128][0]} dims)")
    else:
        print("WARNING: no config passes the gate — consider more pretrain steps "
              "or a different architecture (Fourier features, positional encoding).")

    return results


if __name__ == "__main__":
    sweep()

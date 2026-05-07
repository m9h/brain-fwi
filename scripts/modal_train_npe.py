"""Modal runner: NPE training on a real Phase-0 dataset.

Replaces the prior synthetic-data smoke. Reads a Phase-0 sharded
dataset off the ``brain-fwi-phase0`` volume, builds (theta, d) via
:func:`build_theta_d_matrix`, trains a :class:`ConditionalFlow`, runs
SBC, and serialises the trained flow weights to ``brain-fwi-npe-output``.

Usage::

    # Smoke (200 steps, half the dataset, ~$3 on A10G):
    modal run --detach scripts/modal_train_npe.py \\
        --version phase0_v2a --n-steps 200 --max-samples 64 \\
        --theta-dim-limit 256

    # Full (2000 steps, all 512 samples, ~$15-20):
    modal run --detach scripts/modal_train_npe.py \\
        --version phase0_v2a --n-steps 2000 --theta-dim-limit 256

Pull the trained flow back::

    modal volume get brain-fwi-npe-output \\
        /output/<version>/flow.eqx ./
"""

from __future__ import annotations

import time

import modal

app = modal.App("brain-fwi-npe")

GIT_BRANCH = "feature/parallel-modal-phase0"
CACHE_BUST = "2026-05-06-npe-sbc-fix"

DATA_VOL_NAME = "brain-fwi-phase0"
OUTPUT_VOL_NAME = "brain-fwi-npe-output"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .env({"BRAIN_FWI_CACHE_BUST": CACHE_BUST})
    .run_commands(
        f"git clone --depth 1 --branch {GIT_BRANCH} "
        f"https://github.com/m9h/brain-fwi.git /opt/brain-fwi",
        "cd /opt/brain-fwi && uv pip install --system -e '.[cuda12]'",
    )
)

data_vol = modal.Volume.from_name(DATA_VOL_NAME, create_if_missing=False)
output_vol = modal.Volume.from_name(OUTPUT_VOL_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",                # NPE training is small; A10G is plenty.
    timeout=4 * 60 * 60,       # 4 h max — well above expected 30-90 min.
    memory=32 * 1024,
    volumes={"/data": data_vol, "/output": output_vol},
    retries=2,
)
def train_npe_on_phase0(
    version: str = "phase0_v2a",
    phantom: str = "mida",
    grid_size: int = 96,
    n_steps: int = 2000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    test_fraction: float = 0.2,
    theta_dim_limit: int = 256,
    n_transforms: int = 6,
    nn_width: int = 64,
    nn_depth: int = 2,
    n_sbc_samples: int = 200,
    max_samples: int = 0,       # 0 = use all
    seed: int = 0,
):
    """Phase-0 → Phase-2 NPE training. Saves trained flow + metrics."""
    import json
    import os
    from pathlib import Path

    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np

    from brain_fwi.data import ShardedReader
    from brain_fwi.inference.dataprep import build_theta_d_matrix
    from brain_fwi.inference.flow import ConditionalFlow, train_npe
    from brain_fwi.inference.sbc import calibration_statistic, sbc_ranks

    print(f"JAX devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")

    data_root = Path(f"/data/output/{version}_{phantom}_{grid_size}/merged")
    if not (data_root / "manifest.json").exists():
        raise FileNotFoundError(f"Phase-0 dataset not found at {data_root}")

    out_dir = Path(f"/output/{version}_{phantom}_{grid_size}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading dataset:  {data_root}")
    reader = ShardedReader(data_root)
    sample_ids = list(reader.sample_ids)
    if max_samples > 0:
        sample_ids = sample_ids[:max_samples]
    print(f"  samples:        {len(sample_ids)}")

    print("Building (theta, d) matrix...")
    t0 = time.time()
    theta, d, ids = build_theta_d_matrix(reader[sid] for sid in sample_ids)
    print(f"  built in {time.time() - t0:.1f}s")
    print(f"  theta: {theta.shape}  d: {d.shape}")

    if theta_dim_limit and theta_dim_limit < theta.shape[1]:
        print(f"  truncating theta {theta.shape[1]} -> {theta_dim_limit} dims")
        theta = theta[:, :theta_dim_limit]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_test = max(2, int(round(len(ids) * test_fraction)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    theta_tr, d_tr = theta[train_idx], d[train_idx]
    theta_te, d_te = theta[test_idx], d[test_idx]
    print(f"  split: train={len(train_idx)}  test={len(test_idx)}")

    key = jr.PRNGKey(seed)
    flow_key, train_key, sbc_key = jr.split(key, 3)

    flow = ConditionalFlow(
        theta_dim=theta_tr.shape[1],
        d_dim=d_tr.shape[1],
        key=flow_key,
        n_transforms=n_transforms,
        nn_width=nn_width,
        nn_depth=nn_depth,
    )

    initial_nll = float(
        -jnp.mean(jax.vmap(flow.log_prob)(jnp.asarray(theta_te), jnp.asarray(d_te)))
    )
    print(f"\nInitial test NLL: {initial_nll:.4f}")

    print(f"\nTraining {n_steps} steps, batch={batch_size}...")
    t0 = time.time()
    trained, losses = train_npe(
        flow,
        theta=jnp.asarray(theta_tr),
        d=jnp.asarray(d_tr),
        key=train_key,
        n_steps=n_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        verbose=True,
    )
    train_wall = time.time() - t0

    final_nll = float(
        -jnp.mean(jax.vmap(trained.log_prob)(jnp.asarray(theta_te), jnp.asarray(d_te)))
    )
    print(f"\nFinal test NLL:   {final_nll:.4f}")
    print(f"Δ NLL:            {initial_nll - final_nll:+.4f} (lower is better)")
    print(f"Train wall:       {train_wall:.1f}s ({train_wall/n_steps*1000:.1f} ms/step)")

    # Save trained flow + initial metrics BEFORE SBC so a SBC failure
    # doesn't lose the training outcome (training is the expensive part).
    flow_path = out_dir / "flow.eqx"
    metrics_path = out_dir / "metrics.json"
    eqx.tree_serialise_leaves(str(flow_path), trained)
    metrics = {
        "version": version,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "theta_dim": int(theta_tr.shape[1]),
        "d_dim": int(d_tr.shape[1]),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "initial_nll": initial_nll,
        "final_nll": final_nll,
        "delta_nll": initial_nll - final_nll,
        "train_wall_s": train_wall,
        "sbc_p_value": None,
        "sbc_is_calibrated": None,
        "loss_curve": [float(x) for x in losses],
        "n_transforms": int(n_transforms),
        "nn_width": int(nn_width),
        "nn_depth": int(nn_depth),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    output_vol.commit()
    print(f"\nSaved flow:       {flow_path}")
    print(f"Saved metrics:    {metrics_path}")

    # SBC calibration on the held-out split. Failures here update the
    # already-saved metrics with sbc_error rather than blowing away
    # everything we just spent compute on.
    if n_sbc_samples > 0 and len(test_idx) >= 4:
        print(f"\nSBC ({n_sbc_samples} samples per held-out point)...")
        t0 = time.time()
        try:
            ranks = sbc_ranks(
                sampler=trained,
                theta_held_out=jnp.asarray(theta_te),
                d_held_out=jnp.asarray(d_te),
                n_posterior_samples=n_sbc_samples,
                key=sbc_key,
            )
            stats = calibration_statistic(np.asarray(ranks))
            metrics["sbc_p_value"] = float(stats["p_value"])
            metrics["sbc_is_calibrated"] = bool(stats["is_calibrated"])
            metrics["sbc_chi2"] = float(stats["chi2"])
            metrics["sbc_dof"] = int(stats["dof"])
            print(f"  SBC p-value:    {metrics['sbc_p_value']:.4f}  "
                  f"(well-calibrated if > 0.05)")
            print(f"  is_calibrated:  {metrics['sbc_is_calibrated']}")
            print(f"  SBC wall:       {time.time() - t0:.1f}s")
        except Exception as e:
            metrics["sbc_error"] = f"{type(e).__name__}: {e}"
            print(f"  SBC failed: {metrics['sbc_error']}")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        output_vol.commit()

    print(f"\nFinal metrics:    {metrics_path}")
    print(f"Pull back with:   modal volume get {OUTPUT_VOL_NAME} {out_dir} ./")
    return metrics


@app.local_entrypoint()
def main(
    version: str = "phase0_v2a",
    phantom: str = "mida",
    grid_size: int = 96,
    n_steps: int = 2000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    test_fraction: float = 0.2,
    theta_dim_limit: int = 256,
    n_transforms: int = 6,
    nn_width: int = 64,
    nn_depth: int = 2,
    n_sbc_samples: int = 200,
    max_samples: int = 0,
    seed: int = 0,
):
    print("=" * 70)
    print(f"  NPE training on Phase-0 {version}_{phantom}_{grid_size}")
    print(f"  Steps: {n_steps}  batch: {batch_size}  theta-dim cap: {theta_dim_limit}")
    print("=" * 70)
    metrics = train_npe_on_phase0.remote(
        version=version, phantom=phantom, grid_size=grid_size,
        n_steps=n_steps, batch_size=batch_size, learning_rate=learning_rate,
        test_fraction=test_fraction, theta_dim_limit=theta_dim_limit,
        n_transforms=n_transforms, nn_width=nn_width, nn_depth=nn_depth,
        n_sbc_samples=n_sbc_samples, max_samples=max_samples, seed=seed,
    )
    print(f"\nFinal NLL: {metrics['final_nll']:.4f}  "
          f"(Δ: {metrics['delta_nll']:+.4f})  "
          f"SBC p: {metrics.get('sbc_p_value')}")

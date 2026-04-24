"""Modal runner: NPE trace-noise sensitivity sweep.

Measures how NPE posterior calibration degrades as Gaussian noise is
added to the training-set traces. The knee of the calibration-vs-
noise curve sets the real accuracy bar for the Phase 4 surrogate —
an FNO whose trace error sits below the knee will not degrade NPE
downstream.

Evidence 9.2.2 from ``docs/design/phase4_fno_surrogate.md``.

Self-contained: generates a small synthetic-phantom Phase-0 dataset
inside the container, trains NPE at each noise level, runs SBC on
held-out, reports a summary table. No dependency on an externally
generated Phase-0 dataset — avoids volume-upload orchestration.

Budget: ~20 min on A10G (~$0.60). Seven noise levels × ~2 min each
+ one data-generation pass + overhead.

Usage::

    modal run --detach scripts/modal_npe_noise_sensitivity.py
"""

import modal

app = modal.App("brain-fwi-npe-noise")

GIT_BRANCH = "main"

# Phase-0 generation parameters for the internal dataset.
GRID_SIZE = 24
N_ELEMENTS = 16
N_SAMPLES = 60  # 48 train + 12 test
DX_M = 0.005
FREQ_HZ = 3.0e5
SIREN_PRETRAIN_STEPS = 300

# NPE training parameters.
NPE_N_STEPS = 500
NPE_BATCH_SIZE = 16
NPE_LR = 1e-3
NPE_N_TRANSFORMS = 5
THETA_DIM_LIMIT = 256  # keep NPE tractable under the current SIREN default
SBC_N_POSTERIOR_SAMPLES = 100

# Noise levels as fraction of per-trace peak amplitude. 0% = clean baseline.
NOISE_SIGMAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]

CACHE_BUST = "2026-04-24-npe-noise-v1"

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

results_vol = modal.Volume.from_name(
    "brain-fwi-validation", create_if_missing=True,
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/results": results_vol},
)
def run_sweep():
    import json
    import os
    import subprocess
    import time

    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    subprocess.run(["nvidia-smi", "-L"], check=True)

    # --- Generate fresh Phase-0 dataset ----------------------------
    dataset_path = "/results/noise_sensitivity_phase0"
    print(f"=" * 70)
    print(f"  Generating Phase-0 dataset ({N_SAMPLES} samples, {GRID_SIZE}^3)")
    print(f"=" * 70)

    t0 = time.time()
    subprocess.run([
        "python", "-u", "/opt/brain-fwi/scripts/gen_phase0.py",
        "--out", dataset_path,
        "--phantom", "synthetic",
        "--grid-size", str(GRID_SIZE),
        "--dx", str(DX_M),
        "--freq", str(FREQ_HZ),
        "--n-subjects", "1",
        "--n-augments", str(N_SAMPLES),
        "--n-elements", str(N_ELEMENTS),
        "--siren-pretrain-steps", str(SIREN_PRETRAIN_STEPS),
    ], check=True, cwd="/opt/brain-fwi")
    gen_time = time.time() - t0
    print(f"\n  dataset generation: {gen_time:.1f}s")
    results_vol.commit()

    # --- Load once, then loop over noise levels --------------------
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np

    from brain_fwi.data import ShardedReader
    from brain_fwi.inference.dataprep import build_theta_d_matrix
    from brain_fwi.inference.flow import ConditionalFlow, train_npe
    from brain_fwi.inference.sbc import calibration_statistic, sbc_ranks

    print("\n" + "=" * 70)
    print("  Loading dataset")
    print("=" * 70)
    reader = ShardedReader(dataset_path)
    theta, d, ids = build_theta_d_matrix(reader, d_method="max_abs")
    print(f"  theta={theta.shape}, d={d.shape}")

    # Truncate theta so the MAF is tractable.
    theta = theta[:, :THETA_DIM_LIMIT]
    print(f"  truncated theta to {theta.shape[1]} dims")

    # Fixed train/test split — same across all noise levels so comparison
    # is apples-to-apples.
    rng = np.random.default_rng(0)
    n = theta.shape[0]
    perm = rng.permutation(n)
    n_test = max(4, int(round(n * 0.2)))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    theta_tr, theta_te = theta[train_idx], theta[test_idx]
    d_tr_clean, d_te = d[train_idx], d[test_idx]

    # Per-trace peak amplitude used to scale the noise σ.
    d_peak = float(np.max(np.abs(d_tr_clean)))
    print(f"  train n={len(train_idx)}, test n={len(test_idx)}, d_peak={d_peak:.3e}")

    sweep_results = []
    key_base = jr.PRNGKey(42)

    for sigma_frac in NOISE_SIGMAS:
        print("\n" + "=" * 70)
        print(f"  σ = {sigma_frac * 100:.1f}% of trace peak")
        print("=" * 70)

        sigma_abs = sigma_frac * d_peak
        noise_rng = np.random.default_rng(int(sigma_frac * 1e6) + 1)
        d_tr_noisy = d_tr_clean + noise_rng.standard_normal(d_tr_clean.shape).astype(np.float32) * sigma_abs

        flow_key, train_key, sbc_key = jr.split(key_base, 3)
        key_base = jr.split(key_base)[0]

        flow = ConditionalFlow(
            theta_dim=theta_tr.shape[1],
            d_dim=d_tr_noisy.shape[1],
            key=flow_key,
            n_transforms=NPE_N_TRANSFORMS,
        )

        def _nll(f, theta_batch, d_batch):
            return float(-jnp.mean(jax.vmap(f.log_prob)(theta_batch, d_batch)))

        initial_nll = _nll(flow, jnp.asarray(theta_te), jnp.asarray(d_te))
        t0 = time.time()
        trained, losses = train_npe(
            flow,
            theta=jnp.asarray(theta_tr),
            d=jnp.asarray(d_tr_noisy),
            key=train_key,
            n_steps=NPE_N_STEPS,
            learning_rate=NPE_LR,
            batch_size=NPE_BATCH_SIZE,
            verbose=False,
        )
        train_time = time.time() - t0
        final_nll = _nll(trained, jnp.asarray(theta_te), jnp.asarray(d_te))

        ranks = np.asarray(sbc_ranks(
            trained,
            jnp.asarray(theta_te),
            jnp.asarray(d_te),
            n_posterior_samples=SBC_N_POSTERIOR_SAMPLES,
            key=sbc_key,
        ))
        # Keep SBC report concise — first 4 θ-dims.
        ranks = ranks[:, :4]
        stat = calibration_statistic(ranks, n_bins=min(10, max(2, len(test_idx) // 5)))

        result = {
            "sigma_frac": sigma_frac,
            "sigma_abs": sigma_abs,
            "initial_nll": initial_nll,
            "final_nll": final_nll,
            "nll_delta": initial_nll - final_nll,
            "train_time_s": round(train_time, 1),
            "sbc_p_value": stat["p_value"],
            "is_calibrated": stat["is_calibrated"],
            "per_dim_p": [round(d["p_value"], 3) for d in stat["per_dim"]],
        }
        sweep_results.append(result)
        print(f"  initial NLL: {initial_nll:.3f} → final: {final_nll:.3f}  "
              f"(Δ {initial_nll - final_nll:+.3f})")
        print(f"  SBC p-value: {stat['p_value']:.4f}  "
              f"calibrated={stat['is_calibrated']}")

    # --- Persist sweep results -------------------------------------
    out_path = "/results/npe_noise_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump({
            "dataset": dataset_path,
            "grid_size": GRID_SIZE,
            "n_elements": N_ELEMENTS,
            "n_samples": N_SAMPLES,
            "theta_dim": int(theta_tr.shape[1]),
            "d_dim": int(d_tr_clean.shape[1]),
            "d_peak": d_peak,
            "noise_sigmas": NOISE_SIGMAS,
            "npe": {
                "n_steps": NPE_N_STEPS,
                "batch_size": NPE_BATCH_SIZE,
                "learning_rate": NPE_LR,
                "n_transforms": NPE_N_TRANSFORMS,
            },
            "sbc_n_posterior_samples": SBC_N_POSTERIOR_SAMPLES,
            "results": sweep_results,
        }, f, indent=2, default=str)
    results_vol.commit()

    # --- Summary table ---------------------------------------------
    print("\n" + "=" * 70)
    print("  Noise-sensitivity sweep summary")
    print("=" * 70)
    print(f"  {'σ %':>5}  {'init NLL':>10}  {'final NLL':>10}  "
          f"{'ΔNLL':>8}  {'SBC p':>8}  calibrated")
    print("-" * 70)
    for r in sweep_results:
        print(f"  {r['sigma_frac']*100:>4.1f}%  "
              f"{r['initial_nll']:>9.3f}   {r['final_nll']:>9.3f}   "
              f"{r['nll_delta']:>+7.3f}  "
              f"{r['sbc_p_value']:>7.4f}  "
              f"{r['is_calibrated']}")
    print(f"\n  Written: {out_path}")


@app.local_entrypoint()
def main():
    run_sweep.remote()

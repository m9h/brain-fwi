#!/usr/bin/env python
"""Train NPE on a Phase-0 dataset and report calibration.

End-to-end Phase 0 → Phase 2 runner:

  1. Open a :class:`ShardedReader` on the dataset root.
  2. Build the ``(theta, d)`` training matrix via
     :func:`build_theta_d_matrix`.
  3. Train/test split (default 80/20).
  4. Train a :class:`ConditionalFlow` with :func:`train_npe`.
  5. Run SBC on the held-out split via :func:`sbc_ranks` +
     :func:`calibration_statistic`.
  6. Print final NLL + SBC p-value.

Requires flowjax — run on CI (Linux wheels available) or Modal.

Usage::

    uv run --no-sync python scripts/train_npe_on_phase0.py \\
        --data /tmp/phase0a_smoke \\
        --n-steps 500 --batch-size 16 \\
        --theta-dim-limit 256
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Guard flowjax at the top so the --help path works even when it's absent.
try:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from brain_fwi.inference.flow import ConditionalFlow, train_npe
    _FLOWJAX_OK = True
except Exception as exc:  # pragma: no cover - environment-dependent
    _FLOWJAX_IMPORT_ERROR = str(exc)
    _FLOWJAX_OK = False

from brain_fwi.data import ShardedReader
from brain_fwi.inference.dataprep import build_theta_d_matrix
from brain_fwi.inference.sbc import calibration_statistic, sbc_ranks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True,
                    help="Phase-0 dataset root (contains manifest.json + shards/)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write training metrics + SBC stats to this JSON file")
    ap.add_argument("--n-steps", type=int, default=500,
                    help="Adam steps for NPE training")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--n-transforms", type=int, default=5,
                    help="Number of MAF layers")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--theta-dim-limit", type=int, default=None,
                    help="Truncate theta to this many dims. Useful when "
                         "full SIREN weights (~50k dims) are too big for MAF.")
    ap.add_argument("--n-posterior-samples", type=int, default=100,
                    help="L in SBC — posterior samples per held-out pair")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not _FLOWJAX_OK:
        print(f"flowjax unavailable: {_FLOWJAX_IMPORT_ERROR}", file=sys.stderr)
        print("Run on CI / Linux / Modal where jaxlib wheels are available.",
              file=sys.stderr)
        return 2

    reader = ShardedReader(args.data)
    if len(reader) < 4:
        print(f"Dataset too small for train/test split: {len(reader)} samples")
        return 1

    print("=" * 70)
    print(f"  Phase 0 → Phase 2: NPE on {args.data}")
    print("=" * 70)
    print(f"  samples:      {len(reader)}")
    print(f"  jax devices:  {jax.devices()}")

    # --- Stack ----------------------------------------------------------
    t0 = time.time()
    theta, d, ids = build_theta_d_matrix(reader, d_method="max_abs")
    print(f"\n  theta matrix: {theta.shape}  d matrix: {d.shape}  "
          f"(built in {time.time() - t0:.1f}s)")

    if args.theta_dim_limit is not None and args.theta_dim_limit < theta.shape[1]:
        print(f"  truncating theta to first {args.theta_dim_limit} dims "
              f"(of {theta.shape[1]})")
        theta = theta[:, : args.theta_dim_limit]

    # --- Train/test split ----------------------------------------------
    rng = np.random.default_rng(args.seed)
    n = theta.shape[0]
    perm = rng.permutation(n)
    n_test = max(2, int(round(n * args.test_fraction)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    theta_tr, d_tr = theta[train_idx], d[train_idx]
    theta_te, d_te = theta[test_idx], d[test_idx]

    print(f"  split:        train={len(train_idx)}  test={len(test_idx)}")
    print(f"  theta-dim:    {theta_tr.shape[1]}")
    print(f"  d-dim:        {d_tr.shape[1]}")

    # --- Train flow ----------------------------------------------------
    key = jr.PRNGKey(args.seed)
    flow_key, train_key, sbc_key = jr.split(key, 3)

    flow = ConditionalFlow(
        theta_dim=theta_tr.shape[1],
        d_dim=d_tr.shape[1],
        key=flow_key,
        n_transforms=args.n_transforms,
    )

    def _nll(f, theta_batch, d_batch):
        log_probs = jax.vmap(f.log_prob)(theta_batch, d_batch)
        return float(-jnp.mean(log_probs))

    initial_nll = _nll(flow, jnp.asarray(theta_te), jnp.asarray(d_te))
    print(f"\n  initial test NLL: {initial_nll:.4f}")

    t0 = time.time()
    batch_size = min(args.batch_size, len(train_idx))
    trained, losses = train_npe(
        flow,
        theta=jnp.asarray(theta_tr),
        d=jnp.asarray(d_tr),
        key=train_key,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        batch_size=batch_size,
        verbose=True,
    )
    train_time = time.time() - t0
    final_nll = _nll(trained, jnp.asarray(theta_te), jnp.asarray(d_te))

    print(f"\n  training time: {train_time:.1f}s")
    print(f"  final test NLL: {final_nll:.4f}")
    print(f"  NLL delta:      {initial_nll - final_nll:+.4f}")

    # --- SBC -----------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  SBC calibration (L = {args.n_posterior_samples})")
    print("=" * 70)
    ranks = np.asarray(sbc_ranks(
        trained,
        jnp.asarray(theta_te),
        jnp.asarray(d_te),
        n_posterior_samples=args.n_posterior_samples,
        key=sbc_key,
    ))
    # Keep SBC compute bounded — compare per-dim stats only on first few dims
    max_dims_for_sbc = 8
    if ranks.shape[1] > max_dims_for_sbc:
        print(f"  limiting SBC to first {max_dims_for_sbc} dims "
              f"(of {ranks.shape[1]})")
        ranks = ranks[:, :max_dims_for_sbc]

    n_bins = min(10, max(2, len(test_idx) // 10))
    stat = calibration_statistic(ranks, n_bins=n_bins)
    print(f"  n_bins:         {n_bins}")
    print(f"  aggregate p:    {stat['p_value']:.4f}")
    print(f"  is_calibrated:  {stat['is_calibrated']}")

    # --- Persist -------------------------------------------------------
    if args.out is not None:
        report = {
            "dataset": str(args.data),
            "n_samples": int(n),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "theta_dim": int(theta_tr.shape[1]),
            "d_dim": int(d_tr.shape[1]),
            "n_steps": args.n_steps,
            "batch_size": batch_size,
            "learning_rate": args.learning_rate,
            "n_transforms": args.n_transforms,
            "initial_nll": initial_nll,
            "final_nll": final_nll,
            "nll_delta": initial_nll - final_nll,
            "train_time_s": train_time,
            "sbc": stat,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n  metrics written: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

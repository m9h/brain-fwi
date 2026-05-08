# DGX Spark — FNO surrogate training

The DGX Spark (GB10, ~120 GB unified memory) is the right place for
overnight FNO experiments where Modal H100 budget would otherwise be
the constraint. It can't fit the full production architecture
(`hidden=32, modes=12, depth=2` OOM'd with 150 GiB peak in earlier
runs), but it handles the smoke arch cleanly and is ideal for
ablations where we want to see the loss curve, not chase the §7.2
gate.

## When to use this vs Modal

| If you need... | Use |
|---|---|
| The full-arch model + §7.2 gate | Modal H100/H200 (`scripts/modal_train_fno_phase0.py`) |
| To test a hyperparam change cheaply | DGX (this guide) |
| Overnight runs without burning budget | DGX |
| §7.3 gradient-accuracy gate | Modal (DGX would also OOM) |

## Prerequisites on DGX

1. Phase-0 dataset locally. Either:
   - Sync from Modal volume:
     ```
     uvx modal volume get brain-fwi-phase0 \
         /output/phase0_v2a_mida_96/merged \
         /data/datasets/brain-fwi/phase0_v2a_mida_96
     ```
     (~36 GB, takes ~10 min on a reasonable connection)
   - Or regenerate locally with `scripts/gen_phase0.py` (slower)
2. `uv sync` in the repo root.
3. Sanity check: `uv run pytest tests/test_phase4_fno_pipeline.py -q`

## The "does batch averaging fix the bouncing?" overnight run

Production v3 (Modal H100, hidden=32, 1000 steps, no accumulation)
showed loss bouncing between 0.43 and 1.04 across the second half of
training even with cosine LR. Hypothesis: per-sample gradient noise
dominates late-training updates. Test:

```bash
uv run python scripts/train_fno_on_phase0.py \
    --data /data/datasets/brain-fwi/phase0_v2a_mida_96/merged \
    --out /data/outputs/fno_dgx_accum8/fno_surrogate \
    --n-steps 2000 \
    --hidden-channels 16 --num-modes 8 --depth 1 \
    --n-timesteps 1100 \
    --learning-rate 1e-3 --lr-schedule cosine --lr-alpha 0.01 \
    --accumulation-steps 8 \
    --skip-validation
```

Expected wall: a smoke-arch step at `accumulation_steps=8` is ~8× one
sample's forward+backward, so wall ≈ 8 × prod-v3-per-step ≈ 8 × 3.5 s
≈ 28 s/step. 2000 steps ≈ 16 h overnight. Tune down to 1000 steps if
that's tight.

Compare loss curves against `prod_v3` (Modal):

| Run | accum | min loss | min step | final loss |
|---|---|---|---|---|
| prod_v3 (Modal H100) | 1 | 0.43 | 822 | 1.04 |
| dgx_accum8 (this) | 8 | TBD | TBD | TBD |

The hypothesis is confirmed if `dgx_accum8` shows: (a) smoother loss
curve with smaller bounces, (b) `final - min` gap << prod_v3's 0.61,
(c) `min loss < 0.43`.

## Memory headroom

The smoke arch (`hidden=16, modes=8, depth=1`) at 96³ × 1100 timesteps
× 128 receivers fits well under 80 GB. With `accumulation_steps=8`
the peak is dominated by ONE sample's forward+backward (gradients are
summed, not stacked) — so memory is the same as `accumulation=1`.

If you want to try the production arch (`hidden=32, depth=2`) on DGX,
expect OOM. Stick to smoke arch on DGX, or use Modal.

## Saving + comparing

Outputs land at `<out>.eqx` (best-step weights) and `<out>.json`
(loss curve + config). To compare against the Modal v3 model:

```python
import json
v3 = json.load(open("/path/to/prod_v3/fno_surrogate.json"))
dgx = json.load(open("/data/outputs/fno_dgx_accum8/fno_surrogate.json"))
print("v3 min:", min(v3["loss_history"]),
      "  dgx min:", min(dgx["loss_history"]))
```

# Diffusion-Prior-Guided FWI for Brain Reconstruction

**Status:** prototype built + characterised (2D); 3D is the next phase.
**Code:** `inference/score_unet.py`, `inference/diffusion.py` (Phase-3 VPSDE/DSM/DDIM/DPS),
`phantoms/birnbaum.py`, `scripts/modal_train_unet_score.py`, `examples/dps_fwi_demo.py`.

## Why

Transcranial brain FWI splits into two problems with very different difficulty:

1. **The skull** (high contrast, ~2800 vs 1500 m/s). The 192³ MIDA run (job 984) recovered
   **0% of the skull** — diagnosed (this investigation) as a **gradient-conditioning** failure,
   not cycle-skipping: without the pseudo-Hessian, near-transducer voxels dominate the gradient
   and the deep skull starves. **Fix: `FWIConfig(precondition=True)`** — 2D skull recovery 17%→41%,
   and 3D (GB10 job 1629) 0%→4.8%. (AWI, initially hypothesised, was a dead end — it underperformed
   plain L2; see the AWI PR.) The skull is otherwise handled as a prior via MOFI pose alignment
   (`inversion/mofi.py`).

2. **The brain interior** (low contrast: GM/WM ~60 m/s, lesions ~100–200 m/s). Physics-only FWI —
   even with skull-prior + preconditioning + TV — produces acquisition-footprint ring artifacts,
   not the brain. This is where a **learned anatomical prior** is needed (the current research
   frontier; cf. the Imperial "diffusion-guided FWI" direction). This doc covers that piece.

## Pipeline

```
anatomy data (Birnbaum stroke slices)                MOFI skull pose
        │ phantoms.birnbaum.build_slice_dataset            │
        ▼                                                   ▼
  UNetScore (inference.score_unet)                  skull prior (held fixed)
        │ train_score_matching  (Modal GPU, ~80 s)         │
        ▼                                                   ▼
  learned brain prior  s_phi(x,t) ────────► DPS-FWI: x ← x − lr·∇_data L + λ·s_phi(x_ROI, t)
                                            (preconditioned, masked to intracranial ROI)
```

- **Score net:** conv U-Net (`UNetScore`) over the field, reusing the Phase-3 machinery
  (`VPSDE`, `train_score_matching`, `ddim_sample`). A shallow CNN gave fragmented, globally
  incoherent samples; the multi-scale U-Net gives coherent brains. (`surrogate/uno.py` is the
  spectral-operator cousin; conv U-Net is the right diffusion arch at fixed resolution.)
- **Training venue:** Modal GPU is fine for NN training (U-Net: 73–88 s on A10G, loss→77).
  Modal is **not** viable for the j-Wave FWI itself (pathologically slow — 3 timeouts); FWI runs
  on GB10/Slurm or CPU. Clean split: **Modal trains the prior, GB10/CPU runs the physics.**
- **DPS integration:** per-iteration score nudge on the intracranial ROI (a RED/score-prior
  regulariser; proper Tweedie/DDIM measurement-guided posterior sampling is a TODO).

## Results (2D, held-out real slices)

Interior RMSE (m/s), water start, skull held as prior:

| run | interior RMSE | note |
|---|---|---|
| no prior | → 98–110 | acquisition-footprint artifacts (always) |
| water init | 65–78 | (reference) |
| shallow-CNN prior | → 69 | first to beat init (subj4) |
| **U-Net prior** | **→ 67** | best; beats init in the good case |

**The learned prior reliably REGULARISES** — without it FWI blows up to artifacts; with it the
recon is pulled back to near/below the init, U-Net best. **But clean low-contrast lesion recovery
was NOT achieved in 2D**, consistently, across all variants (synthetic & real priors; shallow &
U-Net; full-head & cerebrum-ROI; tiny & large lesion). Two intrinsic 2D blockers, neither a
prior-quality issue:

1. **Data under-constrains** the low-contrast interior at this frequency/aperture — the prior
   can't invent contrast the data doesn't see.
2. **2D entangles brain with face** — in an axial slice the brain connects to the anterior through
   the skull-base/orbit, so even a largest-connected-component ROI can't isolate the cerebrum, and
   the face/sinus region dominates the error.

## 3D (the fix) — RESULT

The **same** pipeline applied in 3D on the GB10 removes both 2D blockers and, for the first time
in this project, the diffusion prior **improves over the starting model** (in every 2D variant it
could only regularise back toward init). 3D works because the intracranial volume is a clean
connected component (`birnbaum.cerebrum_volume_crop` / `build_volume_dataset` — 3D largest-CC, no
face entanglement) and the helmet gives full angular coverage.

Pipeline (all GB10; Modal trains the NN fine but cannot run j-Wave FWI):
1. `python scripts/build_birnbaum_3d_dataset.py` → 120 cerebrum volumes at 48³ (holds out 4 subjects).
2. `modal run scripts/modal_train_unet3d.py` → trains `UNet3DScore` (`inference/score_unet3d.py`).
3. `python scripts/validate_3d_prior.py` (CPU) → Tweedie denoise of a held-out real cerebrum shrinks
   ROI-RMSE at every noise level (+12% @t=0.05 … +45% @t=0.5): the *local* score is informative even
   though small-N unconditional samples look fragmented — and the local score is what DPS uses.
4. `python examples/dps_fwi_3d_demo.py` (GB10) → 3D DPS-FWI on held-out subj2 (48³/3mm, skull given
   at truth MOFI-style, helmet 16–20 fixed shots, bands 20-45/40-80 kHz, t_end 100 µs).

Held-out subj2 cerebrum-ROI RMSE (init = water, 65.3):

| run | cerebrum RMSE | brain-only RMSE | lesion (true 1660) |
|-----|---------------|-----------------|--------------------|
| no prior (12 it) | 65.3 → 68.1 (−4%) | — | 1531 |
| **3D prior (12 it)** | **65.3 → 54.8 (+16%)** | — | 1536 |
| no prior (16 it) | 65.3 → 74.6 (−14%) | 69.9 | 1538 |
| **3D prior (16 it)** | **65.3 → 61.0 (+7%)** | **55.7** | 1537 |

The prior beats no-prior by ~18–20% relative in both runs (robust); the gain *over init* depends on
iteration count (an iteration sweet spot ~12), because the DPS here is a **fixed-t nudge**, not a
measurement-guided posterior — more FWI iters accumulate data-misfit artifact the simple nudge can't
fully balance. The clearest, most robust win is brain-tissue cleanup (brain-only RMSE 69.9 → 55.7).

**Remaining gap — the focal lesion** (~1537 of 1660, ≈23% contrast) is flat across iteration counts ⇒
**resolution-limited, not prior-limited**: a ~10 mm lesion at 48³/3 mm under a 40–80 kHz band (λ 18–37 mm)
is under-resolved. Next steps: finer grid (96³/1.5 mm, ~150 kHz; needs a 96³ prior + ~8× FWI cost),
**proper DPS** (measurement-guided/annealed-t, replacing the fixed-t nudge — should lift the over-init
gain at any resolution), a richer prior on **SHARM** (196 heads, not yet on disk), and MOFI for the
skull pose (here given at truth). Prior + preconditioning + MOFI all compose unchanged into the 3D loop.

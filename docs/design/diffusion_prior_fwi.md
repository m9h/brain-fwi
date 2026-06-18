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

## Next phase: 3D

Apply the **same** pipeline in 3D on the GB10 (FWI is fast there). 3D removes both 2D blockers:
the intracranial volume is well-defined (no face entanglement) and the acquisition has full
angular coverage. Plus: proper DPS (measurement-guided posterior), a richer prior trained on
**SHARM** (196 cortical/cancellous heads — not yet on disk; SynthRAD2023/Birnbaum are), and
higher frequency. The prior + preconditioning + MOFI all compose unchanged into the 3D loop.

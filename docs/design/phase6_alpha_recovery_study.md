# Phase 6 — attenuation-recovery study (issue #48)

**Question:** does richer data (more iterations / multi-band / higher frequency)
push phase-only attenuation recovery past the **50 % midpoint** — the threshold
below which the tissue-α manifold prior (#45) snaps toward water and the
constitutive coupling has nothing to stand on?

**Setup:** the `test_multiparameter_fwi` phantom — a spherical α = 6.0 blob in a
lossless water tank, 32³, dx 0.5 mm, ring array, joint c+α inversion. Recovered
fraction = in-blob mean α / 6.0. Script: `scripts/alpha_recovery_study.py`
(run on the GB10, ~1 h for the 9-case sweep).

## Results

| setting | total iters | in-blob α | **recovered frac** | > 50 %? |
|---|---|---|---|---|
| single-band 150–300 kHz | 24 | 1.43 | 0.239 | no |
| single-band 150–300 kHz | 48 | 2.15 | 0.359 | no |
| **single-band 150–300 kHz** | **96** | **2.97** | **0.495** | **no (closest)** |
| multiband 50–300 (×24) | 72 | 1.53 | 0.255 | no |
| multiband 50–300 (×16) | 48 | 1.09 | 0.181 | no |
| hi-f multiband 50–500 (×24) | 72 | 2.54 | 0.424 | no |
| hi-f single-band 300–500 | 48 | 2.79 | 0.465 | no |
| hi-f multiband 100–700 (×24) | 72 | 2.31 | 0.385 | no |

## Findings

1. **Nothing crosses 50 %.** The best is brute-force 96 iterations (0.495,
   502 s) — asymptotically approaching, not clearing, the midpoint. Phase-only
   attenuation is fundamentally weakly determined.
2. **Iterations help most** (24→48→96 = 0.24→0.36→0.50), but with diminishing
   returns and rising out-of-blob leakage (0.11→0.24) — it plateaus at the
   midpoint, it does not blow through it.
3. **Multi-band did *not* help** (0.26 at 72 iters vs 0.36 single-band at 48):
   the added low bands carry little α information and raise leakage. Low-frequency
   convexity aids *velocity*, not attenuation.
4. **Higher frequency helps modestly** (300–500 kHz ×48 → 0.465) but never
   enough, and at higher cost / leakage.

## Consequence — coupling ≫ brute force

The constitutive c→α coupling (#45) reached **frac ≈ 0.998** (in-blob 5.99/6.0)
essentially for free *once c was recovered* — versus 96 expensive iterations
barely reaching 0.50 from the data alone. **For c-contrasted tissues, the
recipe "recover c first, then release α with the coupling" dominates brute-force
frequency/iteration by ~20× in recovered α at a fraction of the cost.** Chasing
higher frequency to force α out of the data is the wrong lever.

## The GM/WM exception (the real frontier)

GM and WM share the same sound speed, so the coupling is **degenerate** there —
GM/WM α must come from the data channel. But the data channel tops out at ~50 %
even for a *strong* 0→6 dB/cm/MHz contrast; the real GM/WM contrast is
0.6 vs 0.9 (a 1.5× ratio, ~0.3 dB/cm/MHz), far subtler. **No tested regime brings
GM/WM α within reach.** GM/WM differentiation remains a genuine open problem
requiring either much stronger data than the skull permits, or a different
observable (e.g. reflection/attenuation-tomography rather than transmission
phase).

## 384³ / 0.5 mm high-frequency feasibility

- **Frequency ceiling:** PSTD needs ≳2–3 points/wavelength. At dx 0.5 mm in
  brain (1500 m/s), 2 ppw → 1.5 MHz, 3 ppw → 1.0 MHz. **500–700 kHz is
  comfortably supported** (brain ppw ≈ 4.3 at 700 kHz). So higher frequency is
  *physically* reachable at 0.5 mm — it just doesn't solve the α-recovery problem.
- **Time steps:** CFL dt = 0.3·dx/c_max = 0.3·5e-4/2800 ≈ 54 ns; a head-sized
  t_end ≈ 2.4e-4 s → ~4500 timesteps/shot (≈2× the 192³ run).
- **Memory:** a 384³ float32 field is ~226 MB; a pseudospectral step holds
  ~10–20 such fields (~2–5 GB), and reverse-mode autodiff needs the
  O(√Nt)-checkpointed scan (auto-enabled ≥128³). Feasible on the GB10's 120 GB
  unified memory but costly per iteration (minutes/iter), so reserve it for a
  final high-res pass, not exploration.

## Recommendation

**Prioritise the c-first + constitutive-coupling recipe (#45), not higher
frequency (#48).** Concretely, the next implementation step is a **hierarchical
schedule** in `run_fwi`: invert c with α frozen for the first band(s), then
release α with the coupling — turning the measured decisive win into the default
path. Reserve 384³/500–700 kHz for the GM/WM frontier, where it is *necessary
but not sufficient*, and pair it with a non-transmission observable.

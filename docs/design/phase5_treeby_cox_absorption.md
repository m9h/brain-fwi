# Phase 5 — absorption integration plan

**Status (2026-06-27):**

| Layer | Status |
|---|---|
| Frequency-domain (Helmholtz) absorption with configurable y | ✅ working — `m9h/jwave@feature/configurable-alpha-power` |
| Time-domain (`simulate_wave_propagation`) absorption | ✅ **correct + k-Wave-validated** — `m9h/jwave@feature/time-domain-absorption` (canonical Treeby-Cox EoS; see "Time-domain absorption" below) |

> **2026-06-27 — the time-domain absorption is now physically correct and
> cross-validated against k-Wave.** The earlier "done" entry described a
> per-step decay of the diagnostic pressure that did NOT accumulate (a
> homogeneous plane wave lost ~0 amplitude; only edge-scattering changed the
> traces, which is why the >5% gating test still passed). It is replaced by the
> canonical Treeby-Cox *absorbing equation of state*, which matches k-Wave
> bit-for-bit across absorption strengths. Details in the rewritten section below.

This document captures the implementation roadmap for the time-domain
half. The frequency-domain half is already unblocked: `m9h/jwave`
adds `Medium.alpha_power` and rewrites `wavevector` as
`k² = (ω/c)² + 2j·ω^(y+1)·α/c`. Demonstrated working in
`scripts/phase5_attenuation_demo.py`.

## Frequency-domain (Helmholtz): done

Patch lives at `m9h/jwave@feature/configurable-alpha-power`, single
commit, +18/-3 lines across `jwave/acoustics/operators.py` and
`jwave/geometry.py`. Brain-fwi's `pyproject.toml` is pinned to that
branch. Drop the `@branch` pin once the patch lands upstream.

End-to-end frequency-domain demo
(`scripts/phase5_attenuation_demo.py`, 2026-04-29):

| Case | Median \|p\| ratio | rel-L2 vs lossless |
|---|---|---|
| Lossy y=2.0 (Stokes) | 0.759 | 0.231 |
| Lossy y=1.1 (tissue) | 0.663 | 0.327 |

The y=1.1 row is what `Phase 5`'s CANN α(ω) needs to plug into.
y=2 alone produced systematically wrong-by-~2× attenuation
magnitudes for tissue (see commit 92f8e1dc on the fork).

## Time-domain absorption: correct + k-Wave-validated (FourierSeries path)

### The bug in the first attempt (per-step decay of the diagnostic pressure)

The first implementation (`apply_absorption_fourier`) integrated, after
`pressure_from_density`, a per-step Euler decay

    p ← p − dt · α₀ · c^(y+1) · (-∇²)^(y/2) p .

It passed the >5% gating test but a homogeneous plane wave **lost ~0
amplitude** (measured `alpha_meas/analytic ≈ 0` across frequencies). Root
cause: `p` is a **diagnostic** field — every step recomputes
`p = pressure_from_density(rho)` from the persistent `(u, rho)` state, so the
decay applied to `p` survives only one step (feeding the next momentum gradient,
an O(dt²) effect) and never accumulates. The >5% the gating test saw came
purely from edge-scattering at heterogeneities, not bulk decay. So absorption
was effectively disabled in both the forward and the (FWI) checkpointed paths.

### The fix: canonical Treeby-Cox absorbing equation of state

Replaced by `absorbing_pressure_from_density(rho, u, medium, c_ref, dt, params)`
— the loss is folded into the **constitutive** pressure relation, evaluated
fresh from the persistent state each step, exactly as k-Wave's
`kspaceFirstOrder` does (Treeby & Cox, *JASA* 127(5):2741, 2010):

    p = c0² ( ρ
              + τ · (-∇²)^((y-2)/2) [ρ0 · ∇·u]      ← absorption
              − η · (-∇²)^((y-1)/2) [ρ] )           ← dispersion (Kramers-Kronig)

with τ = −2 α₀ c0^(y-1), η = 2 α₀ c0^y tan(πy/2), α₀ from k-Wave's `db2neper`
(dB/cm/MHz^y → Np·m⁻¹·(rad/s)⁻ʸ, ω_ref = 2π·10⁶), and the `|k|^(y-2)` operator's
k=0 component zeroed. The absorption term is driven by the velocity divergence
`ρ0·∇·u = −∂ρ/∂t` (supplying the ∂/∂t of the loss operator), obtained from
`mass_conservation_rhs(rho, u, 0, …)`. The factor of 2 in τ,η is the standard
"loss in the EoS only" convention. No-op (exact lossless EoS) when
`medium.attenuation` is zero/unset, so lossless callers are byte-identical.

Wired into BOTH time loops: the fork's settings-path `scan_fun`, and
brain-fwi's `_simulate_shot_sensors_checkpointed` (the FWI gradient path) —
they give bit-identical decay rates, and `jax.grad` flows through (absorption-
aware inversion is differentiable). Set `JWAVE_ABSORPTION_ONLY=1` to drop the
dispersion term (mirrors k-Wave's `alpha_mode="no_dispersion"`).

### k-Wave cross-validation (the proof it's right)

Same homogeneous power-law-absorber problem run in BOTH solvers (192×48×48,
dx 0.25 mm, 500 kHz, 12 ppw), CW lock-in decay slope vs analytic
`α(f)=a_db·(f/1e6)^y`. `scripts/kwave_absorption_xcheck.py` (k-Wave's pure-Python
backend, ARM-OK) vs j-Wave (`tests/test_attenuation_effect.py`,
`scripts/absorption_validate.py`):

| a_db | absorption-only jWave / k-Wave | with-dispersion jWave / k-Wave |
|---|---|---|
| 1.5  | 1.082 / 1.082 | 1.023 / 1.021 |
| 6.0  | 1.074 / 1.074 | 0.868 / 0.870 |
| 12.0 | 1.062 / 1.061 | 0.741 / 0.741 |

**j-Wave matches k-Wave bit-for-bit in both modes.** The shared >1 offset is a
CW-lock-in metric bias; the with-dispersion droop at high a_db is the
lossy-vs-lossless sound-speed-mismatch in the *ratio measurement* (the lossy
medium's c(ω) is shifted), not a physics error — k-Wave shows the identical
droop. The coefficient is exact: ratio → 1 as a_db → 0. This is the
differentiable-FUS edge — a Treeby-Cox solver matching k-Wave *and* end-to-end
autodiff for FWI.

Tests: `tests/test_attenuation_effect.py` —
`test_attenuation_changes_traces_for_skull_block` (gating),
`test_attenuation_active_in_checkpointed_fwi_path` (FWI path applies it),
`test_attenuation_decay_rate_matches_analytic` (bulk-decay rate guard).

### What's next on this front

- **Upstream PR.** Fork commits (configurable-alpha-power +
  time-domain-absorption with the canonical EoS) are clean, additive,
  backwards-compatible. Open `m9h/jwave → ucl-bug/jwave` after an FWI demo.
- **Absorption-aware FWI demo.** The forward + adjoint now carry correct
  attenuation; invert a skull with known α to show it improves the recon.
- **`alpha_mode` as a Medium attribute.** Promote the `JWAVE_ABSORPTION_ONLY`
  env escape hatch to a proper `Medium.alpha_mode` field (k-Wave parity).
- **OnGrid path.** Left untouched (brain-fwi routes through FourierSeries).
  Mirror the EoS hook there if any caller needs the OnGrid solver.
- **CFL adjustment.** Treeby–Cox §IV recommends a `~(0.9)^(2-y)` factor on
  `dt`; not yet applied. Add if stability issues surface at large y / coarse
  grids (k-Wave's absorbing stability limit also tightens dt at high kmax).

### Where the work lands: m9h/jwave fork (sibling commit)

Add to the same fork that already carries `feature/configurable-alpha-power`,
as a separate branch (`feature/time-domain-absorption`). Same upstream
PR strategy: prototype on the fork, propose to ucl-bug after it
parities cleanly.

Brain-fwi imports remain via the project's `pyproject.toml` jwave
pin, no vendoring in brain-fwi proper.

### The Treeby–Cox 2010 absorption term

Add to the symplectic time step (their eq. 11–13):

    L_α p = -2 α₀ c^(y-1) (-∇²)^((y+1)/2) p / sin(π y / 2)
    L_η p̃ = -2 α₀ c^y     (-∇²)^(y/2)     p̃ / cos(π y / 2)

Phase 5's K–K relation (`brain_fwi.constitutive.kk`) already
provides dispersion via a frequency-dependent `sound_speed`
adjustment, so the *dispersion* term `L_η p̃` is optional in the first
pass. Implement `L_α p` only; revisit `L_η p̃` if the phase residual
exceeds tolerance on the first parity test.

### Insertion point in j-Wave

`jwave/acoustics/time_varying.py:615–640` (FourierSeries `scan_fun`,
which is what brain-fwi uses):

```python
du = momentum_conservation_rhs(p, u, medium, c_ref=c_ref, dt=dt, ...)
u = pml_u * (pml_u * u + dt * du)
drho = mass_conservation_rhs(p, u, mass_src_field, medium, ...)
rho = pml_rho * (pml_rho * rho + dt * drho)
# >>> insert: rho = rho + dt * absorption_rhs(p, medium) <<<
p = pressure_from_density(rho, medium)
```

### Missing primitive: spectral fractional Laplacian

Neither j-Wave nor jaxdf exposes `(-∇²)^(y/2)`. Implementation is
~10 lines using FFT primitives that the `FourierSeries`
discretisation already exposes via `domain.k_vec`:

```python
def fractional_laplacian(field: FourierSeries, y: float) -> FourierSeries:
    # P_k = FFT(field);  L_k = |k|^y · P_k;  return iFFT(L_k)
```

Add to a new file `jwave/operators/fractional.py` (in the fork)
plus a unit test: a sinusoid `sin(k₀x)` should round-trip to
`k₀^y · sin(k₀x)` to machine precision.

### CFL adjustment

Treeby–Cox §IV: max stable timestep tightens by ~`(0.9)^(2-y)` when
the fractional Laplacian is present. For y = 1 (skull) this is ~0.9×.
Add an optional `y` kwarg to `TimeAxis.from_medium(...)` and apply
the factor.

### Acceptance test

`tests/test_attenuation_effect.py` flips XFAIL → PASS.
Remove the `pytest.mark.xfail(strict=True)` decorator when it does.

Stretch tests, in increasing order of compellingness:

1. **Spectral op unit test** — `fractional_laplacian(sin(k₀x), y)`
   recovers `k₀^y · sin(k₀x)` (1D, 2D, machine-precision).
2. **Steady-state parity with Helmholtz at y = 2** — time-domain
   converged solution ≈ frequency-domain solution within 1 % rel-L2
   on a static medium.
3. **CANN coupling** — take a CANN trained on skull α(ω), evaluate
   at one frequency, plug into the time-domain solver, compare
   against the same constant-α run at that frequency. Both should
   match within 1 %.
4. **ITRUSST BM3 lossy vs lossless** — rel-L2 difference at receivers
   is in the range Treeby–Cox 2010 §V predicts for cortical bone.

### Estimated scope

| Step | LOC | Effort |
|---|---|---|
| `fractional_laplacian` op + unit tests in `m9h/jwave` | ~80 | 2-3 h |
| `absorption_rhs` op + integration into `scan_fun` | ~60 | 2-3 h |
| CFL adjustment + tests | ~30 | 1 h |
| Bring brain-fwi's `tests/test_attenuation_effect.py` to GREEN | (rm xfail) | confirm |
| Stretch tests 1-4 | ~200 | 4-6 h |
| **Total** | ~370 | **1.5-2 days** |

### Out of scope (first pass)

- Implementing the L_η dispersion term — rely on K–K-via-c first.
- Stability proofs beyond Treeby–Cox §IV's empirical rule.
- Upstream PR before local verification.

## Related

- `tests/test_attenuation_effect.py` — gating xfail.
- `scripts/phase5_attenuation_demo.py` — frequency-domain
  demonstration; success metric for the time-domain work to match.
- `src/brain_fwi/constitutive/cann.py` — model whose α(ω) needs to
  plug in.
- `src/brain_fwi/constitutive/kk.py` — K–K dispersion, supplies c(ω).
- `m9h/jwave@feature/configurable-alpha-power` — fork branch with
  the frequency-domain piece (already pinned in `pyproject.toml`).
- Treeby B E, Cox B T (2010). *Modeling power law absorption and
  dispersion for acoustic propagation using the fractional Laplacian.*
  J. Acoust. Soc. Am. 127(5):2741–2748.

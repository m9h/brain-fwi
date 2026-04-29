# Phase 5 — absorption integration plan

**Status (2026-04-29):**

| Layer | Status |
|---|---|
| Frequency-domain (Helmholtz) absorption with configurable y | ✅ working — `m9h/jwave@feature/configurable-alpha-power` |
| Time-domain (`simulate_wave_propagation`) absorption | ✅ working — `m9h/jwave@feature/time-domain-absorption` |

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

## Time-domain absorption: done (FourierSeries path)

Gating test now PASSES:
```
tests/test_attenuation_effect.py::test_attenuation_changes_traces_for_skull_block PASSED
```
8 dB/cm/MHz^1.1 in a 4 mm skull block, 500 kHz Ricker → ~13 % rel-L2
between lossy and lossless 32³ traces. The `pytest.mark.xfail(strict=True)`
decorator has been removed.

Implementation: a single new function ``apply_absorption_fourier`` in
`jwave/acoustics/time_varying.py` that integrates the per-step
absorption term

    ∂p/∂t ⊃ -α₀ · c^(y+1) · (-∇²)^(y/2) p

via FFT, wired into the FourierSeries `scan_fun` symplectic step right
after `pressure_from_density`. No-op when `medium.attenuation` is
zero or unset, so existing lossless callers are unchanged.

This drops Treeby–Cox 2010 eq 11's `1/sin(π y / 2)` pole (so the
operator is defined at y = 2) and reshapes the spectral exponent
from `(y+1)/2` to `y/2`. The two forms agree on plane-wave decay
rate in homogeneous media and match the frequency-domain wavevector
op's `Im(k) = α(ω) = α₀ω^y` to leading order in α.

OnGrid path is left untouched; brain-fwi's `build_medium` routes
through FourierSeries by default.

### What's next on this front

- **Upstream PR.** Both fork commits (configurable-alpha-power +
  time-domain-absorption) are clean, additive, backwards-compatible.
  Open a PR `m9h/jwave → ucl-bug/jwave` once we've shipped one or
  two FWI experiments through this stack.
- **Optional dispersion term.** The companion `L_η ∂p/∂t` was
  intentionally skipped — phase-velocity dispersion is supplied at
  the constitutive layer via Kramers–Kronig and a frequency-dependent
  `sound_speed`. If that strategy proves insufficient (e.g. residual
  phase error in inversion), add `L_η`.
- **OnGrid path.** Left untouched. Add the same hook if any caller
  needs the OnGrid solver.
- **CFL adjustment.** Treeby–Cox §IV recommends a `~(0.9)^(2-y)`
  factor on `dt`. Not yet applied; `build_time_axis` ignores `y`.
  Add if stability issues surface at large y or coarse grids.

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

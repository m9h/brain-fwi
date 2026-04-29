"""Phase 5 — couple AttenuationCANN into the frequency-domain solver.

This is the second leg of the productive-direction story (the first
was ``phase5_attenuation_demo.py``: showing the fork's
``alpha_power`` knob actually changes Helmholtz output). Here we
plug the constitutive layer into the forward operator.

Procedure:
  1. Train an :class:`AttenuationCANN` on a cortical-bone power-law
     α(f) curve over the FWI band (100 kHz – 2 MHz), y = 1.5,
     α₀ = 4 dB/cm/MHz^1.5 at 1 MHz.
  2. Evaluate the trained CANN at 500 kHz → ``α_cann`` (Np/m).
  3. Compare to the analytic α(500 kHz) — sanity check that the CANN
     learnt the curve before we plug it in.
  4. Run two frequency-domain Helmholtz solves on the same ITRUSST-
     BM3-like skull plate at 500 kHz:
       (a) constant α₀ = 4 dB/cm/MHz^1.5 — the "we already know the
           tissue parameters" reference
       (b) CANN-derived α — equivalent dB/cm/MHz^1.5 coefficient
           reconstructed by inverting ``db2neper`` at 500 kHz so the
           wavevector op produces the same α_Np_per_m the CANN
           predicted
  5. Report rel-L2 between (a) and (b). It should be tiny — the CANN
     is the constitutive law's *interpolation* layer, so plugging it
     in at one frequency had better reproduce the reference.

Why this matters: multi-frequency FWI updates a single CANN per
tissue. Per-cell α coefficients can't carry frequency information.
This demo proves the wiring works at the frequency-domain layer
(time-domain wiring waits on Treeby–Cox; see the design doc).

Run:
    uv run python scripts/phase5_cann_coupling_demo.py
"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from brain_fwi.constitutive.cann import AttenuationCANN
from brain_fwi.constitutive.benchmark import (
    DB_PER_CM_TO_NP_PER_M,
    F_REF_HZ,
    alpha_curve,
    db_per_cm_per_mhz_to_np_per_m_at_1mhz,
)

# Same helpers as scripts/phase5_attenuation_demo.py — inlined here so
# this file is self-contained (scripts/ is not an importable package).
from importlib.util import module_from_spec, spec_from_file_location
import os
_demo_path = os.path.join(os.path.dirname(__file__), "phase5_attenuation_demo.py")
_spec = spec_from_file_location("phase5_attenuation_demo", _demo_path)
_demo = module_from_spec(_spec)
_spec.loader.exec_module(_demo)
build_phantom_2d = _demo.build_phantom_2d
run_helmholtz = _demo.run_helmholtz
rel_l2 = _demo.rel_l2


def train_cann_on_power_law(
    alpha0_db_cm_mhz: float,
    y: float,
    omega_grid: jnp.ndarray,
    *,
    n_steps: int = 2000,
    seed: int = 0,
) -> AttenuationCANN:
    """Fit an :class:`AttenuationCANN` to ``α(ω) = α₀·(f/f_ref)^y``."""
    alpha0_np = db_per_cm_per_mhz_to_np_per_m_at_1mhz(alpha0_db_cm_mhz)
    target = alpha_curve(alpha0_np, y, omega_grid)

    model = AttenuationCANN(
        n_basis=2,
        key=jr.PRNGKey(seed),
        omega_scale=float(omega_grid.max()),
        alpha_scale=float(target.max()),
    )
    opt = optax.adam(1e-2)
    state = opt.init(model)

    @eqx.filter_jit
    def step(m, st):
        def loss_fn(mm):
            return jnp.mean((mm(omega_grid) - target) ** 2)
        loss, g = eqx.filter_value_and_grad(loss_fn)(m)
        upd, st = opt.update(g, st)
        return eqx.apply_updates(m, upd), st, loss

    for _ in range(n_steps):
        model, state, _ = step(model, state)

    return model, target


def np_per_m_to_db_per_cm_per_mhz_y(
    alpha_np_per_m_at_freq: float,
    freq_hz: float,
    y: float,
) -> float:
    """Invert j-Wave's ``db2neper(α, y)`` so the wavevector op produces
    the requested ``α_Np`` at the requested frequency.

    j-Wave's ``wavevector`` does:
        α_Np_per_m_at_freq = α_input · (freq_MHz)^y · 100 / 8.686
    so:
        α_input = α_Np_per_m_at_freq · 8.686 / (100 · (freq_MHz)^y)
    """
    f_mhz = freq_hz / 1e6
    return float(alpha_np_per_m_at_freq / (DB_PER_CM_TO_NP_PER_M * f_mhz ** y))


def main():
    freq_hz = 500e3
    y_skull = 1.5
    alpha0_db = 4.0  # dB/cm/MHz^y at 1 MHz
    alpha0_dip = 8.0

    band_hz = jnp.linspace(1e5, 2e6, 64)
    omega_band = 2.0 * jnp.pi * band_hz

    print(f"Training CANN on cortical-bone α(ω) over "
          f"{band_hz.min()/1e3:.0f}-{band_hz.max()/1e3:.0f} kHz "
          f"(y = {y_skull}, α₀ = {alpha0_db} dB/cm/MHz^{y_skull})")
    t0 = time.time()
    cann_cort, target_cort = train_cann_on_power_law(
        alpha0_db, y_skull, omega_band, n_steps=2000,
    )
    cann_dip, target_dip = train_cann_on_power_law(
        alpha0_dip, y_skull, omega_band, n_steps=2000,
    )
    print(f"  CANN training took {time.time()-t0:.1f}s")

    # CANN fit quality on the band
    pred_cort = cann_cort(omega_band)
    pred_dip = cann_dip(omega_band)
    rel_cort = float(jnp.linalg.norm(pred_cort - target_cort) /
                     jnp.linalg.norm(target_cort))
    rel_dip = float(jnp.linalg.norm(pred_dip - target_dip) /
                    jnp.linalg.norm(target_dip))
    print(f"  CANN cortical-bone fit rel-L2 over band: {rel_cort:.3e}")
    print(f"  CANN diploe         fit rel-L2 over band: {rel_dip:.3e}")

    # Evaluate CANN at the demo frequency
    omega_demo = 2.0 * jnp.pi * freq_hz
    alpha_cann_cort_np = float(cann_cort(jnp.array(omega_demo)))
    alpha_cann_dip_np = float(cann_dip(jnp.array(omega_demo)))
    alpha_analytic_cort = float(alpha_curve(
        db_per_cm_per_mhz_to_np_per_m_at_1mhz(alpha0_db), y_skull,
        jnp.array(omega_demo),
    ))
    alpha_analytic_dip = float(alpha_curve(
        db_per_cm_per_mhz_to_np_per_m_at_1mhz(alpha0_dip), y_skull,
        jnp.array(omega_demo),
    ))
    print(f"\nAt {freq_hz/1e3:.0f} kHz:")
    print(f"  cortical α: CANN={alpha_cann_cort_np:7.2f} Np/m  "
          f"analytic={alpha_analytic_cort:7.2f} Np/m  "
          f"(diff {alpha_cann_cort_np - alpha_analytic_cort:+.2e})")
    print(f"  diploe   α: CANN={alpha_cann_dip_np:7.2f} Np/m  "
          f"analytic={alpha_analytic_dip:7.2f} Np/m  "
          f"(diff {alpha_cann_dip_np - alpha_analytic_dip:+.2e})")

    # Convert CANN α back to dB/cm/MHz^y units for the wavevector
    a_cort_db_cann = np_per_m_to_db_per_cm_per_mhz_y(
        alpha_cann_cort_np, freq_hz, y_skull,
    )
    a_dip_db_cann = np_per_m_to_db_per_cm_per_mhz_y(
        alpha_cann_dip_np, freq_hz, y_skull,
    )
    print(f"  CANN-equivalent α_input for wavevector: "
          f"cort={a_cort_db_cann:.3f}, dip={a_dip_db_cann:.3f} dB/cm/MHz^{y_skull}")

    # Helmholtz setup matches phase5_attenuation_demo.py
    dx = 5e-4
    grid_shape = (160, 160)
    pml = 16
    src_xz = (grid_shape[0] // 2, pml + int(round(0.010 / dx)))
    recv_x_mm = np.arange(20, 61, 5)
    recv_x = (recv_x_mm * 1e-3 / dx).astype(int)
    recv_z = grid_shape[1] - pml - int(round(0.010 / dx))
    recv_z_arr = np.full_like(recv_x, recv_z)
    recv = (recv_x, recv_z_arr)

    cases = [
        ("reference (analytic α)",   alpha0_db,         alpha0_dip),
        ("CANN-derived α",           a_cort_db_cann,    a_dip_db_cann),
    ]

    results = {}
    for name, a_cort, a_dip in cases:
        print(f"\n--- {name} (α_cort={a_cort:.3f}, α_dip={a_dip:.3f}) ---")
        c, rho, alpha = build_phantom_2d(
            grid_shape, dx, pml,
            alpha_cortical=a_cort, alpha_diploe=a_dip,
        )
        t0 = time.time()
        p = run_helmholtz(
            c, rho, alpha, dx, freq_hz, src_xz, recv,
            pml=pml, alpha_power=y_skull,
        )
        amp = np.abs(p)
        print(f"  ran in {time.time()-t0:.1f}s, |p| median = {np.median(amp):.3e}")
        results[name] = p

    rl2 = rel_l2(results["CANN-derived α"], results["reference (analytic α)"])
    amp_ratio = float(np.median(np.abs(results["CANN-derived α"])) /
                       np.median(np.abs(results["reference (analytic α)"])))
    print("\n" + "=" * 70)
    print(f"CANN-derived α vs reference α at {freq_hz/1e3:.0f} kHz")
    print("-" * 70)
    print(f"  rel-L2 (Helmholtz outputs):      {rl2:.3e}")
    print(f"  median |p| ratio:                {amp_ratio:.4f}")
    print("=" * 70)
    if rl2 < 0.01:
        print("→ Wiring works: CANN evaluated at this frequency reproduces the")
        print("  reference Helmholtz solve to within 1 % rel-L2. Phase 5's")
        print("  constitutive layer can drive the frequency-domain forward op.")
    else:
        print(f"→ rel-L2 {rl2:.3e} is larger than expected. Likely cause: CANN")
        print("  fit residual at this frequency (rel band fit was "
              f"{rel_cort:.2e}). Increase n_steps or revisit unit conversion.")


if __name__ == "__main__":
    main()

"""TDD red-green spec for the Phase 5 CANN.

One behaviour per test. Each was written failing first.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest


def test_attenuation_cann_is_nonnegative_for_arbitrary_omega():
    """Energy dissipation: α(ω) ≥ 0 for any frequency, any params."""
    from brain_fwi.constitutive import AttenuationCANN

    model = AttenuationCANN(n_basis=3, key=jr.PRNGKey(0))
    omega = jnp.linspace(-5e6, 5e6, 64)
    alpha = model(omega)
    assert alpha.shape == omega.shape
    assert jnp.all(alpha >= 0.0), f"got negative α: min={float(alpha.min())}"


def test_attenuation_cann_vanishes_at_dc():
    """DC limit: α(0) = 0. Tissues do not attenuate static fields."""
    from brain_fwi.constitutive import AttenuationCANN

    for seed in range(5):
        model = AttenuationCANN(n_basis=3, key=jr.PRNGKey(seed))
        alpha0 = model(jnp.zeros(()))
        assert float(alpha0) == pytest.approx(0.0, abs=1e-12)


def test_attenuation_cann_monotone_in_abs_omega():
    """α grows with |ω| since exponent y ∈ (1, 2) is strictly positive."""
    from brain_fwi.constitutive import AttenuationCANN

    model = AttenuationCANN(n_basis=2, key=jr.PRNGKey(7))
    omega = jnp.linspace(1e3, 1e7, 128)
    alpha = model(omega)
    diffs = jnp.diff(alpha)
    assert jnp.all(diffs > 0), "α(ω) is not strictly monotone in |ω|"


def test_attenuation_cann_recovers_synthetic_power_law():
    """End-to-end trainability: SGD recovers α₀ |ω|^y₀ to within 5% RMSE.

    This is the real spec — the architecture is only useful if a
    handful of gradient steps can fit a typical tissue power law.
    Soft-tissue at MHz: y ≈ 1.1, α₀ ≈ 0.05 Np/m/(rad/s)^y order.
    """
    import optax

    from brain_fwi.constitutive import AttenuationCANN

    omega = jnp.linspace(1e5, 6e6, 256)
    alpha_true = 0.05 * jnp.abs(omega) ** 1.1

    model = AttenuationCANN(
        n_basis=2, key=jr.PRNGKey(0),
        omega_scale=float(omega.max()),
        alpha_scale=float(alpha_true.max()),
    )
    opt = optax.adam(1e-2)
    state = opt.init(model)

    def loss_fn(m):
        return jnp.mean((m(omega) - alpha_true) ** 2)

    import equinox as eqx
    for _ in range(2000):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, state = opt.update(grads, state)
        model = eqx.apply_updates(model, updates)

    alpha_pred = model(omega)
    rmse = float(jnp.sqrt(jnp.mean((alpha_pred - alpha_true) ** 2)))
    rel = rmse / float(jnp.sqrt(jnp.mean(alpha_true ** 2)))
    assert rel < 0.05, f"relative RMSE {rel:.3f} exceeds 5% gate"


def test_kk_zero_alpha_gives_no_dispersion():
    """Zero attenuation ⇒ no dispersion: c(ω) = c_ref ∀ ω."""
    from brain_fwi.constitutive import kramers_kronig_dispersion

    omega = jnp.linspace(1e5, 6e6, 256)
    alpha = jnp.zeros_like(omega)
    c_ref = 1500.0
    omega_ref = float(omega[len(omega) // 2])
    c = kramers_kronig_dispersion(alpha, omega, omega_ref=omega_ref, c_ref=c_ref)
    assert c.shape == omega.shape
    assert jnp.allclose(c, c_ref, atol=1e-6), (
        f"non-zero dispersion from zero α: max |Δc| = "
        f"{float(jnp.max(jnp.abs(c - c_ref))):.3e}"
    )


def test_kk_power_law_alpha_gives_positive_dispersion():
    """Power-law α(ω) ∝ ω^y with y∈(1,2) ⇒ c grows with ω.

    This is the standard tissue-physics expectation: at higher freq
    the medium is faster (anomalous-but-causal dispersion), and the
    deviation Δc(ω) = c(ω) − c_ref is monotone in ω above ω_ref.
    """
    from brain_fwi.constitutive import kramers_kronig_dispersion

    omega = jnp.linspace(1e5, 6e6, 512)
    alpha = 1e-7 * jnp.abs(omega) ** 1.1
    c_ref = 1500.0
    omega_ref = float(omega[0])
    c = kramers_kronig_dispersion(alpha, omega, omega_ref=omega_ref, c_ref=c_ref)
    delta_c = c - c_ref
    assert jnp.all(jnp.diff(delta_c) > 0), (
        "Δc(ω) is not monotone above ω_ref"
    )
    assert float(delta_c[-1]) > 0, (
        f"Δc at top freq should be positive, got {float(delta_c[-1])}"
    )


def test_kk_consistency_loss_zero_on_consistent_pair():
    """If c is the K-K image of α, the consistency loss is zero."""
    from brain_fwi.constitutive import (
        kk_consistency_loss,
        kramers_kronig_dispersion,
    )

    omega = jnp.linspace(1e5, 6e6, 256)
    alpha = 1e-7 * jnp.abs(omega) ** 1.1
    c_ref, omega_ref = 1500.0, float(omega[0])
    c_kk = kramers_kronig_dispersion(alpha, omega, omega_ref=omega_ref, c_ref=c_ref)
    loss = kk_consistency_loss(
        alpha, c_kk, omega, omega_ref=omega_ref, c_ref=c_ref,
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-12)


def test_kk_consistency_loss_positive_on_inconsistent_pair():
    """A constant c plus a non-zero α is non-causal ⇒ positive loss."""
    from brain_fwi.constitutive import kk_consistency_loss

    omega = jnp.linspace(1e5, 6e6, 256)
    alpha = 1e-7 * jnp.abs(omega) ** 1.1
    c_ref, omega_ref = 1500.0, float(omega[0])
    c_constant = jnp.full_like(omega, c_ref)
    loss = kk_consistency_loss(
        alpha, c_constant, omega, omega_ref=omega_ref, c_ref=c_ref,
    )
    assert float(loss) > 0.0


def test_fit_tissue_alpha_curves_full_table_under_10pct():
    """All ITRUSST tissues fit under 10% rel-RMSE; median under 2%.

    Regression guard for the full-table benchmark. The 10% gate
    accommodates blood_vessels (the empirically-hardest tissue at
    8.5%) while still flagging architectural regressions. Median
    gate enforces the typical-case quality (most fits are <1%).
    Water/air are excluded from the relative-RMSE gate (α≡0) but
    must still satisfy the absolute residual gate.
    """
    import numpy as np

    from brain_fwi.constitutive.benchmark import fit_tissue_alpha_curves

    omega = jnp.linspace(2 * jnp.pi * 5e4, 2 * jnp.pi * 6e6, 256)
    results = fit_tissue_alpha_curves(
        omega=omega, n_steps=2000, key=jr.PRNGKey(0),
    )
    assert len(results) >= 13

    rel_rmses = []
    for tissue, r in results.items():
        if r.alpha0_np_per_m == 0.0:
            assert r.recon_rms_np_per_m < 0.1, (
                f"{tissue}: residual {r.recon_rms_np_per_m:.3f} Np/m"
            )
            continue
        rel_rmses.append(r.rel_rmse)
        assert r.rel_rmse < 0.10, (
            f"{tissue}: rel-RMSE {r.rel_rmse:.3f} > 10%"
        )

    median = float(np.median(rel_rmses))
    assert median < 0.02, f"median rel-RMSE {median:.4f} > 2%"


def test_fit_tissue_alpha_curves_subset_under_5pct():
    """Per-tissue CANN fits ITRUSST α(ω) to <5% relative RMSE on a
    representative subset (zero/low/high attenuation regimes)."""
    from brain_fwi.constitutive.benchmark import fit_tissue_alpha_curves

    omega = jnp.linspace(2 * jnp.pi * 5e4, 2 * jnp.pi * 6e6, 256)
    results = fit_tissue_alpha_curves(
        tissues=["water", "csf", "cortical_bone"],
        omega=omega,
        n_steps=2000,
        key=jr.PRNGKey(0),
    )
    assert set(results.keys()) == {"water", "csf", "cortical_bone"}
    for tissue, r in results.items():
        # water has α ≡ 0 — assert residual is small in absolute Np/m
        # (softplus tail leaves a tiny residual; 0.1 Np/m is 100× below
        # the smallest attenuating tissue, so still physically zero)
        if tissue == "water":
            assert r.recon_rms_np_per_m < 0.1
            continue
        assert r.rel_rmse < 0.05, (
            f"{tissue}: rel-RMSE {r.rel_rmse:.4f} > 5%"
        )

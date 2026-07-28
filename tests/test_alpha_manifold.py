"""Phase 6.2 — CANN-derived tissue-α manifold prior for attenuation inversion.

Phase-only FWI constrains α weakly, so a free per-voxel α smears. The fix
(design doc §3.2): tie α to the low-dimensional set of *physical tissue α
values* (the CANN α(ω) manifold, in coefficient form) — "invert a few
coefficients + a spatial field" instead of a noisy image. This is the red-green
spec for that manifold prior.

One behaviour per test. Each written failing first.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest


def test_tissue_alpha_coefficients_are_physical():
    """The archetype set carries the real ITRUSST α levels (dB/cm/MHz):
    ~0 for water/CSF, 0.6 for brain, 4.0 cortical skull, 8.0 trabecular."""
    from brain_fwi.constitutive import tissue_alpha_coefficients

    arch = tissue_alpha_coefficients()
    assert arch["brain"] == pytest.approx(0.6)
    assert arch["cortical_bone"] == pytest.approx(4.0)
    assert arch["trabecular_bone"] == pytest.approx(8.0)
    assert arch["water"] == pytest.approx(0.0, abs=1e-2)


def test_manifold_proximal_fixes_archetype_values():
    """A value sitting on an archetype is a fixed point of the proximal snap."""
    from brain_fwi.constitutive import manifold_proximal

    archetypes = jnp.array([0.0, 0.6, 4.0, 8.0])
    x = jnp.array([0.0, 0.6, 4.0, 8.0])
    out = manifold_proximal(x, archetypes, beta=0.5)
    assert jnp.allclose(out, x, atol=1e-6)


def test_manifold_proximal_moves_toward_nearest_archetype():
    """Snap moves x by exactly beta·(nearest_archetype − x)."""
    from brain_fwi.constitutive import manifold_proximal

    archetypes = jnp.array([0.0, 4.0])
    x = jnp.array([3.2, 0.5])           # nearest: 4.0, 0.0
    out = manifold_proximal(x, archetypes, beta=0.5)
    expected = x + 0.5 * (jnp.array([4.0, 0.0]) - x)
    assert jnp.allclose(out, expected, atol=1e-6)


def test_manifold_prior_sharpens_an_adequately_recovered_field():
    """On a field whose anomaly is recovered ABOVE the archetype midpoint,
    repeated proximal snaps sharpen it: lower RMSE to truth, tighter within-
    region spread (structure, not smear)."""
    from brain_fwi.constitutive import manifold_proximal

    rng = np.random.default_rng(0)
    n = 24
    zz, yy, xx = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    r = np.sqrt((zz - n / 2) ** 2 + (yy - n / 2) ** 2 + (xx - n / 2) ** 2)
    blob = r <= n * 0.2
    truth = np.where(blob, 4.0, 0.0).astype(np.float32)         # cortical archetype
    # A recovered field: blob near-but-not-at 4, background near 0, both noisy.
    rec = truth + rng.normal(0, 0.6, truth.shape).astype(np.float32)
    rec = np.clip(rec, 0, None)

    archetypes = jnp.array([0.0, 4.0])
    a = jnp.asarray(rec)
    for _ in range(6):
        a = manifold_proximal(a, archetypes, beta=0.4)
    a = np.asarray(a)

    rmse0 = float(np.sqrt(np.mean((rec - truth) ** 2)))
    rmse1 = float(np.sqrt(np.mean((a - truth) ** 2)))
    spread0 = float(rec[blob].std())
    spread1 = float(a[blob].std())
    assert rmse1 < rmse0, f"manifold prior did not reduce RMSE: {rmse0:.3f} -> {rmse1:.3f}"
    assert spread1 < spread0, f"in-blob spread not tightened: {spread0:.3f} -> {spread1:.3f}"


def test_manifold_prior_erases_weak_subthreshold_signal_honest_limit():
    """Documented limitation: a weak, sub-midpoint α is pulled to water, not up.
    This is WHY the prior must be RAMPED LATE (after the data has grown α past
    the archetype midpoint) — encoded as a guard so the limit can't silently
    regress into a false 'it always helps' claim."""
    from brain_fwi.constitutive import manifold_proximal

    archetypes = jnp.array([0.0, 4.0])   # midpoint 2.0
    weak = jnp.array([1.2])              # below midpoint
    out = manifold_proximal(weak, archetypes, beta=0.5)
    assert float(out[0]) < float(weak[0]), "sub-threshold signal should snap toward water"


def test_fwi_config_carries_manifold_prior_and_defaults_off():
    """Wiring: the prior fields plumb through FWIConfig and are OFF by default
    (weight 0 ⇒ free-voxel α, byte-identical to Phase 6.1)."""
    from brain_fwi.inversion.fwi import FWIConfig
    from brain_fwi.constitutive import tissue_alpha_coefficients

    assert FWIConfig().attenuation_prior_weight == 0.0
    assert FWIConfig().attenuation_archetypes is None

    arch = jnp.asarray(sorted(tissue_alpha_coefficients().values()))
    cfg = FWIConfig(invert_attenuation=True, attenuation_archetypes=arch,
                    attenuation_prior_weight=0.3, attenuation_prior_ramp=0.6)
    assert cfg.attenuation_archetypes.shape[0] == 4
    assert cfg.attenuation_prior_ramp == 0.6

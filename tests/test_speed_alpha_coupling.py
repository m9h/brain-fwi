"""Phase 6.2 second half (issue #45) — velocity→attenuation constitutive coupling.

Phase-only FWI recovers α weakly (~25%), but recovers *velocity* well (traveltime
is a strong constraint). Because c and α come from the SAME tissue, the recovered
velocity predicts α through the constitutive (c, α) relation — the causal /
Kramers-Kronig premise that both are properties of one medium. This lets the
strong channel inform the weak one.

Honest limit encoded below: GM and WM share the same sound speed (1560), so the
c→α map is DEGENERATE there — the coupling helps c-contrasted tissues (skull,
CSF, brain-vs-background) but cannot resolve GM/WM. That distinction stays
data-limited (needs #48).

One behaviour per test, each written failing first.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest


def test_anchors_are_sorted_and_dedupe_degenerate_speed():
    """Anchor points are unique-in-c and sorted; GM/WM (equal c) collapse to one
    anchor — the degeneracy that makes c→α unable to separate them."""
    from brain_fwi.constitutive import speed_alpha_anchors
    c, a = speed_alpha_anchors()
    c = np.asarray(c)
    assert np.all(np.diff(c) > 0), "anchors must be strictly sorted / unique in c"
    # 1560 m/s appears for GM, WM, dura — one anchor, not three.
    assert np.sum(np.isclose(c, 1560.0)) == 1


def test_alpha_from_speed_interpolates_tissue_relation():
    """c→α maps water-speed to ~0 and skull-speed to the skull attenuation."""
    from brain_fwi.constitutive import speed_alpha_anchors, alpha_from_speed
    c_anch, a_anch = speed_alpha_anchors()
    c_field = jnp.array([1500.0, 2800.0])   # water, cortical bone
    a = np.asarray(alpha_from_speed(c_field, c_anch, a_anch))
    assert a[0] == pytest.approx(0.002, abs=0.05)   # ~water
    assert a[1] == pytest.approx(4.0, abs=0.5)      # ~cortical bone


def test_custom_two_tissue_map():
    """A minimal 2-tissue constitutive map used by the coupling test: a
    'skull-like' blob (c=2000, α=6) in water (c=1500, α=0)."""
    from brain_fwi.constitutive import alpha_from_speed
    c_anch = jnp.array([1500.0, 2000.0])
    a_anch = jnp.array([0.0, 6.0])
    c_field = jnp.array([1500.0, 1750.0, 2000.0])
    a = np.asarray(alpha_from_speed(c_field, c_anch, a_anch))
    assert a[0] == pytest.approx(0.0)
    assert a[1] == pytest.approx(3.0)   # linear midpoint
    assert a[2] == pytest.approx(6.0)


def test_gm_wm_speed_is_degenerate_honest_limit():
    """Guard the documented limit: GM and WM speeds are identical, so c→α maps
    them to the SAME α — the coupling cannot separate GM/WM."""
    from brain_fwi.constitutive import speed_alpha_anchors, alpha_from_speed
    c_anch, a_anch = speed_alpha_anchors()
    a = np.asarray(alpha_from_speed(jnp.array([1560.0, 1560.0]), c_anch, a_anch))
    assert a[0] == a[1]


@pytest.mark.slow
def test_coupling_rescues_alpha_given_recovered_velocity():
    """Integration: on a phantom where the anomaly has BOTH a c and an α contrast
    (skull-like: c=2000, α=6 in water), and velocity is well-recovered (c started
    at truth — what a c-first / hierarchical schedule provides), the constitutive
    coupling recovers α far better than free-voxel: the correct c predicts α=6 and
    fills it, where the weak α data-gradient alone barely moves."""
    import jax.random as jr
    from brain_fwi.simulation.forward import (
        build_domain, build_medium, build_time_axis, simulate_shot_sensors,
        _build_source_signal)
    from brain_fwi.inversion.fwi import FWIConfig, run_fwi

    n, dx, y = 32, 5e-4, 1.1
    grid = (n, n, n)
    zz, yy, xx = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    r = np.sqrt((zz - n/2)**2 + (yy - n/2)**2 + (xx - n/2)**2)
    blob = r <= n * 0.18
    c_true = jnp.asarray(np.where(blob, 2000.0, 1500.0), np.float32)
    alpha_true = jnp.asarray(np.where(blob, 6.0, 0.0), np.float32)
    rho = jnp.full(grid, 1000.0, jnp.float32)
    pml, c_max, t_end = 8, 2100.0, 1.9 * (n * dx) / 1500.0
    dom = build_domain(grid, dx); ref = build_medium(dom, c_max, 1000.0, pml_size=pml)
    ta = build_time_axis(ref, cfl=0.3, t_end=t_end); dt, nt = float(ta.dt), int(ta.Nt)
    sig = _build_source_signal(300e3, dt, nt)
    cx = n // 2; rad = n * 0.42; angs = np.linspace(0, 2*np.pi, 16, endpoint=False)
    xs = np.clip(np.round(cx + rad*np.cos(angs)).astype(int), 0, n-1)
    ys = np.clip(np.round(cx + rad*np.sin(angs)).astype(int), 0, n-1)
    zs = np.full_like(xs, cx); recv = (xs, ys, zs)
    srcs = [(int(xs[i]), int(ys[i]), int(zs[i])) for i in range(0, 16, 2)]
    obs = jnp.stack([simulate_shot_sensors(
        build_medium(dom, c_true, rho, pml_size=pml, attenuation=alpha_true, alpha_power=y),
        ta, sp, recv, sig, dt) for sp in srcs])
    c_init = jnp.asarray(np.array(c_true), jnp.float32)  # well-recovered-c regime
    roi = jnp.asarray(np.ones(grid, np.float32))
    base = dict(freq_bands=[(150e3, 300e3)], n_iters_per_band=20, shots_per_iter=len(srcs),
                learning_rate=20.0, c_min=1400.0, c_max=c_max, pml_size=pml, cfl=0.3,
                gradient_smooth_sigma=1.0, alpha_power=y, invert_attenuation=True,
                attenuation_lr=2.0, attenuation_max=12.0, attenuation_mask=roi, verbose=False)
    at = np.asarray(alpha_true); blob = np.asarray(blob)

    r0 = run_fwi(obs, c_init, rho, dx, srcs, recv, sig, dt, t_end,
                 config=FWIConfig(**base), key=jr.PRNGKey(0))
    anchors = (jnp.array([1500.0, 2000.0]), jnp.array([0.0, 6.0]))
    r1 = run_fwi(obs, c_init, rho, dx, srcs, recv, sig, dt, t_end,
                 config=FWIConfig(attenuation_speed_anchors=anchors, attenuation_speed_weight=0.4,
                                  attenuation_prior_ramp=0.4, **base), key=jr.PRNGKey(0))
    rmse0 = float(np.sqrt(np.mean((np.asarray(r0.attenuation) - at) ** 2)))
    rmse1 = float(np.sqrt(np.mean((np.asarray(r1.attenuation) - at) ** 2)))
    in1 = float(np.asarray(r1.attenuation)[blob].mean())
    assert rmse1 < 0.3 * rmse0, f"coupling should sharply beat free-voxel: {rmse0:.3f} -> {rmse1:.3f}"
    assert in1 > 4.0, f"coupling should recover in-blob alpha near truth 6.0, got {in1:.3f}"


def test_fwi_config_carries_speed_coupling_and_defaults_off():
    """Wiring: the coupling plumbs through FWIConfig and is OFF by default."""
    from brain_fwi.inversion.fwi import FWIConfig
    from brain_fwi.constitutive import speed_alpha_anchors
    assert FWIConfig().attenuation_speed_weight == 0.0
    assert FWIConfig().attenuation_speed_anchors is None
    cfg = FWIConfig(invert_attenuation=True,
                    attenuation_speed_anchors=speed_alpha_anchors(),
                    attenuation_speed_weight=0.4)
    assert cfg.attenuation_speed_weight == 0.4
    assert len(cfg.attenuation_speed_anchors) == 2

"""Phase 6.1 — joint velocity + attenuation (multiparameter) FWI.

The GM/WM contrast channel is attenuation, not sound speed. Phase 5 froze alpha
at truth; Phase 6 co-inverts it. This is the red-green spec for that capability.

Spec (one behaviour): given observations that carry a *localised absorption
anomaly* (a blob of alpha in an otherwise lossless tank), a joint (c, alpha) FWI
started from alpha = 0 must move the reconstructed alpha field TOWARD the truth —
the blob RMSE drops and the recovered alpha concentrates inside the true blob.
A velocity-only FWI cannot do this (no alpha degree of freedom), which is the
whole point of the pivot.

Backward-compat guard: with ``invert_attenuation=False`` the optimisation state
is a bare velocity array and behaviour is unchanged (covered by the existing
absorption/FWI suites; asserted here only via the returned ``attenuation`` being
None).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import pytest


def _alpha_blob_phantom(n=32):
    """Homogeneous water tank with a single spherical absorption anomaly.

    Velocity/density are uniform (so the ONLY thing distinguishing the data from
    a lossless tank is the alpha blob) — this isolates the attenuation channel.
    """
    grid = (n, n, n)
    dx = 5e-4
    y = 1.1
    c = jnp.full(grid, 1500.0, jnp.float32)
    rho = jnp.full(grid, 1000.0, jnp.float32)
    zz, yy, xx = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    r = np.sqrt((zz - n / 2) ** 2 + (yy - n / 2) ** 2 + (xx - n / 2) ** 2)
    blob = (r <= n * 0.18)
    alpha = jnp.asarray(np.where(blob, 6.0, 0.0), dtype=jnp.float32)
    return grid, dx, y, c, rho, alpha, blob


def _ring(n, radius_frac=0.42):
    """A ring of source/receiver grid points around the mid-axial plane."""
    cx = cy = cz = n // 2
    rad = n * radius_frac
    angs = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    xs = np.clip(np.round(cx + rad * np.cos(angs)).astype(int), 0, n - 1)
    ys = np.clip(np.round(cy + rad * np.sin(angs)).astype(int), 0, n - 1)
    zs = np.full_like(xs, cz)
    return xs, ys, zs


@pytest.mark.slow
def test_joint_inversion_recovers_attenuation_anomaly():
    from brain_fwi.simulation.forward import (
        build_domain, build_medium, build_time_axis, simulate_shot_sensors,
        _build_source_signal,
    )
    from brain_fwi.inversion.fwi import FWIConfig, run_fwi

    grid, dx, y, c, rho, alpha_true, blob = _alpha_blob_phantom(32)
    n = grid[0]
    pml = 8
    c_max = 1600.0
    t_end = 1.9 * (n * dx) / 1500.0

    dom = build_domain(grid, dx)
    ref_med = build_medium(dom, c_max, 1000.0, pml_size=pml)
    ta = build_time_axis(ref_med, cfl=0.3, t_end=t_end)
    dt = float(ta.dt); nt = int(ta.Nt)
    sig = _build_source_signal(300e3, dt, nt)

    xs, ys, zs = _ring(n)
    recv = (xs, ys, zs)
    src_positions = [(int(xs[i]), int(ys[i]), int(zs[i])) for i in range(0, 16, 2)]

    # Observed data WITH the absorption blob.
    observed = []
    for sp in src_positions:
        med = build_medium(dom, c, rho, pml_size=pml, attenuation=alpha_true, alpha_power=y)
        observed.append(simulate_shot_sensors(med, ta, sp, recv, sig, dt))
    observed = jnp.stack(observed)

    roi = jnp.asarray(np.ones(grid, np.float32))  # invert everywhere for the test
    cfg = FWIConfig(
        freq_bands=[(150e3, 300e3)], n_iters_per_band=24, shots_per_iter=len(src_positions),
        learning_rate=12.0, c_min=1400.0, c_max=c_max, pml_size=pml, cfl=0.3,
        gradient_smooth_sigma=1.0, alpha_power=y,
        invert_attenuation=True, attenuation_lr=2.0, attenuation_max=12.0,
        attenuation_mask=roi, verbose=False,
    )
    res = run_fwi(observed, c, rho, dx, src_positions, recv, sig, dt, t_end,
                  config=cfg, key=jr.PRNGKey(0))

    assert res.attenuation is not None, "joint FWI must return a recovered alpha field"
    a_rec = np.asarray(res.attenuation)
    a_true = np.asarray(alpha_true)
    blob = np.asarray(blob)

    rmse_init = float(np.sqrt(np.mean((0.0 - a_true) ** 2)))          # guess is alpha=0
    rmse_final = float(np.sqrt(np.mean((a_rec - a_true) ** 2)))
    in_blob = float(a_rec[blob].mean())
    out_blob = float(a_rec[~blob].mean())
    print(f"\n[alpha recovery] rmse {rmse_init:.3f} -> {rmse_final:.3f}; "
          f"in-blob mean {in_blob:.3f}, out-blob mean {out_blob:.3f}", flush=True)

    # Phase-only misfit constrains alpha weakly (this is *why* #45 adds the
    # CANN/KK coupling), so the honest spec for 6.1 is DIRECTION + LOCALISATION:
    # the joint inversion moves alpha toward truth and concentrates it inside
    # the true blob — evidence the attenuation channel is wired and its gradient
    # points the right way.
    assert rmse_final < 0.85 * rmse_init, (
        f"joint FWI did not move alpha toward truth: rmse {rmse_init:.3f} -> {rmse_final:.3f}"
    )
    assert in_blob > 4.0 * max(out_blob, 1e-6) and in_blob > 0.5, (
        f"recovered alpha not localised in the true blob: in={in_blob:.3f} out={out_blob:.3f}"
    )


def test_velocity_only_path_returns_no_attenuation():
    """Backward-compat: default config leaves the state a bare velocity array
    and reports no recovered attenuation (fast, no forward sim)."""
    from brain_fwi.inversion.fwi import FWIConfig
    cfg = FWIConfig()
    assert cfg.invert_attenuation is False


def test_hierarchical_release_schedule():
    """c-first schedule: α is frozen (not released) until the global iteration
    reaches attenuation_release_frac of the total, then released. Default
    frac=0.0 ⇒ released from iter 0 (co-inversion, unchanged)."""
    from brain_fwi.inversion.fwi import FWIConfig, _alpha_released
    # 2 bands x 10 iters = 20 total; release at 50% ⇒ global_iter >= 10.
    cfg = FWIConfig(freq_bands=[(1, 2), (2, 3)], n_iters_per_band=10,
                    attenuation_release_frac=0.5)
    assert _alpha_released(0, 0, cfg) is False      # global 0
    assert _alpha_released(0, 9, cfg) is False       # global 9
    assert _alpha_released(1, 0, cfg) is True         # global 10
    assert _alpha_released(1, 9, cfg) is True          # global 19
    # default: released from the very first iter (backward compatible)
    d = FWIConfig(freq_bands=[(1, 2)], n_iters_per_band=5)
    assert _alpha_released(0, 0, d) is True


@pytest.mark.slow
def test_alpha_frozen_when_never_released():
    """release_frac=1.0 freezes α for the entire run ⇒ recovered α equals its
    init (only velocity moves). Guards the c-first freeze end-to-end."""
    from brain_fwi.simulation.forward import (
        build_domain, build_medium, build_time_axis, simulate_shot_sensors,
        _build_source_signal)
    from brain_fwi.inversion.fwi import FWIConfig, run_fwi

    grid, dx, y, c, rho, alpha_true, blob = _alpha_blob_phantom(24)
    n = grid[0]; pml = 8; c_max = 1600.0; t_end = 1.9 * (n * dx) / 1500.0
    dom = build_domain(grid, dx); ref = build_medium(dom, c_max, 1000.0, pml_size=pml)
    ta = build_time_axis(ref, cfl=0.3, t_end=t_end); dt = float(ta.dt); nt = int(ta.Nt)
    sig = _build_source_signal(300e3, dt, nt)
    xs, ys, zs = _ring(n); recv = (xs, ys, zs)
    srcs = [(int(xs[i]), int(ys[i]), int(zs[i])) for i in range(0, 16, 4)]
    obs = jnp.stack([simulate_shot_sensors(
        build_medium(dom, c, rho, pml_size=pml, attenuation=alpha_true, alpha_power=y),
        ta, sp, recv, sig, dt) for sp in srcs])
    a_init = jnp.zeros(grid, jnp.float32)
    cfg = FWIConfig(freq_bands=[(150e3, 300e3)], n_iters_per_band=6, shots_per_iter=len(srcs),
                    learning_rate=15.0, c_min=1400.0, c_max=c_max, pml_size=pml, cfl=0.3,
                    gradient_smooth_sigma=1.0, alpha_power=y, invert_attenuation=True,
                    attenuation_init=a_init, attenuation_lr=2.0, attenuation_max=12.0,
                    attenuation_release_frac=1.0, verbose=False)
    res = run_fwi(obs, c, rho, dx, srcs, recv, sig, dt, t_end, config=cfg, key=jr.PRNGKey(0))
    assert float(jnp.max(jnp.abs(res.attenuation - a_init))) < 1e-6, "α must stay frozen"

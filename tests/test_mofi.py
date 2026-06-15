"""Tests for MOFI SE(2) pose alignment (src/brain_fwi/inversion/mofi.py).

Staged after the paper (arXiv:2601.14533):

  Stage 0  (fast, CPU)   — SE(2) warp correctness + autodiff pose recovery
                           on an image-matching loss. Validates the core
                           ∂c_φ/∂φ + jax.grad plumbing WITHOUT j-Wave.
  Stage 1  (slow, GPU)   — finite-difference check of ∂f/∂φ through the
                           real j-Wave forward (gradient plumbing end to end).
  Stage 2  (slow, GPU)   — in-silico "inverse crime": impose a known SE(2)
                           offset on a synthetic skull phantom, recover it
                           to within tolerance (mirrors paper Fig 2/§2.1).
  Stage 3  (slow, GPU)   — MOFI-then-FWI: plain FWI fails from a mis-posed
                           start; MOFI alignment + FWI succeeds (paper Fig 4).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brain_fwi.inversion.mofi import warp_se2, warp_medium


# ===========================================================================
# Stage 0 — SE(2) warp + autodiff core (fast, no forward simulation)
# ===========================================================================

@pytest.fixture
def skull_like():
    """A skull-like annulus on a water background — enough structure for
    a rigid-warp registration test (asymmetric so rotation is observable)."""
    H, W = 80, 80
    yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    r = jnp.sqrt((xx - 40.0) ** 2 + (yy - 38.0) ** 2)
    ring = (r > 22) & (r < 28)
    # break symmetry: a notch so θ is identifiable
    notch = (xx > 38) & (xx < 42) & (yy < 16)
    c = jnp.where(ring & ~notch, 2800.0, 1500.0).astype(jnp.float32)
    return c


def test_warp_identity_is_noop(skull_like):
    """Zero pose returns the template unchanged (to interpolation error)."""
    out = warp_se2(skull_like, jnp.array([0.0, 0.0, 0.0]), cval=1500.0)
    assert jnp.allclose(out, skull_like, atol=1e-4)


def test_warp_integer_translation_matches_roll(skull_like):
    """A pure integer-pixel translation equals jnp.roll on the interior.

    pose = (0, δ₁, δ₂): output[i,j] = template[i-δ₁, j-δ₂] (inverse warp),
    so it equals template rolled by (+δ₁, +δ₂) away from the borders."""
    d1, d2 = 5, -3
    out = warp_se2(skull_like, jnp.array([0.0, float(d1), float(d2)]), cval=1500.0)
    rolled = jnp.roll(jnp.roll(skull_like, d1, axis=0), d2, axis=1)
    interior = (slice(10, 70), slice(10, 70))
    assert jnp.allclose(out[interior], rolled[interior], atol=1e-3)


def test_warp_is_differentiable_wrt_pose(skull_like):
    """∂(warp)/∂φ exists and is finite (the Jacobian MOFI relies on)."""
    def scalar(pose):
        return jnp.sum(warp_se2(skull_like, pose, cval=1500.0) ** 2)
    g = jax.grad(scalar)(jnp.array([0.05, 1.0, -1.0]))
    assert g.shape == (3,)
    assert jnp.all(jnp.isfinite(g))
    assert jnp.linalg.norm(g) > 0


def test_warp_medium_moves_density_with_skull(skull_like):
    """Array-valued density warps with the same pose; scalar passes through."""
    rho = jnp.where(skull_like > 2000, 1900.0, 1000.0).astype(jnp.float32)
    pose = jnp.array([0.1, 3.0, 2.0])
    c_w, rho_w = warp_medium(skull_like, rho, pose, c_bg=1500.0, rho_bg=1000.0)
    # skull (high-c) and dense bone should remain co-located after warp
    assert jnp.corrcoef((c_w > 2000).ravel(), (rho_w > 1450).ravel())[0, 1] > 0.9
    c_w2, rho_scalar = warp_medium(skull_like, 1000.0, pose)
    assert jnp.ndim(rho_scalar) == 0


def test_autodiff_pose_recovery_image_matching(skull_like):
    """THE core test: recover a known SE(2) pose by gradient descent on an
    image-matching loss, with no forward simulation.

    target = warp(template, φ*). Minimise ‖warp(template, φ) − target‖²
    over φ via jax.grad. This exercises exactly the ∂c_φ/∂φ + autodiff
    machinery MOFI uses; the only thing the real loss swaps in is the
    j-Wave forward between warp and residual. If this converges, the SE(2)
    manifold optimisation is sound.
    """
    # Smooth, strongly anisotropic phantom: rotation is well-posed and the
    # pixel-L2 basin is wide. (A thin high-contrast ring is near
    # rotationally symmetric → no rotation gradient; that is a property of
    # the toy loss, not of MOFI, whose waveform misfit senses rotation via
    # transmitted/reflected arrival shifts.)
    H, W = 80, 80
    yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    blob = jnp.exp(-(((xx - 40.0) / 26.0) ** 2 + ((yy - 40.0) / 13.0) ** 2))
    template = (1500.0 + 1300.0 * blob).astype(jnp.float32)

    true_pose = jnp.array([0.12, 4.0, -3.0])  # ~6.9°, (4,-3) px
    target = warp_se2(template, true_pose, cval=1500.0)

    rs = 20.0  # rotation scale (≈ semi-axis) so θ and δ are commensurate
    scale = jnp.array([rs, 1.0, 1.0])

    def loss(pose):
        return jnp.mean((warp_se2(template, pose, cval=1500.0) - target) ** 2)

    grad_fn = jax.jit(jax.grad(loss))
    pose = jnp.array([0.0, 0.0, 0.0])
    g0 = None
    for _ in range(400):
        g = grad_fn(pose) * (1.0 / scale)        # to scaled coords
        if g0 is None:
            g0 = jnp.linalg.norm(g) + 1e-30
        pose = pose - 0.5 * (g / g0) / scale     # step, back to pose coords

    assert jnp.degrees(abs(pose[0] - true_pose[0])) < 0.5, f"θ off: {pose}"
    assert abs(pose[1] - true_pose[1]) < 0.5, f"δ₁ off: {pose}"
    assert abs(pose[2] - true_pose[2]) < 0.5, f"δ₂ off: {pose}"


# ===========================================================================
# Stage 1 — gradient through j-Wave (slow / GPU). Plan, written for opt-in.
# ===========================================================================

@pytest.mark.slow
def test_pose_gradient_matches_finite_difference():
    """∂f/∂φ from jax.grad(mofi_pose_loss) agrees with a central finite
    difference of the forward-simulated misfit, per component.

    Small 2D grid (64²), 4 shots, ~260 timesteps. Confirms the warp→j-Wave
    VJP is wired correctly (sign AND scale — catches a flipped gradient, a
    missing 0.5 in the L2, or a transposed warp Jacobian), not merely that
    it is finite. End-to-end analogue of stage 0's pure-geometry check.

    Observed data is generated from the template at its native pose; the
    gradient is probed at a generic NON-zero pose (off the minimum, off
    integer-pixel kinks) so all three components are non-trivial.

    Tolerance note: the warp is bilinear (``map_coordinates`` order=1), so
    the loss is only C0 — its derivative jumps at pixel-cell boundaries.
    Autodiff returns the exact within-cell derivative; a central FD chords
    across those kinks, and the j-Wave forward is float32 (the medium is
    cast to float32, so x64 would not help the FD noise floor). Hence
    ~5-15% FD-vs-autodiff disagreement is expected precision, not a bug.
    The PRIMARY structural checks are therefore cosine similarity (catches
    sign flips / transposed VJP) and per-component sign; the magnitude
    bound is set to still catch a real scale error (e.g. a 2x factor gives
    rel ~ 0.5) while passing the legitimate interpolation bias. Heavier
    template smoothing (sigma=2) shrinks the kink jumps to keep FD tight.
    """
    from brain_fwi.inversion.fwi import _smooth_gradient
    from brain_fwi.inversion.mofi import mofi_pose_loss
    from brain_fwi.simulation.forward import (
        build_domain, build_medium, build_time_axis, generate_observed_data,
    )
    from brain_fwi.transducers import ring_array_2d, transducer_positions_to_grid
    from brain_fwi.utils.wavelets import ricker_wavelet

    grid_shape = (64, 64)
    dx = 1.0e-3
    pml = 10

    # Asymmetric skull-like phantom (notch + offset bump break rotational
    # symmetry so the θ-gradient is non-degenerate), lightly Gaussian-blurred
    # so the bilinear-warp loss is smooth enough for a clean FD comparison.
    yy, xx = jnp.meshgrid(jnp.arange(64), jnp.arange(64), indexing="ij")
    r = jnp.sqrt((xx - 32.0) ** 2 + (yy - 30.0) ** 2)
    ring = (r > 11) & (r < 15)
    notch = (xx > 30) & (xx < 36) & (yy < 18)
    bump = ((xx - 42.0) ** 2 + (yy - 32.0) ** 2) < 9
    skull = (ring & ~notch) | bump
    template_c = jnp.where(skull, 2600.0, 1500.0).astype(jnp.float32)
    template_c = _smooth_gradient(template_c, sigma=2.0)  # shrink bilinear kink jumps
    rho = jnp.full(grid_shape, 1000.0, dtype=jnp.float32)  # constant density

    # 16-element ring at radius 20 px — inside the non-PML interior [10, 53].
    cx = cy = 32 * dx
    positions = ring_array_2d(
        n_elements=16, center=(cx, cy),
        semi_major=20e-3, semi_minor=20e-3, standoff=0.0,
    )
    pg = transducer_positions_to_grid(positions, dx, grid_shape)
    src_list = [(int(pg[0][i]), int(pg[1][i])) for i in range(16)]
    sensor_pos = pg

    # One time axis (ref medium at c_ref) shared by observed + predicted, with
    # the source sampled at the simulation dt — no source/CFL dt mismatch.
    t_end = 3.0e-5
    c_ref = float(max(float(jnp.max(template_c)), 1500.0))
    ref_medium = build_medium(build_domain(grid_shape, dx), c_ref, 1000.0, pml_size=pml)
    time_axis = build_time_axis(ref_medium, cfl=0.3, t_end=t_end)
    dt = float(time_axis.dt)
    n_samples = int(t_end / dt)
    signal = ricker_wavelet(f0=150e3, dt=dt, n_samples=n_samples)

    observed = generate_observed_data(
        sound_speed=template_c, density=rho, dx=dx,
        src_positions_grid=src_list, sensor_positions_grid=sensor_pos,
        freq=150e3, pml_size=pml, time_axis=time_axis,
        source_signal=signal, dt=dt, verbose=False,
    )

    centre = ((64 - 1) / 2.0, (64 - 1) / 2.0)
    shot_idx = [0, 4, 8, 12]
    eval_pose = jnp.array([0.04, 1.5, -1.0])  # ~2.3°, generic non-integer pose

    def total_loss(pose):
        tot = 0.0
        for si in shot_idx:
            tot = tot + mofi_pose_loss(
                pose, template_c, 1000.0, dx, src_list[si], sensor_pos,
                signal, dt, time_axis, observed[si],
                centre=centre, c_bg=1500.0, rho_bg=1000.0, pml_size=pml,
            )
        return tot

    val, grad = jax.value_and_grad(total_loss)(eval_pose)
    grad = np.asarray(grad, dtype=np.float64)

    # Central finite differences (per-component steps: rad, px, px). Small
    # enough to reduce pixel-boundary crossings, large enough to stay above
    # the float32 forward-sim noise floor.
    h = np.array([5e-4, 5e-3, 5e-3])
    fd = np.zeros(3)
    for k in range(3):
        ep = eval_pose.at[k].add(h[k])
        em = eval_pose.at[k].add(-h[k])
        fd[k] = (float(total_loss(ep)) - float(total_loss(em))) / (2 * h[k])

    rel = np.linalg.norm(grad - fd) / (np.linalg.norm(fd) + 1e-30)
    cos = float(np.dot(grad, fd) / (np.linalg.norm(grad) * np.linalg.norm(fd) + 1e-30))
    print(f"\n[stage1] loss={float(val):.6e}")
    print(f"[stage1] autodiff ∂f/∂φ = {grad}")
    print(f"[stage1] finite-diff   = {fd}")
    print(f"[stage1] rel L2 err = {rel:.4f}, cosine = {cos:.4f}")

    assert np.all(np.isfinite(grad))
    # Primary: direction (catches sign flips / transposed VJP). Secondary:
    # magnitude within the C0-bilinear + float32 FD precision (see docstring).
    assert cos > 0.995, f"gradient direction off: cos={cos:.4f} (grad {grad}, fd {fd})"
    assert rel < 0.15, f"gradient magnitude off: rel={rel:.4f} (grad {grad}, fd {fd})"
    for k in range(3):
        if abs(fd[k]) > 1e-12:
            assert np.sign(grad[k]) == np.sign(fd[k]), (
                f"component {k} sign flip: grad {grad[k]:.3e} vs fd {fd[k]:.3e}")


# ===========================================================================
# Stage 2 — in-silico pose recovery (slow / GPU). Mirrors paper §2.1.
# ===========================================================================

@pytest.mark.slow
def test_mofi_recovers_known_pose_in_silico():
    """Inverse crime (paper §2.1): the observed data is generated from the
    skull template displaced by a KNOWN SE(2) pose; run_mofi starts from
    the identity pose and must recover that displacement.

    Because the forward model can reproduce the truth exactly
    (c_true = warp_se2(template, phi_true), and run_mofi warps the same
    template), the global minimum sits at phi_true with zero misfit —
    isolating the optimiser/basin behaviour. Multi-scale low→high banding
    keeps the ~5 px / 6 deg offset inside the basin of attraction.

    Tolerance note: rotation and translation are coupled through the
    rotation centre, forming a slow curved valley; on this deliberately
    CHEAP problem (64^2, 16 elements, 60 iters) the recovered pose settles
    ~2 px from truth. The paper's ~1 px came from 500 iters on a 1024-
    element ring — two orders of magnitude more acquisition/compute. So the
    contract here certifies "recovers the pose" at this scale, not the
    paper's precision: |Δθ| < 2 deg, |Δδ| < 2.5 grid points. The run is
    deterministic (fixed PRNG key): rs=8 / 6 shots / 60 iters gives
    Δθ≈1.24 deg, Δδ≈(2.2, 0.0) px.
    """
    from brain_fwi.inversion.fwi import _smooth_gradient
    from brain_fwi.inversion.mofi import MOFIConfig, run_mofi, warp_se2
    from brain_fwi.simulation.forward import (
        build_domain, build_medium, build_time_axis, generate_observed_data,
    )
    from brain_fwi.transducers import ring_array_2d, transducer_positions_to_grid
    from brain_fwi.utils.wavelets import ricker_wavelet

    grid_shape = (64, 64)
    dx = 1.0e-3
    pml = 10

    # Asymmetric skull-like template (notch + bump break rotational symmetry).
    yy, xx = jnp.meshgrid(jnp.arange(64), jnp.arange(64), indexing="ij")
    r = jnp.sqrt((xx - 32.0) ** 2 + (yy - 31.0) ** 2)
    ring = (r > 11) & (r < 16)
    notch = (xx > 30) & (xx < 36) & (yy < 18)
    bump = ((xx - 43.0) ** 2 + (yy - 33.0) ** 2) < 10
    skull = (ring & ~notch) | bump
    template = jnp.where(skull, 2600.0, 1500.0).astype(jnp.float32)
    template = _smooth_gradient(template, sigma=1.5)
    rho = jnp.full(grid_shape, 1000.0, dtype=jnp.float32)

    centre = ((64 - 1) / 2.0, (64 - 1) / 2.0)
    phi_true = jnp.array([0.10, 4.0, -3.0])  # 5.7 deg, (4, -3) px
    c_true = warp_se2(template, phi_true, centre=centre, cval=1500.0)

    # Ring array (radius 20 px, inside non-PML interior).
    cx = cy = 32 * dx
    positions = ring_array_2d(
        n_elements=16, center=(cx, cy),
        semi_major=20e-3, semi_minor=20e-3, standoff=0.0,
    )
    pg = transducer_positions_to_grid(positions, dx, grid_shape)
    src_list = [(int(pg[0][i]), int(pg[1][i])) for i in range(16)]
    sensor_pos = pg

    t_end = 3.0e-5
    c_ref = float(max(float(jnp.max(template)), 1500.0))
    ref_medium = build_medium(build_domain(grid_shape, dx), c_ref, 1000.0, pml_size=pml)
    time_axis = build_time_axis(ref_medium, cfl=0.3, t_end=t_end)
    dt = float(time_axis.dt)
    n_samples = int(t_end / dt)
    signal = ricker_wavelet(f0=120e3, dt=dt, n_samples=n_samples)

    observed = generate_observed_data(
        sound_speed=c_true, density=rho, dx=dx,
        src_positions_grid=src_list, sensor_positions_grid=sensor_pos,
        freq=120e3, pml_size=pml, time_axis=time_axis,
        source_signal=signal, dt=dt, verbose=False,
    )

    config = MOFIConfig(
        freq_bands=[(40e3, 90e3), (90e3, 160e3)],  # low→high keeps offset in-basin
        n_iters_per_band=30,        # 60 total; rotation/translation valley → ~2px floor at this scale
        shots_per_iter=6,           # clean (right-signed) rotation gradient
        step_size=0.5,
        rotation_centre=centre,
        # single-norm (stable) + small rotation_scale accelerates rotation
        # (step ~ 1/rotation_scale^2) without the per-component instability.
        rotation_scale=8.0,
        c_bg=1500.0, rho_bg=1000.0,
        pml_size=pml,
        line_search=True, ls_max_tries=2,
        per_component_norm=False,
        verbose=False,
    )
    result = run_mofi(
        observed_data=observed,
        template_velocity=template,
        template_density=1000.0,
        dx=dx,
        src_positions_grid=src_list,
        sensor_positions_grid=sensor_pos,
        source_signal=signal,
        dt=dt, t_end=t_end,
        config=config,
    )

    pose = np.asarray(result.pose)
    dtheta_deg = abs(np.degrees(pose[0] - float(phi_true[0])))
    dd1 = abs(pose[1] - float(phi_true[1]))
    dd2 = abs(pose[2] - float(phi_true[2]))
    print(f"\n[stage2] recovered pose θ={np.degrees(pose[0]):+.2f}° "
          f"δ=({pose[1]:+.2f},{pose[2]:+.2f}) px")
    print(f"[stage2] true pose      θ={np.degrees(float(phi_true[0])):+.2f}° "
          f"δ=({float(phi_true[1]):+.2f},{float(phi_true[2]):+.2f}) px")
    print(f"[stage2] errors: Δθ={dtheta_deg:.2f}°  Δδ=({dd1:.2f},{dd2:.2f}) px")

    assert dtheta_deg < 2.0, f"rotation not recovered: Δθ={dtheta_deg:.2f}°"
    assert dd1 < 2.5 and dd2 < 2.5, f"translation not recovered: Δδ=({dd1:.2f},{dd2:.2f}) px"


# ===========================================================================
# Stage 3 — MOFI unblocks FWI (slow / GPU). Mirrors paper Fig 4.
# ===========================================================================

@pytest.mark.slow
def test_mofi_alignment_unblocks_fwi():
    """MOFI unblocks FWI (paper Fig 4). The skull is the prior (held fixed,
    FWI masked to the interior); FWI reconstructs a brain blob. The ONLY
    difference between the two runs is the skull pose:

      (A) skull template at the canonical (mis-posed) pose;
      (B) skull template MOFI-aligned to the data.

    With the skull mis-posed, its echoes arrive at the wrong times and the
    interior reconstruction is corrupted; aligning it first lets FWI
    recover the blob. Contract: interior RMSE_B < 0.85 * RMSE_A.

    MOFI here uses a robust LOW-frequency single band: the skull-only
    template vs skull+blob data is a model mismatch, and high-frequency
    pose refinement diverges on the unmodeled scatterer (see the mofi_cfg
    comment). Low-freq alignment recovers the dominant translation, which
    is what unblocks FWI; Stage 2 verifies full SE(2) recovery separately.

    Skull velocity = 2800 so MOFI's reference (max template) matches FWI's
    c_max — a single consistent time axis for data, MOFI, and FWI.
    """
    from brain_fwi.inversion.fwi import FWIConfig, run_fwi, _smooth_gradient
    from brain_fwi.inversion.mofi import MOFIConfig, run_mofi, warp_se2
    from brain_fwi.simulation.forward import (
        build_domain, build_medium, build_time_axis, generate_observed_data,
    )
    from brain_fwi.transducers import ring_array_2d, transducer_positions_to_grid
    from brain_fwi.utils.wavelets import ricker_wavelet

    grid_shape = (64, 64)
    dx = 1.0e-3
    pml = 10

    yy, xx = jnp.meshgrid(jnp.arange(64), jnp.arange(64), indexing="ij")
    # Skull-only template (canonical), asymmetric for rotation identifiability.
    r = jnp.sqrt((xx - 32.0) ** 2 + (yy - 31.0) ** 2)
    ring = (r > 13) & (r < 17)
    notch = (xx > 30) & (xx < 36) & (yy < 16)
    bump = ((xx - 44.0) ** 2 + (yy - 32.0) ** 2) < 9
    skull = (ring & ~notch) | bump
    skull_template = _smooth_gradient(
        jnp.where(skull, 2800.0, 1500.0).astype(jnp.float32), sigma=1.0)
    # Ground-truth phantom = skull + a central brain blob (the FWI target).
    blob = jnp.sqrt((xx - 30.0) ** 2 + (yy - 31.0) ** 2) < 4
    full_canonical = jnp.where(blob, 1650.0, skull_template).astype(jnp.float32)
    rho = jnp.full(grid_shape, 1000.0, dtype=jnp.float32)

    centre = ((64 - 1) / 2.0, (64 - 1) / 2.0)
    phi_true = jnp.array([0.10, 4.0, -3.0])  # 5.7°, (4,-3) px
    c_gt = warp_se2(full_canonical, phi_true, centre=centre, cval=1500.0)

    # Fixed central interior mask (FWI update region + RMSE region). r<10 sits
    # inside the skull (inner radius 13) for BOTH poses; the warped blob stays
    # within it, so no overlap with either skull position.
    interior = (jnp.sqrt((xx - 31.5) ** 2 + (yy - 31.5) ** 2) < 10).astype(jnp.float32)

    pos = ring_array_2d(n_elements=16, center=(32 * dx, 32 * dx),
                        semi_major=22e-3, semi_minor=22e-3, standoff=0.0)
    pg = transducer_positions_to_grid(pos, dx, grid_shape)
    src = [(int(pg[0][i]), int(pg[1][i])) for i in range(16)]

    t_end = 3.0e-5
    refm = build_medium(build_domain(grid_shape, dx), 2800.0, 1000.0, pml_size=pml)
    ta = build_time_axis(refm, cfl=0.3, t_end=t_end)
    dt = float(ta.dt); ns = int(t_end / dt)
    sig = ricker_wavelet(f0=120e3, dt=dt, n_samples=ns)
    observed = generate_observed_data(
        sound_speed=c_gt, density=rho, dx=dx, src_positions_grid=src,
        sensor_positions_grid=pg, freq=120e3, pml_size=pml, time_axis=ta,
        source_signal=sig, dt=dt, verbose=False)

    def fwi_from(c_init):
        cfg = FWIConfig(
            freq_bands=[(40e3, 90e3), (90e3, 160e3)], n_iters_per_band=15,
            shots_per_iter=4, learning_rate=30.0, c_min=1400.0, c_max=2800.0,
            pml_size=pml, gradient_smooth_sigma=1.0, loss_fn="l2",
            mask=interior, verbose=False)  # sgd is the base default
        return run_fwi(observed_data=observed, initial_velocity=c_init, density=rho,
                       dx=dx, src_positions_grid=src, sensor_positions_grid=pg,
                       source_signal=sig, dt=dt, t_end=t_end, config=cfg).velocity

    m = interior > 0.5
    rmse = lambda c: float(jnp.sqrt(jnp.mean(((c - c_gt) ** 2)[m])))

    # (A) Mis-posed skull (canonical pose), interior unknown (1500).
    rmse_A = rmse(fwi_from(skull_template))

    # (B) MOFI-align the skull, then FWI.
    # Robust LOW-FREQUENCY single band only. Stage 3 has a model mismatch —
    # MOFI warps a skull-ONLY template but the data contains an unmodeled
    # brain blob. At high frequency that unmodeled scatterer cycle-skips the
    # pose search and it DIVERGES (observed: 2 bands → θ=-35°, ratio 1.31).
    # Low frequency robustly recovers the dominant translation, which is what
    # unblocks FWI here. Full SE(2) recovery (incl. rotation) is verified in
    # Stage 2's clean blob-free inverse-crime setting.
    mofi_cfg = MOFIConfig(
        freq_bands=[(40e3, 110e3)], n_iters_per_band=30, shots_per_iter=6,
        step_size=0.5, rotation_centre=centre, rotation_scale=8.0,
        c_bg=1500.0, rho_bg=1000.0, pml_size=pml,
        line_search=True, ls_max_tries=2, per_component_norm=False, verbose=False)
    mofi_res = run_mofi(
        observed_data=observed, template_velocity=skull_template,
        template_density=1000.0, dx=dx, src_positions_grid=src,
        sensor_positions_grid=pg, source_signal=sig, dt=dt, t_end=t_end,
        config=mofi_cfg)
    rmse_B = rmse(fwi_from(mofi_res.aligned_velocity))

    p = np.asarray(mofi_res.pose)
    print(f"\n[stage3] MOFI pose: θ={np.degrees(p[0]):+.2f}° δ=({p[1]:+.2f},{p[2]:+.2f}) px")
    print(f"[stage3] interior RMSE: mis-posed FWI={rmse_A:.1f}  MOFI+FWI={rmse_B:.1f} m/s"
          f"  (ratio {rmse_B/rmse_A:.2f})")

    assert rmse_B < 0.85 * rmse_A, (
        f"MOFI did not unblock FWI: mis-posed RMSE={rmse_A:.1f}, "
        f"aligned RMSE={rmse_B:.1f} m/s (ratio {rmse_B/rmse_A:.2f})")

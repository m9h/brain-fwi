"""MOFI — Manifold Optimisation for Full-Waveform Inversion (pose alignment).

JAX re-implementation of Bates, Cueto, Coleman, Smith, Guasch &
Calderón Agudo, "Automatic skull-template alignment without a guidance
image" (arXiv:2601.14533, 2026).

Problem
-------
Transcranial FWI needs a skull *template* (sound speed + shape, usually
from CT) as a starting model, rigidly aligned to the patient's head
pose. Conventionally that alignment uses a guidance MRI/CT. A mis-posed
template makes pixel-wise FWI fail (cycle-skipping on the global
translation/rotation mismatch). MOFI recovers the pose **from the
acoustic data alone** by optimising a 3-DOF SE(2) rigid transform of the
template instead of the ~10^5-10^6 voxels.

The JAX angle
-------------
The paper assembles the pose gradient by hand:

    ∂f/∂φ = (∂c_φ/∂φ)ᵀ · ∂f/∂c_φ          (their Eq. for the chain rule)

where ∂f/∂c_φ is the ordinary adjoint-state FWI gradient (Stride) and
∂c_φ/∂φ is the Jacobian of the rigid warp, applied as a
Jacobian-transpose–vector product via PyTorch autodiff. Two frameworks,
manual plumbing.

Here it is a *single* ``jax.grad`` of a loss that composes the SE(2)
warp with the existing differentiable j-Wave forward — JAX builds the
whole VJP automatically:

    loss(φ) = ‖ L( warp_se2(template, φ) ) − d ‖²
    ∂f/∂φ   = jax.grad(loss)(φ)            # warp + j-Wave, end to end

This module provides:
  * :func:`warp_se2`              — differentiable bilinear SE(2) warp
  * :func:`mofi_pose_loss`        — single-shot pose misfit through j-Wave
  * :func:`run_mofi`              — gradient-normalised + line-searched
                                    pose descent over frequency bands

Output is a recovered pose ``(θ, δ₁, δ₂)`` and the warped (aligned)
template, which is then handed to ``run_fwi`` as the starting model.

Status: SCAFFOLD. SE(2)/warp/autodiff core is complete and unit-tested
(see tests/test_mofi.py, stage 0). The forward-in-the-loop driver mirrors
``run_fwi``'s shot accumulation; tune step/line-search on real data.
Paper's hard-won practicalities are encoded: normalise by the *first*
gradient's norm (not running max — that was unstable for them), explicit
line search, stochastic source batching. Rigid SE(2) / 2D only, like the
paper; SE(3) and non-rigid (diffeomorphism group) are the noted
extensions.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.scipy.ndimage import map_coordinates

from ..simulation.forward import (
    build_domain,
    build_medium,
    build_time_axis,
    simulate_shot_sensors,
)
from .losses import envelope_loss, l2_loss, multiscale_loss

Pose = jnp.ndarray  # shape (3,): (theta_rad, d1_pix, d2_pix)


# ---------------------------------------------------------------------------
# SE(2) rigid warp  (the ∂c_φ/∂φ that JAX differentiates for us)
# ---------------------------------------------------------------------------

def warp_se2(
    template: jnp.ndarray,
    pose: Pose,
    centre: Optional[Tuple[float, float]] = None,
    cval: float = 1500.0,
) -> jnp.ndarray:
    """Inverse-warp a 2D field by an SE(2) rigid pose (rotate about
    ``centre``, then translate). Bilinear interpolation, fully
    differentiable w.r.t. ``pose`` and ``template``.

    Convention (matches rigid image registration / the paper's T_φ):
    the output pixel at grid coordinate ``x_trans`` takes the template
    value at ``T_φ⁻¹(x_trans)``. The forward map in centred coordinates
    is ``x_trans = R(θ) x_orig + δ``; resampling uses the inverse
    ``x_orig = R(θ)⁻¹ (x_trans − δ)``.

    Args:
        template: (H, W) field on the simulation grid (e.g. skull
            sound-speed template, m/s).
        pose: (3,) array ``(θ, δ₁, δ₂)`` — rotation in **radians**,
            translation in **grid points**.
        centre: rotation centre in grid coords. Default = grid centre.
        cval: fill value for samples mapping outside the template
            (the surrounding medium — water ≈ 1500 m/s for sound speed,
            1000 kg/m³ for density).

    Returns:
        (H, W) warped field.
    """
    H, W = template.shape
    if centre is None:
        centre = ((H - 1) / 2.0, (W - 1) / 2.0)

    theta, d1, d2 = pose[0], pose[1], pose[2]
    cos, sin = jnp.cos(theta), jnp.sin(theta)
    # R(θ)⁻¹ = R(−θ)
    r_inv = jnp.stack([
        jnp.stack([cos, sin]),
        jnp.stack([-sin, cos]),
    ])  # (2, 2)

    ii, jj = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    out = jnp.stack([ii.ravel(), jj.ravel()]).astype(jnp.float32)  # (2, N) output coords
    c = jnp.asarray(centre, dtype=jnp.float32).reshape(2, 1)
    t = jnp.stack([d1, d2]).astype(jnp.float32).reshape(2, 1)

    src = r_inv @ ((out - c) - t) + c  # (2, N) source coords in template
    warped = map_coordinates(
        template, [src[0], src[1]], order=1, mode="constant", cval=cval,
    )
    return warped.reshape(H, W)


def warp_medium(
    template_c: jnp.ndarray,
    template_rho: Union[float, jnp.ndarray],
    pose: Pose,
    centre: Optional[Tuple[float, float]] = None,
    c_bg: float = 1500.0,
    rho_bg: float = 1000.0,
) -> Tuple[jnp.ndarray, Union[float, jnp.ndarray]]:
    """Warp sound-speed (and, if array-valued, density) by the *same* pose.

    The paper's in-vitro setup uses constant density; for in-silico work
    where the skull template carries a density contrast, the density map
    must move rigidly with the skull — hence the shared pose.
    """
    c_phi = warp_se2(template_c, pose, centre=centre, cval=c_bg)
    if jnp.ndim(template_rho) == 0:
        return c_phi, template_rho
    rho_phi = warp_se2(template_rho, pose, centre=centre, cval=rho_bg)
    return c_phi, rho_phi


# ---------------------------------------------------------------------------
# Pose misfit through the j-Wave forward  (one shot)
# ---------------------------------------------------------------------------

def mofi_pose_loss(
    pose: Pose,
    template_c: jnp.ndarray,
    template_rho: Union[float, jnp.ndarray],
    dx: float,
    src_pos: Tuple[int, ...],
    sensor_positions_grid: Tuple,
    signal: jnp.ndarray,
    dt: float,
    time_axis,
    observed: jnp.ndarray,
    *,
    centre: Optional[Tuple[float, float]] = None,
    c_bg: float = 1500.0,
    rho_bg: float = 1000.0,
    pml_size: int = 10,
    loss_fn: Callable = l2_loss,
    checkpointed: bool = False,
) -> jnp.ndarray:
    """Single-shot data misfit as a function of pose ``φ`` only.

    ``jax.value_and_grad(mofi_pose_loss)(pose, ...)`` returns
    ``(f, ∂f/∂φ)`` with the full warp→j-Wave chain rule built by JAX.
    """
    grid_shape = template_c.shape
    c_phi, rho_phi = warp_medium(
        template_c, template_rho, pose, centre=centre, c_bg=c_bg, rho_bg=rho_bg,
    )
    domain = build_domain(grid_shape, dx)
    medium = build_medium(domain, c_phi, rho_phi, pml_size=pml_size)
    pred = simulate_shot_sensors(
        medium, time_axis, src_pos, sensor_positions_grid,
        signal, dt, checkpointed=checkpointed,
    )
    min_t = min(pred.shape[0], observed.shape[0])
    return loss_fn(pred[:min_t], observed[:min_t])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@dataclass
class MOFIConfig:
    """Configuration for MOFI pose alignment.

    Pose is optimised in scaled coordinates so rotation and translation
    are commensurate: a unit step moves the skull rim by ~1 grid point
    whether it comes from rotation or translation. ``rotation_scale`` is
    the characteristic radius (grid points) used for that scaling;
    default ≈ quarter-perimeter of the grid.
    """
    freq_bands: List[Tuple[float, float]] = field(default_factory=lambda: [
        (50e3, 100e3),
        (100e3, 200e3),
    ])
    n_iters_per_band: int = 200
    shots_per_iter: int = 8
    step_size: float = 0.5            # max pose step (scaled units ≈ grid pts) / iter
    init_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_centre: Optional[Tuple[float, float]] = None  # default = grid centre
    rotation_scale: Optional[float] = None   # grid pts; default ≈ (H+W)/4
    c_bg: float = 1500.0
    rho_bg: float = 1000.0
    pml_size: int = 10
    cfl: float = 0.3
    loss_fn: str = "l2"
    line_search: bool = True
    ls_max_tries: int = 4
    ls_shrink: float = 0.5
    # Gradient normalisation. Default False = the paper's single scalar (norm
    # of the first gradient), which is STABLE and converges (steps shrink as
    # the gradient does). Weakly-identified components (rotation) converge
    # more slowly under it; the cure is a smaller `rotation_scale` (rotation
    # step ~ 1/rotation_scale^2), NOT per-component normalisation.
    # per_component_norm=True (EXPERIMENTAL) normalises each component by its
    # own first-gradient magnitude; this decouples convergence rates but is
    # fragile — at a mis-posed start the rotation first-gradient is small and
    # noisy, so dividing by it amplifies noise and can diverge. Left in as an
    # option, not recommended.
    per_component_norm: bool = False
    skip_bandpass: bool = False
    verbose: bool = True


@dataclass
class MOFIResult:
    pose: jnp.ndarray                  # (3,) (θ_rad, δ₁_pix, δ₂_pix)
    aligned_velocity: jnp.ndarray      # template warped by the recovered pose
    aligned_density: Union[float, jnp.ndarray]
    pose_history: List[jnp.ndarray]
    loss_history: List[float]


def _get_loss_fn(name: str) -> Callable:
    return {"l2": l2_loss, "envelope": envelope_loss,
            "multiscale": lambda p, o: multiscale_loss(p, o, 0.5)}[name]


# Diagonal metric: scale θ (rad) into pixel-equivalent arc length at the
# characteristic radius, so the normalised 3-vector gradient treats
# rotation and translation on the same footing. We optimise u = S·pose
# with S = diag(rotation_scale, 1, 1); g_u = g_pose · diag(1/rs, 1, 1).
def _to_scaled(pose, rs):
    return pose * jnp.array([rs, 1.0, 1.0])

def _from_scaled(u, rs):
    return u * jnp.array([1.0 / rs, 1.0, 1.0])

def _grad_to_scaled(g_pose, rs):
    return g_pose * jnp.array([1.0 / rs, 1.0, 1.0])


def run_mofi(
    observed_data: jnp.ndarray,
    template_velocity: jnp.ndarray,
    template_density: Union[float, jnp.ndarray],
    dx: float,
    src_positions_grid: list,
    sensor_positions_grid: Tuple,
    source_signal: jnp.ndarray,
    dt: float,
    t_end: float,
    config: Optional[MOFIConfig] = None,
    key: Optional[jax.Array] = None,
) -> MOFIResult:
    """Recover the SE(2) pose of ``template_velocity`` from acoustic data.

    Mirrors ``run_fwi``: stochastic shot batches, per-shot gradient
    accumulation (O(1 shot) memory), multi-scale frequency bands, a fixed
    CFL time axis. Differs only in the optimisation variable — the 3-DOF
    pose ``φ`` rather than the voxel grid — and the update rule
    (first-gradient-norm normalisation + backtracking line search, both
    per the paper).

    Returns the recovered pose and the warped template ready to seed
    ``run_fwi``.
    """
    if config is None:
        config = MOFIConfig()
    if key is None:
        key = jr.PRNGKey(0)

    grid_shape = template_velocity.shape
    assert len(grid_shape) == 2, "SE(2) MOFI is 2D; SE(3) is the noted extension."
    H, W = grid_shape
    n_sources = len(src_positions_grid)
    loss_fn = _get_loss_fn(config.loss_fn)
    centre = config.rotation_centre or ((H - 1) / 2.0, (W - 1) / 2.0)
    rs = config.rotation_scale or (H + W) / 4.0

    pose = jnp.asarray(config.init_pose, dtype=jnp.float32)

    # Fixed time axis at a reference speed (max of template + background) for
    # CFL stability — identical pattern to run_fwi.
    c_ref = float(max(float(jnp.max(template_velocity)), config.c_bg))
    ref_domain = build_domain(grid_shape, dx)
    ref_medium = build_medium(ref_domain, c_ref, config.rho_bg, pml_size=config.pml_size)
    fixed_time_axis = build_time_axis(ref_medium, cfl=config.cfl, t_end=t_end)

    def batch_loss_and_grad(p, shot_idx, bp_signal, bp_observed):
        """Accumulate (loss, ∂f/∂φ) over a shot batch, one shot at a time."""
        total_loss, grad_accum = 0.0, jnp.zeros(3)
        for si in shot_idx:
            obs = bp_observed[int(si)]
            src = src_positions_grid[int(si)]

            def f(pp):
                return mofi_pose_loss(
                    pp, template_velocity, template_density, dx, src,
                    sensor_positions_grid, bp_signal, dt, fixed_time_axis, obs,
                    centre=centre, c_bg=config.c_bg, rho_bg=config.rho_bg,
                    pml_size=config.pml_size, loss_fn=loss_fn,
                )

            l, g = jax.value_and_grad(f)(p)
            total_loss = total_loss + float(l)
            grad_accum = grad_accum + g
        n = len(shot_idx)
        return total_loss / n, grad_accum / n

    def batch_loss(p, shot_idx, bp_signal, bp_observed):
        """Loss only (for line-search trial evaluations)."""
        tot = 0.0
        for si in shot_idx:
            tot += float(mofi_pose_loss(
                p, template_velocity, template_density, dx,
                src_positions_grid[int(si)], sensor_positions_grid,
                bp_signal, dt, fixed_time_axis, bp_observed[int(si)],
                centre=centre, c_bg=config.c_bg, rho_bg=config.rho_bg,
                pml_size=config.pml_size, loss_fn=loss_fn,
            ))
        return tot / len(shot_idx)

    pose_history, loss_history = [pose], []
    g0 = None  # first-gradient normaliser: scalar (paper) or per-component vector

    if config.verbose:
        print(f"MOFI: {len(config.freq_bands)} bands × {config.n_iters_per_band} iters, "
              f"{config.shots_per_iter} shots/iter, rot_centre={centre}, rot_scale={rs:.1f}")

    from .fwi import _bandpass_signal  # reuse the exact band filter run_fwi uses

    for band_idx, (fmin, fmax) in enumerate(config.freq_bands):
        if config.skip_bandpass:
            bp_signal, bp_observed = source_signal, observed_data
        else:
            bp_signal = _bandpass_signal(source_signal, dt, fmin, fmax)
            bp_observed = jax.vmap(
                lambda d: jax.vmap(lambda col: _bandpass_signal(col, dt, fmin, fmax))(d.T).T
            )(observed_data)

        if config.verbose:
            print(f"\n  Band {band_idx+1}/{len(config.freq_bands)}: "
                  f"{fmin/1e3:.0f}-{fmax/1e3:.0f} kHz")

        for it in range(config.n_iters_per_band):
            key, subkey = jr.split(key)
            if config.shots_per_iter >= n_sources:
                shot_idx = np.arange(n_sources)
            else:
                shot_idx = np.array(jr.choice(
                    subkey, n_sources, shape=(config.shots_per_iter,), replace=False))

            loss_val, grad = batch_loss_and_grad(pose, shot_idx, bp_signal, bp_observed)
            loss_history.append(loss_val)

            # Work in scaled (pixel-commensurate) coordinates.
            g_u = _grad_to_scaled(grad, rs)
            if g0 is None:
                if config.per_component_norm:
                    a = jnp.abs(g_u)
                    g0 = a + 1e-6 * jnp.max(a)     # per-component first-grad magnitude
                else:
                    g0 = jnp.linalg.norm(g_u) + 1e-30   # paper: single first-grad norm
            direction_u = g_u / g0                 # scalar or per-component normalise

            # Backtracking line search on the same shot batch.
            step = config.step_size
            if config.line_search:
                u = _to_scaled(pose, rs)
                accepted = False
                for _ in range(config.ls_max_tries):
                    trial_pose = _from_scaled(u - step * direction_u, rs)
                    trial_loss = batch_loss(trial_pose, shot_idx, bp_signal, bp_observed)
                    if trial_loss < loss_val:
                        pose, accepted = trial_pose, True
                        break
                    step *= config.ls_shrink
                if not accepted:           # take the smallest step anyway
                    pose = _from_scaled(u - step * direction_u, rs)
            else:
                u = _to_scaled(pose, rs)
                pose = _from_scaled(u - step * direction_u, rs)

            pose_history.append(pose)
            if config.verbose and (it + 1) % 25 == 0:
                th, d1, d2 = float(pose[0]), float(pose[1]), float(pose[2])
                print(f"    iter {it+1}: loss={loss_val:.6e}  "
                      f"θ={np.degrees(th):+.2f}°  δ=({d1:+.2f},{d2:+.2f}) px")

    aligned_c, aligned_rho = warp_medium(
        template_velocity, template_density, pose,
        centre=centre, c_bg=config.c_bg, rho_bg=config.rho_bg,
    )
    return MOFIResult(
        pose=pose,
        aligned_velocity=aligned_c,
        aligned_density=aligned_rho,
        pose_history=pose_history,
        loss_history=loss_history,
    )

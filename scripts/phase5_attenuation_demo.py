"""Phase 5 productive-direction demo (frequency-domain).

Goal: show that j-Wave's frequency-domain solver responds to
``medium.attenuation``, that the response is large enough to matter
for transcranial imaging, and — *with the fork-pinned
``alpha_power``* — that the y-exponent of the absorption power law is
configurable. This is what Phase 5's CANN α(ω) needs to plug into.

Setup (2D, ITRUSST BM3-flavoured flat three-layer skull plate):

    z = 0      ────── source @ (40, 10) mm
                     water
    z = 30 mm ──── outer cortical (2 mm)
                    diploe        (3 mm)
              ──── inner cortical (2 mm)
                    water
    z = 70 mm ────── receiver line, 5 mm spacing

Domain: 80 mm × 80 mm at dx = 0.5 mm = (160, 160) cells.
PML: 16 cells (j-Wave's, on the outer 16 rows of the grid).
Source/receivers placed inside the inner 128 × 128 region.
Frequency: 500 kHz.

Cases:
  - lossless                         (α_cort = α_dip = 0)
  - lossy y=2 (Stokes / fluid)       (α_cort = 4, α_dip = 8 dB/cm/MHz²)
  - lossy y=1.1 (tissue-realistic)   (α_cort = 4, α_dip = 8 dB/cm/MHz¹·¹)

Reports |p| at receivers and rel-L2 vs lossless. Tissue-realistic y
should give roughly equivalent attenuation magnitude at 500 kHz to
the y=2 case (different units), but with a different frequency
dependence — the demo's numerical value of `α` isn't comparable
across y. The point is: BOTH are non-trivially different from
lossless, and the y=1.1 case requires the fork's `alpha_power`.

Run:
    uv run python scripts/phase5_attenuation_demo.py
"""

from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np


def build_phantom_2d(grid_shape, dx, pml, *, alpha_cortical, alpha_diploe):
    """ITRUSST-BM3-flavoured 2D flat skull plate, full grid (PML cells
    are plain water — j-Wave applies its own absorbing layer)."""
    nx, nz = grid_shape
    c = np.full(grid_shape, 1500.0, dtype=np.float32)
    rho = np.full(grid_shape, 1000.0, dtype=np.float32)
    alpha = np.zeros(grid_shape, dtype=np.float32)

    outer_t = int(round(0.002 / dx))
    diploe_t = int(round(0.003 / dx))
    inner_t = int(round(0.002 / dx))
    total = outer_t + diploe_t + inner_t

    z_mid = nz // 2
    z0 = z_mid - total // 2
    z1 = z0 + outer_t
    z2 = z1 + diploe_t
    z3 = z2 + inner_t

    # Skull plate spans the full x-extent except the PML borders, so the
    # plate doesn't touch the absorbing layer.
    x_lo, x_hi = pml, nx - pml
    c[x_lo:x_hi, z0:z1] = 2800.0
    rho[x_lo:x_hi, z0:z1] = 1850.0
    alpha[x_lo:x_hi, z0:z1] = alpha_cortical
    c[x_lo:x_hi, z1:z2] = 2300.0
    rho[x_lo:x_hi, z1:z2] = 1700.0
    alpha[x_lo:x_hi, z1:z2] = alpha_diploe
    c[x_lo:x_hi, z2:z3] = 2800.0
    rho[x_lo:x_hi, z2:z3] = 1850.0
    alpha[x_lo:x_hi, z2:z3] = alpha_cortical

    return c, rho, alpha


def run_helmholtz(c, rho, alpha, dx, freq_hz, src_xz, recv_xz, *,
                   pml=16, alpha_power=2.0):
    """Run j-Wave's frequency-domain Helmholtz solver and return
    complex pressure at the receiver positions."""
    from jwave import FourierSeries
    from jwave.acoustics.time_harmonic import helmholtz_solver
    from jwave.geometry import Medium
    from jaxdf.geometry import Domain

    grid_shape = c.shape
    domain = Domain(grid_shape, (dx,) * len(grid_shape))

    def to_field(arr):
        return FourierSeries(jnp.asarray(arr)[..., jnp.newaxis], domain)

    has_alpha = float(np.max(alpha)) > 0
    medium = Medium(
        domain=domain,
        sound_speed=to_field(c),
        density=to_field(rho),
        attenuation=to_field(alpha) if has_alpha else 0.0,
        pml_size=pml,
        alpha_power=alpha_power,
    )
    omega = 2.0 * jnp.pi * freq_hz

    src_arr = jnp.zeros(grid_shape, dtype=jnp.complex64)
    src_arr = src_arr.at[src_xz[0], src_xz[1]].set(1.0 + 0.0j)
    src_field = FourierSeries(src_arr[..., jnp.newaxis], domain)

    p = helmholtz_solver(
        medium, omega, src_field,
        method="gmres", tol=1e-3, maxiter=2000, restart=20,
    )
    p_arr = np.asarray(p.on_grid).squeeze()
    rx, rz = recv_xz
    return p_arr[rx, rz]


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main():
    dx = 5e-4              # 0.5 mm grid
    grid_shape = (160, 160)
    pml = 16

    src_xz = (grid_shape[0] // 2, pml + int(round(0.010 / dx)))
    recv_x_mm = np.arange(20, 61, 5)
    recv_x = (recv_x_mm * 1e-3 / dx).astype(int)
    recv_z = grid_shape[1] - pml - int(round(0.010 / dx))
    recv_z_arr = np.full_like(recv_x, recv_z)

    print(f"Domain: {grid_shape[0]}×{grid_shape[1]} cells at dx={dx*1e3:.2f} mm "
          f"(j-Wave PML = {pml} cells outer)")
    inner_x_mm = (grid_shape[0] - 2 * pml) * dx * 1e3
    inner_z_mm = (grid_shape[1] - 2 * pml) * dx * 1e3
    print(f"Inner usable area: {inner_x_mm:.0f}×{inner_z_mm:.0f} mm")
    print(f"Source: cell {src_xz} = "
          f"({src_xz[0]*dx*1e3:.1f}, {src_xz[1]*dx*1e3:.1f}) mm")
    print(f"Receivers: x cells {recv_x.tolist()}, z = "
          f"{recv_z*dx*1e3:.1f} mm")
    print(f"Frequency: 500 kHz, λ_water = {1500/500e3*1e3:.1f} mm = "
          f"{1500/500e3/dx:.0f} cells/wavelength")

    cases = [
        # (name,                    a_cort, a_dip, alpha_power)
        ("lossless",                 0.0,    0.0,  2.0),
        ("lossy  y=2.0 (Stokes)",    4.0,    8.0,  2.0),
        ("lossy  y=1.1 (tissue)",    4.0,    8.0,  1.1),
    ]

    results = {}
    for name, a_cort, a_dip, y in cases:
        print(f"\n--- {name} (α_cort={a_cort}, α_dip={a_dip} dB/cm/MHz^y, y={y}) ---")
        c, rho, alpha = build_phantom_2d(
            grid_shape, dx, pml,
            alpha_cortical=a_cort, alpha_diploe=a_dip,
        )
        t0 = time.time()
        p_recv = run_helmholtz(
            c, rho, alpha, dx, 500e3,
            src_xz, (recv_x, recv_z_arr),
            pml=pml, alpha_power=y,
        )
        dt_run = time.time() - t0
        amp = np.abs(p_recv)
        phase_deg = np.unwrap(np.angle(p_recv)) * 180 / np.pi
        print(f"  ran in {dt_run:.1f}s; |p| = "
              f"{amp.min():.3e} – {amp.max():.3e}  median={np.median(amp):.3e}")
        print(f"  phase span = {phase_deg.max() - phase_deg.min():.1f}°")
        results[name] = p_recv

    print("\n" + "=" * 78)
    print(f"{'comparison':36s}{'rel-L2':>10s}"
          f"{'med |p| ratio':>16s}{'phase shift':>16s}")
    print("=" * 78)
    base = results["lossless"]
    base_phase = np.unwrap(np.angle(base))
    for name in ["lossy  y=2.0 (Stokes)", "lossy  y=1.1 (tissue)"]:
        p = results[name]
        amp_ratio = float(np.median(np.abs(p)) / (np.median(np.abs(base)) + 1e-30))
        rl2 = rel_l2(p, base)
        phase_shift = float(np.median(
            np.unwrap(np.angle(p)) - base_phase
        ) * 180 / np.pi)
        print(f"{name + ' vs lossless':36s}"
              f"{rl2:>10.3f}{amp_ratio:>16.3f}{phase_shift:>14.1f}°")
    print("=" * 78)

    print("\nINTERPRETATION")
    print("-" * 78)
    print("- |p| ratio < 1 means attenuation reduced amplitude (expected).")
    print("- Phase shift (in degrees) reflects K-K dispersion — even at constant")
    print("  α, the y exponent shifts the effective phase velocity.")
    print("- A non-zero rel-L2 between y=2 and y=1.1 with the same α coefficients")
    print("  shows the alpha_power knob actually does work; this is the pivot")
    print("  point for plugging in Phase 5's CANN-derived y(tissue).")


if __name__ == "__main__":
    main()

"""Absorption-aware FWI demo: a known skull alpha removes the amplitude bias.

A real skull attenuates ultrasound by ~10-20% per pass (power-law absorption),
so transcranial data arrives with the skull's absorption baked in. If the FWI
forward model is *lossless* it can never match those attenuated amplitudes no
matter the velocity, so the inversion is forced to an irreducible misfit floor
and tries to explain the missing amplitude with spurious brain structure.

The clinical setup: the skull (geometry, density, AND attenuation) is known from
CT. We hold the skull FIXED and image the brain *through* it. Freezing the skull
is what makes this a clean test — the skull-transmission amplitude can only be
explained by modeling alpha, so the absorption term has nowhere to hide.

This runs the SAME 2D ring-array brain reconstruction twice from an identical
start (homogeneous brain, true skull) — once with a lossless forward, once with
the known alpha (Treeby-Cox absorbing EoS) — and compares the recovered brain
sound-speed anomalies and the final data misfit. Only the forward absorption
differs.

Run:  uv run python examples/05_absorption_aware_fwi.py
      (PYTHONPATH=~/dev/jwave if using the m9h/jwave fork directly)

Writes a figure + metrics to results/absorption_aware_fwi/.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import jax.random as jr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, generate_observed_data,
)
from brain_fwi.utils.wavelets import ricker_wavelet
from brain_fwi.inversion.fwi import FWIConfig, run_fwi


# ---------------------------------------------------------------------------
# Geometry / acoustic constants
# ---------------------------------------------------------------------------
N = 120                      # grid (N x N)
DX = 5e-4                    # 0.5 mm  -> 60 mm field of view
CX = CY = N // 2
Y_POWER = 1.1               # tissue/bone power-law exponent
F0 = 250e3                  # Ricker peak frequency
PML = 12
CFL = 0.3
T_END = 5e-5
C_MIN, C_MAX = 1400.0, 3200.0

R_SKULL_OUT = 40            # skull annulus (voxels): 20.0 mm
R_SKULL_IN = 33            # 16.5 mm  -> ~3.5 mm thick skull
R_RING = 46                # transducer ring radius (voxels): 23 mm

# Acoustic properties (ITRUSST-ish): water / brain / cortical bone
C_WATER, RHO_WATER, A_WATER = 1500.0, 1000.0, 0.0
C_BRAIN, RHO_BRAIN, A_BRAIN = 1540.0, 1000.0, 0.5
C_SKULL, RHO_SKULL, A_SKULL = 2800.0, 1850.0, 8.0   # dB/cm/MHz^1.1


def build_phantom():
    """True medium + masks. Brain carries two sound-speed anomalies to
    reconstruct; the skull is the (known) attenuator we image through."""
    yy, xx = np.mgrid[0:N, 0:N]
    R = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)

    c = np.full((N, N), C_WATER, np.float32)
    rho = np.full((N, N), RHO_WATER, np.float32)
    alpha = np.full((N, N), A_WATER, np.float32)

    brain = R < R_SKULL_IN
    skull = (R >= R_SKULL_IN) & (R < R_SKULL_OUT)
    c[brain], rho[brain], alpha[brain] = C_BRAIN, RHO_BRAIN, A_BRAIN
    c[skull], rho[skull], alpha[skull] = C_SKULL, RHO_SKULL, A_SKULL

    # Two brain velocity anomalies (the imaging targets).
    a1 = np.sqrt((xx - CX - 9) ** 2 + (yy - CY + 7) ** 2) < 8
    a2 = np.sqrt((xx - CX + 10) ** 2 + (yy - CY + 9) ** 2) < 7
    c[a1 & brain] = 1610.0
    c[a2 & brain] = 1485.0

    invert = (R < (R_SKULL_IN - 2)).astype(np.float32)   # brain interior only
    roi = R < (R_SKULL_IN - 3)                            # metric ROI
    return c, rho, alpha, brain, invert, roi


def ring(n, r):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.round(CX + r * np.cos(th)).astype(int)
    ys = np.round(CY + r * np.sin(th)).astype(int)
    return xs, ys


def roi_rmse(c_est, c_true, mask):
    d = (np.asarray(c_est) - np.asarray(c_true))[mask]
    return float(np.sqrt(np.mean(d ** 2)))


def main():
    out = Path("results/absorption_aware_fwi")
    out.mkdir(parents=True, exist_ok=True)

    c_true, rho_true, alpha_true, brain, invert, roi = build_phantom()
    c_true_j = jnp.asarray(c_true)
    rho_true_j = jnp.asarray(rho_true)
    alpha_true_j = jnp.asarray(alpha_true)

    sx, sy = ring(16, R_RING)
    src_positions = [(int(x), int(y)) for x, y in zip(sx, sy)]
    rx, ry = ring(48, R_RING)
    sensor_positions = (rx, ry)

    # Time axis exactly as run_fwi derives it (reference medium at c_max).
    ref_med = build_medium(build_domain((N, N), DX), C_MAX, RHO_WATER, pml_size=PML)
    ta = build_time_axis(ref_med, cfl=CFL, t_end=T_END)
    dt = float(ta.dt); nt = int(ta.Nt)
    sig = ricker_wavelet(F0, dt, nt)
    print(f"Grid {N}x{N}, dx={DX*1e3:.2f} mm, dt={dt*1e9:.1f} ns, Nt={nt}, "
          f"{len(src_positions)} src / {len(rx)} recv")

    # --- Observed data: TRUE medium WITH skull absorption ---
    t0 = time.time()
    print("Generating observed data (with skull absorption) ...")
    observed = generate_observed_data(
        c_true_j, rho_true_j, DX, src_positions, sensor_positions, F0,
        pml_size=PML, time_axis=ta, source_signal=sig, dt=dt,
        attenuation=alpha_true_j, alpha_power=Y_POWER, verbose=False,
    )
    print(f"  done ({time.time()-t0:.0f}s)")

    # --- Identical start: homogeneous brain, TRUE skull + water (frozen) ---
    c_init = c_true.copy()
    c_init[brain] = C_BRAIN          # remove the anomalies; skull/water stay true
    c_init_j = jnp.asarray(c_init)
    mask_j = jnp.asarray(invert)

    bands = [(150e3, 250e3), (250e3, 400e3)]
    common = dict(
        freq_bands=bands, n_iters_per_band=15, shots_per_iter=8,
        learning_rate=30.0, c_min=C_MIN, c_max=C_MAX, pml_size=PML, cfl=CFL,
        gradient_smooth_sigma=1.0, mask=mask_j, loss_fn="l2", verbose=True,
    )

    def fwi(attenuation, tag):
        print(f"\n=== FWI [{tag}] ===")
        t0 = time.time()
        res = run_fwi(
            observed, c_init_j, rho_true_j, DX, src_positions, sensor_positions,
            sig, dt, T_END,
            config=FWIConfig(attenuation=attenuation, alpha_power=Y_POWER, **common),
            key=jr.PRNGKey(0),
        )
        print(f"  [{tag}] {time.time()-t0:.0f}s, final misfit {res.loss_history[-1]:.4e}")
        return np.asarray(res.velocity), float(res.loss_history[-1])

    c_lossless, misfit_lossless = fwi(None, "lossless forward")
    c_aware, misfit_aware = fwi(alpha_true_j, "absorption-aware (known alpha)")

    # --- Metrics: brain-interior velocity RMSE + data-misfit floor ---
    rmse_init = roi_rmse(c_init, c_true, roi)
    rmse_lossless = roi_rmse(c_lossless, c_true, roi)
    rmse_aware = roi_rmse(c_aware, c_true, roi)
    print("\n" + "=" * 60)
    print(f"{'model':<28}{'brain RMSE (m/s)':>18}{'misfit':>12}")
    print(f"{'starting model':<28}{rmse_init:>18.2f}{'-':>12}")
    print(f"{'lossless FWI':<28}{rmse_lossless:>18.2f}{misfit_lossless:>12.3e}")
    print(f"{'absorption-aware FWI':<28}{rmse_aware:>18.2f}{misfit_aware:>12.3e}")
    print("-" * 60)
    impr_rmse = 100.0 * (rmse_lossless - rmse_aware) / rmse_lossless
    impr_misfit = 100.0 * (misfit_lossless - misfit_aware) / misfit_lossless
    print(f"absorption-aware vs lossless: brain RMSE {impr_rmse:+.0f}%, "
          f"data misfit {impr_misfit:+.0f}%")
    print("=" * 60)

    # --- Figure ---
    vmin, vmax = 1470, 1620
    emax = 90
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
    # show only the head region for the brain panels
    sl = slice(CX - R_SKULL_OUT - 4, CX + R_SKULL_OUT + 4)
    panels = [
        (ax[0, 0], c_true, "truth c", vmin, vmax, "viridis"),
        (ax[0, 1], c_lossless, f"lossless FWI (RMSE {rmse_lossless:.1f})", vmin, vmax, "viridis"),
        (ax[0, 2], c_aware, f"absorption-aware FWI (RMSE {rmse_aware:.1f})", vmin, vmax, "viridis"),
        (ax[1, 0], c_init, f"start: homog. brain (RMSE {rmse_init:.1f})", vmin, vmax, "viridis"),
        (ax[1, 1], np.abs(c_lossless - c_true), "lossless |error|", 0, emax, "magma"),
        (ax[1, 2], np.abs(c_aware - c_true), "absorption-aware |error|", 0, emax, "magma"),
    ]
    for a, img, title, lo, hi, cmap in panels:
        im = a.imshow(img[sl, sl].T, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
        a.set_title(title, fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Absorption-aware FWI: image the brain through a known skull "
        "(only the forward skull-alpha differs)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig_path = out / "absorption_aware_fwi.png"
    fig.savefig(fig_path, dpi=130)
    print(f"\nFigure -> {fig_path}")

    np.savez(
        out / "absorption_aware_fwi.npz",
        c_true=c_true, c_init=c_init, c_lossless=c_lossless, c_aware=c_aware,
        alpha_true=alpha_true, roi=roi,
        rmse_init=rmse_init, rmse_lossless=rmse_lossless, rmse_aware=rmse_aware,
        misfit_lossless=misfit_lossless, misfit_aware=misfit_aware,
    )


if __name__ == "__main__":
    main()

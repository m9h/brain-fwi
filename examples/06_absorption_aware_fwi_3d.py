"""3D absorption-aware FWI on a real patient head (Birnbaum), helmet array.

The 3D counterpart of examples/05. Images a real stroke-patient brain — including
the lesion — through that patient's own skull, with whole-head helmet coverage
(many more sensors than a 2D ring). The skull (c, rho, AND alpha, as if from CT)
is held FIXED and we invert only the brain interior; the only difference between
the two runs is whether the known skull alpha is in the forward model.

Why freezing the skull matters: a real skull attenuates, so the recorded data
arrives quieter. With the skull frozen, a *lossless* forward cannot match those
amplitudes and is driven to an irreducible misfit floor, back-projecting the
deficit into spurious brain structure. Modeling the known alpha (Treeby-Cox
absorbing EoS, now differentiable through the checkpointed FWI path) removes the
bias and recovers a clean image — the lesion included.

Run (validate the pipeline fast, ~minutes):
    PYTHONPATH=~/dev/jwave uv run python examples/06_absorption_aware_fwi_3d.py --smoke
Full run (96^3, ~1-2 h on a GB10):
    PYTHONPATH=~/dev/jwave uv run python examples/06_absorption_aware_fwi_3d.py

Writes a figure + metrics to results/absorption_aware_fwi_3d/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import jax.random as jr
from scipy.ndimage import zoom, binary_erosion, label as cc_label

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from brain_fwi.phantoms import birnbaum as B
from brain_fwi.transducers.helmet import helmet_array_3d, transducer_positions_to_grid
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, generate_observed_data,
    _build_source_signal,
)
from brain_fwi.inversion.fwi import FWIConfig, run_fwi


# Birnbaum labels: 0=bg 1=lesion 2/3/4=brain(GM/WM/deep) 5=skull 6=scalp/other.
# Give the three brain labels a small realistic velocity spread so there is true
# intra-brain structure to recover (not just the lesion). Skull alpha=6 dB/cm is
# realistic and inside the stable band (alpha>~10 destabilises the EoS).
C_MAP = {0: 1500.0, 1: 1640.0, 2: 1555.0, 3: 1575.0, 4: 1562.0, 5: 2800.0, 6: 1500.0}
RHO_MAP = {0: 1000.0, 1: 1040.0, 2: 1040.0, 3: 1040.0, 4: 1040.0, 5: 1850.0, 6: 1000.0}
A_MAP = {0: 0.0, 1: 0.6, 2: 0.6, 3: 0.6, 4: 0.6, 5: 6.0, 6: 0.0}
C_BRAIN0 = 1560.0   # homogeneous starting velocity inside the brain
Y_POWER = 1.1
C_MIN, C_MAX = 1400.0, 3000.0


def _lut(d, labels):
    out = np.zeros(labels.shape, np.float32)
    for k, v in d.items():
        out[labels == k] = v
    return out


def load_head(N, subject, margin_mm=22.0):
    """Crop a cube around the brain (+ skull shell + water margin) from a real
    Birnbaum head and resample to N^3. Returns c, rho, alpha, labels, dx."""
    import nibabel as nib
    f = B.label_files()[subject]
    vol = np.round(np.asarray(nib.load(f).get_fdata())).astype(np.int32)  # 1mm iso

    brain = np.isin(vol, B.BRAIN) | (vol == B.LESION)
    xs, ys, zs = np.where(brain)
    ctr = np.array([xs.mean(), ys.mean(), zs.mean()])
    half = max(np.ptp(xs), np.ptp(ys), np.ptp(zs)) / 2.0 + margin_mm  # +skull+water
    lo = np.floor(ctr - half).astype(int)
    hi = np.ceil(ctr + half).astype(int)
    lo = np.maximum(lo, 0); hi = np.minimum(hi, np.array(vol.shape))
    cube = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    # resample labels to N^3 (nearest preserves the discrete tissue map)
    factors = [N / s for s in cube.shape]
    labels = zoom(cube, factors, order=0).astype(np.int32)
    side_mm = float((hi - lo).max())          # cube physical side (mm, 1mm voxels)
    dx = (side_mm / N) * 1e-3
    return _lut(C_MAP, labels), _lut(RHO_MAP, labels), _lut(A_MAP, labels), labels, dx


def synthetic_head(N, fov_m=0.22):
    """Self-contained anatomically-informed head in the Birnbaum label scheme
    (0=water 1=lesion 2=GM 3=WM 4=deep 5=skull 6=scalp). No patient data —
    safe to ship to a cloud GPU. Returns c, rho, alpha, labels, dx."""
    dx = fov_m / N
    cx = N // 2
    a = min(0.085 / dx, cx - 3); b = min(0.070 / dx, cx - 3); c_ = min(0.082 / dx, cx - 3)
    gx, gy, gz = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    r = np.sqrt(((gx - cx) / a) ** 2 + ((gy - cx) / b) ** 2 + ((gz - cx) / c_) ** 2)
    sc, sk, cs, co = (0.003 / (a * dx), 0.007 / (a * dx),
                      0.002 / (a * dx), 0.004 / (a * dx))
    r_sc = 1.0; r_sko = r_sc - sc; r_ski = r_sko - sk; r_csi = r_ski - cs
    r_coi = r_csi - co
    lab = np.zeros((N, N, N), np.int32)
    lab[r <= r_sc] = 6           # scalp
    lab[r <= r_sko] = 5          # skull
    lab[r <= r_ski] = 0          # CSF/subarachnoid -> water
    lab[r <= r_csi] = 2          # grey matter
    lab[r <= r_coi] = 3          # white matter
    lab[r <= r_coi * 0.45] = 4   # deep gray (3rd brain velocity)
    # lateral ventricles (CSF -> water)
    for off in (-0.014 / dx, 0.014 / dx):
        v = np.sqrt(((gx - cx) / (0.010 / dx)) ** 2 + ((gy - cx - off) / (0.008 / dx)) ** 2
                    + ((gz - cx + 0.004 / dx) / (0.024 / dx)) ** 2)
        lab[(v <= 1.0) & (r <= r_coi)] = 0
    # haemorrhage-like lesion (off-centre)
    les = np.sqrt(((gx - cx + 0.020 / dx) / (0.007 / dx)) ** 2
                  + ((gy - cx + 0.012 / dx) / (0.007 / dx)) ** 2
                  + ((gz - cx) / (0.007 / dx)) ** 2)
    lab[(les <= 1.0) & (r <= r_coi)] = 1
    return _lut(C_MAP, lab), _lut(RHO_MAP, lab), _lut(A_MAP, lab), lab, dx


MIDA_HEAD_PATH = "/data/datasets/MIDAv1-0/MIDA_v1.0/MIDA_v1_voxels/MIDA_v1.nii"


def load_mida_head(N, mida_path=MIDA_HEAD_PATH, margin_mm=20.0):
    """Crop the cranial vault from the MIDA ITRUSST head (480^3 @ 0.5 mm) and
    resample to N^3. Real 3-layer skull (cortical tables + trabecular diploe) +
    intracranial brain/CSF/ventricles/vessels from MIDA's acoustic mapping;
    water coupling outside the skull+brain. Returned in the Birnbaum label
    scheme (5=skull, 2=brain) so the rest of the demo is unchanged."""
    from brain_fwi.phantoms.mida import (
        load_mida_volume, map_mida_labels_to_acoustic, MIDA_TISSUE_GROUPS)
    g = MIDA_TISSUE_GROUPS
    skull_labels = list(set(g.get("cortical_bone", [])) | set(g.get("trabecular_bone", [])))
    brain_labels = []
    for grp in ("grey_matter", "white_matter", "csf", "blood_vessels", "dura"):
        brain_labels += g.get(grp, [])
    vol = np.asarray(load_mida_volume(Path(mida_path)))                 # 0.5 mm voxels
    skull = np.isin(vol, skull_labels)
    soft = np.isin(vol, brain_labels)                                   # intracranial soft tissue
    # Crop bbox from the cerebral PARENCHYMA mass only (GM+WM) so the brainstem /
    # spinal CSF sprawl doesn't inflate the cube -> keeps dx fine.
    parench = np.isin(vol, list(set(g.get("grey_matter", [])) | set(g.get("white_matter", []))))
    lab, n = cc_label(parench)
    if n > 1:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        parench = lab == int(sizes.argmax())

    xs, ys, zs = np.where(parench)
    ctr = np.array([xs.mean(), ys.mean(), zs.mean()])
    half = max(np.ptp(xs), np.ptp(ys), np.ptp(zs)) / 2.0 + margin_mm / 0.5
    lo = np.maximum(np.floor(ctr - half).astype(int), 0)
    hi = np.minimum(np.ceil(ctr + half).astype(int), np.array(vol.shape))
    sub = lambda a: a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    fac = [N / s for s in sub(vol).shape]
    lab_n = zoom(sub(vol), fac, order=0).astype(np.int32)
    skn = zoom(sub(skull).astype(np.float32), fac, order=0) > 0.5
    brn = zoom(sub(soft).astype(np.float32), fac, order=0) > 0.5

    props = map_mida_labels_to_acoustic(lab_n)
    inside = skn | brn
    c = np.where(inside, np.asarray(props["sound_speed"]), 1500.0).astype(np.float32)
    rho = np.where(inside, np.asarray(props["density"]), 1000.0).astype(np.float32)
    alpha = np.where(inside, np.asarray(props["attenuation"]), 0.0).astype(np.float32)
    geom = np.zeros((N, N, N), np.int32); geom[brn] = 2; geom[skn] = 5
    dx = (float((hi - lo).max()) * 0.5 / N) * 1e-3
    return c, rho, alpha, geom, dx


def brain_roi(labels):
    """Largest connected brain+lesion component, eroded 1 voxel off the skull."""
    roi = np.isin(labels, B.BRAIN) | (labels == B.LESION)
    lab, n = cc_label(roi)
    if n > 1:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        roi = lab == int(sizes.argmax())
    return binary_erosion(roi, iterations=1)


def make_helmet(labels, dx, n_elem, n_src):
    """Helmet receivers (all elements) + a source subset, snapped to water just
    outside the skull. Returns (src_positions list, sensor_positions tuple)."""
    N = labels.shape[0]
    skull = labels == B.SKULL
    xs, ys, zs = np.where(skull)
    ctr_m = (np.array([xs.mean(), ys.mean(), zs.mean()]) * dx)
    # skull outer extent -> helmet radius just beyond it (in the water margin)
    rad = (max(np.ptp(xs), np.ptp(ys), np.ptp(zs)) / 2.0 + 4) * dx
    pos = np.asarray(helmet_array_3d(
        n_elements=n_elem, center=tuple(ctr_m),
        radius_ap=rad, radius_lr=rad, radius_si=rad, standoff=0.0,
        coverage_angle=3.1416, exclude_face=False,
    ))
    grid = transducer_positions_to_grid(jnp.asarray(pos), dx, (N, N, N))
    gx, gy, gz = (np.array(g, dtype=int) for g in grid)  # writable copies

    # Snap any element that landed in solid tissue radially outward to water.
    solid = (labels == B.SKULL) | np.isin(labels, B.BRAIN) | (labels == B.LESION)
    cgrid = np.array([xs.mean(), ys.mean(), zs.mean()])
    for i in range(len(gx)):
        p = np.array([gx[i], gy[i], gz[i]], float)
        d = p - cgrid; d /= (np.linalg.norm(d) + 1e-9)
        for _ in range(N):
            ix, iy, iz = np.clip(np.round(p).astype(int), 0, N - 1)
            if not solid[ix, iy, iz]:
                gx[i], gy[i], gz[i] = ix, iy, iz
                break
            p += d
    sensor_positions = (gx.astype(int), gy.astype(int), gz.astype(int))
    src_idx = np.linspace(0, len(gx) - 1, n_src).astype(int)
    src_positions = [(int(gx[i]), int(gy[i]), int(gz[i])) for i in src_idx]
    return src_positions, sensor_positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny fast end-to-end check")
    ap.add_argument("--full", action="store_true", help="192^3 preset (top-end GPU)")
    ap.add_argument("--phantom", choices=["birnbaum", "synthetic", "mida"], default="birnbaum",
                    help="synthetic = self-contained (cloud-safe); mida = ITRUSST head; "
                         "birnbaum = real patient (+lesion). mida/birnbaum are local data only.")
    ap.add_argument("--n", type=int, default=None, help="grid size override")
    ap.add_argument("--subject", type=int, default=0)
    ap.add_argument("--sources", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--smooth", type=float, default=None,
                    help="gradient-smoothing sigma (grid pts); overrides preset")
    ap.add_argument("--precondition", action="store_true",
                    help="force tempered pseudo-Hessian preconditioning on")
    ap.add_argument("--precond-floor", type=float, default=0.05,
                    help="illumination water-level (frac of peak); ~0.05 tempers "
                         "periphery-ring vs deep-brain-speckle at high res")
    args = ap.parse_args()

    # precond: OFF at low res (uniform illumination -> clean without it); tempered
    # ON at 192^3, where un-preconditioned gradients over-update the bright
    # near-skull periphery into a ring (and full normalisation speckles the
    # deep brain) -- the water-level floor balances the two.
    if args.smoke:
        N, n_elem, n_src, n_iters, bands, shots, smooth, precond = 48, 32, 4, 2, [(40e3, 80e3)], 4, 1.0, False
    elif args.full:
        N, n_elem, n_src, n_iters, smooth, precond = 192, 256, 32, 14, 2.0, True
        bands, shots = [(50e3, 100e3), (100e3, 180e3), (180e3, 280e3)], 12
    else:
        N, n_elem, n_src, n_iters, smooth, precond = 96, 160, 16, 10, 1.5, False
        bands, shots = [(50e3, 100e3), (100e3, 160e3)], 8
    if args.n: N = args.n
    if args.sources: n_src = args.sources
    if args.iters: n_iters = args.iters
    if args.smooth is not None: smooth = args.smooth
    precond = precond or args.precondition

    out = Path("results/absorption_aware_fwi_3d"); out.mkdir(parents=True, exist_ok=True)
    f0 = max(fmax for _, fmax in bands)

    if args.phantom == "synthetic":
        print(f"[1/4] Building synthetic anatomical head -> {N}^3 ...")
        c_true, rho_true, alpha_true, labels, dx = synthetic_head(N)
    elif args.phantom == "mida":
        print(f"[1/4] Loading MIDA ITRUSST head -> {N}^3 ...")
        c_true, rho_true, alpha_true, labels, dx = load_mida_head(N)
    else:
        print(f"[1/4] Loading real head (Birnbaum subj {args.subject}) -> {N}^3 ...")
        c_true, rho_true, alpha_true, labels, dx = load_head(N, args.subject)
    roi = brain_roi(labels)
    n_les = int((labels == B.LESION).sum())
    print(f"  dx={dx*1e3:.2f} mm, skull voxels={(labels==B.SKULL).sum()}, "
          f"brain ROI={roi.sum()}, lesion voxels~{n_les}")

    print(f"[2/4] Helmet: {n_elem} receivers, {n_src} sources ...")
    src_positions, sensor_positions = make_helmet(labels, dx, n_elem, n_src)
    solid = (labels == B.SKULL) | np.isin(labels, B.BRAIN) | (labels == B.LESION)
    in_solid = sum(solid[p] for p in zip(*sensor_positions))
    print(f"  receivers in solid tissue: {in_solid} (want 0)")

    c_true_j, rho_true_j, alpha_true_j = map(jnp.asarray, (c_true, rho_true, alpha_true))
    ref_med = build_medium(build_domain((N, N, N), dx), C_MAX, 1000.0, pml_size=8)
    t_end = 1.9 * (N * dx) / 1500.0
    ta = build_time_axis(ref_med, cfl=0.3, t_end=t_end)
    dt = float(ta.dt); nt = int(ta.Nt)
    sig = _build_source_signal(f0, dt, nt)
    print(f"  dt={dt*1e9:.1f} ns, Nt={nt}, t_end={t_end*1e6:.0f} us")

    print("[3/4] Generating observed data (with skull absorption) ...")
    t0 = time.time()
    observed = generate_observed_data(
        c_true_j, rho_true_j, dx, src_positions, sensor_positions, f0,
        pml_size=8, time_axis=ta, source_signal=sig, dt=dt,
        attenuation=alpha_true_j, alpha_power=Y_POWER, verbose=False,
    )
    print(f"  {observed.shape} in {time.time()-t0:.0f}s")

    # Identical start: homogeneous brain, TRUE skull + water (both frozen).
    c_init = c_true.copy(); c_init[roi] = C_BRAIN0
    c_init_j = jnp.asarray(c_init); mask_j = jnp.asarray(roi.astype(np.float32))

    common = dict(
        freq_bands=bands, n_iters_per_band=n_iters, shots_per_iter=shots,
        learning_rate=30.0, c_min=C_MIN, c_max=C_MAX, pml_size=8, cfl=0.3,
        gradient_smooth_sigma=smooth, mask=mask_j, precondition=precond,
        precondition_floor=args.precond_floor, loss_fn="l2", verbose=True,
    )

    def fwi(attenuation, tag):
        print(f"\n=== 3D FWI [{tag}] ===")
        t0 = time.time()
        res = run_fwi(
            observed, c_init_j, rho_true_j, dx, src_positions, sensor_positions,
            sig, dt, t_end,
            config=FWIConfig(attenuation=attenuation, alpha_power=Y_POWER, **common),
            key=jr.PRNGKey(0),
        )
        print(f"  [{tag}] {time.time()-t0:.0f}s, final misfit {res.loss_history[-1]:.4e}")
        return np.asarray(res.velocity), float(res.loss_history[-1])

    print("[4/4] Running FWI (lossless, then absorption-aware) ...")
    c_lossless, mf_lossless = fwi(None, "lossless forward")
    c_aware, mf_aware = fwi(alpha_true_j, "absorption-aware (known alpha)")

    def rmse(c):
        return float(np.sqrt(np.mean((c[roi] - c_true[roi]) ** 2)))
    les = labels == B.LESION
    def les_rmse(c):
        return float(np.sqrt(np.mean((c[les] - c_true[les]) ** 2))) if les.any() else float("nan")

    r0, rl, ra = rmse(c_init), rmse(c_lossless), rmse(c_aware)
    print("\n" + "=" * 62)
    print(f"{'model':<26}{'brain RMSE':>12}{'lesion RMSE':>14}{'misfit':>10}")
    print(f"{'starting model':<26}{r0:>12.2f}{les_rmse(c_init):>14.2f}{'-':>10}")
    print(f"{'lossless FWI':<26}{rl:>12.2f}{les_rmse(c_lossless):>14.2f}{mf_lossless:>10.2e}")
    print(f"{'absorption-aware FWI':<26}{ra:>12.2f}{les_rmse(c_aware):>14.2f}{mf_aware:>10.2e}")
    print("-" * 62)
    if rl > 0:
        print(f"absorption-aware vs lossless: brain RMSE {100*(rl-ra)/rl:+.0f}%, "
              f"misfit {100*(mf_lossless-mf_aware)/mf_lossless:+.0f}%")
    print("=" * 62)

    # Figure: axial slice through the lesion centroid (or brain centroid).
    if les.any():
        zc = int(np.round(np.where(les)[2].mean()))
    else:
        zc = int(np.round(np.where(roi)[2].mean()))
    vmin, vmax, emax = 1500, 1660, 120
    sl = lambda a: a[:, :, zc].T
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.6))
    panels = [
        (ax[0, 0], c_true, f"truth c (z={zc})", vmin, vmax, "viridis"),
        (ax[0, 1], c_lossless, f"lossless FWI (RMSE {rl:.1f})", vmin, vmax, "viridis"),
        (ax[0, 2], c_aware, f"absorption-aware FWI (RMSE {ra:.1f})", vmin, vmax, "viridis"),
        (ax[1, 0], c_init, f"start (RMSE {r0:.1f})", vmin, vmax, "viridis"),
        (ax[1, 1], np.abs(c_lossless - c_true), "lossless |error|", 0, emax, "magma"),
        (ax[1, 2], np.abs(c_aware - c_true), "absorption-aware |error|", 0, emax, "magma"),
    ]
    for a, img, title, lo, hi, cmap in panels:
        im = a.imshow(sl(img), origin="lower", cmap=cmap, vmin=lo, vmax=hi)
        # outline the skull on each panel
        a.contour(sl(labels == B.SKULL).astype(float), levels=[0.5], colors="w", linewidths=0.4)
        a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle("3D absorption-aware FWI: image a real patient brain (+lesion) "
                 "through a known skull", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "absorption_aware_fwi_3d.png", dpi=130)
    print(f"\nFigure -> {out/'absorption_aware_fwi_3d.png'}")
    np.savez(out / "absorption_aware_fwi_3d.npz",
             c_true=c_true, c_init=c_init, c_lossless=c_lossless, c_aware=c_aware,
             labels=labels, roi=roi, dx=dx, zc=zc,
             rmse=(r0, rl, ra), misfit=(mf_lossless, mf_aware))


if __name__ == "__main__":
    main()

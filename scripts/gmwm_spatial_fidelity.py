"""Spatial-fidelity metrics for the 3-D GM/WM reconstructions (risk-register item 7).

AUC says "GM voxels are faster than WM voxels as a population"; it says nothing about
whether the image LOOKS like the anatomy. The real-anatomy recon scored AUC 0.985 and is
visibly speckled; every ladder image is speckle-dominated. This scores the saved
`results/gmwm_velocity_3d/c_rec_<TAG>.npy` arrays offline (no GPU) against the truth
phantom, which `gmwm_velocity_3d.py` builds deterministically from (N, phase):

  * brain-masked SSIM (3-D, Gaussian window) on the 1540-1580 m/s display range
  * speckle index: std of c inside the WM interior (truth is flat 1551.5)
  * ribbon radial profile: mean c vs radius through the cortical shell, and the
    profile CONTRAST (ribbon mean - WM mean) / 17 m/s -- the honest "did the ribbon
    show up as a ring" number, vs the ribbon AUC
  * edge sharpness: 10-90% rise width (mm) of the profile across the inner ribbon edge
  * nuclei/ventricle blob RMSE inside their masks (shape fidelity of resolvable features)

    .venv/bin/python scripts/gmwm_spatial_fidelity.py            # all c_rec_*.npy
    .venv/bin/python scripts/gmwm_spatial_fidelity.py s1B s2B s3B s2B_it40s8
Writes results/gmwm_velocity_3d/spatial_fidelity.json + .png (ribbon profiles).
"""
from __future__ import annotations
import glob, json, os, re, sys
import numpy as np
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = "results/gmwm_velocity_3d"
mm = 1e-3; FOV = 0.180
C_WM, C_GM, C_CSF, C_WATER, C_SKULL = 1551.5, 1568.5, 1500.0, 1500.0, 2800.0
CONTRAST = C_GM - C_WM
R_BRAIN, RIBBON = 55 * mm, 3 * mm
VMIN, VMAX = 1540.0, 1580.0


def phantom(N, phase):
    """Byte-for-byte the construction in gmwm_velocity_3d.py (keep in sync)."""
    dx = FOV / N; cx = N // 2
    gx, gy, gz = np.meshgrid(*[np.arange(N)] * 3, indexing="ij")
    X, Y, Z = (gx - cx) * dx, (gy - cx) * dx, (gz - cx) * dx
    r = np.sqrt(X**2 + Y**2 + Z**2)
    lab = np.zeros((N, N, N), np.int32)
    lab[r <= R_BRAIN] = 2
    lab[r <= R_BRAIN - RIBBON] = 3
    for off in (-16 * mm, 16 * mm):
        lab[np.sqrt((X - off)**2 + (Y + 6 * mm)**2 + Z**2) <= 7 * mm] = 4
    for off in (-8 * mm, 8 * mm):
        lab[np.sqrt((X - off)**2 + (Y - 10 * mm)**2 + (Z / 2.2)**2) <= 5 * mm] = 5
    if phase == "B":
        lab[(r > R_BRAIN) & (r <= R_BRAIN + 1 * mm)] = 0
        lab[(r > R_BRAIN + 1 * mm) & (r <= R_BRAIN + 7 * mm)] = 6
    CMAP = {0: C_WATER, 2: C_GM, 3: C_WM, 4: C_GM, 5: C_CSF, 6: C_SKULL}
    c_true = np.vectorize(CMAP.get)(lab).astype(np.float32)
    return lab, c_true, r, dx


def ssim3d(a, b, mask, sigma=1.5, L=VMAX - VMIN):
    """Masked 3-D SSIM with a Gaussian window (Wang et al. 2004 constants)."""
    a = np.clip(a, VMIN, VMAX) - VMIN; b = np.clip(b, VMIN, VMAX) - VMIN
    g = lambda x: ndi.gaussian_filter(x, sigma, mode="nearest")
    mu_a, mu_b = g(a), g(b)
    s_aa, s_bb, s_ab = g(a * a) - mu_a**2, g(b * b) - mu_b**2, g(a * b) - mu_a * mu_b
    C1, C2 = (0.01 * L)**2, (0.03 * L)**2
    s = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a**2 + mu_b**2 + C1) * (s_aa + s_bb + C2))
    return float(s[mask].mean())


def radial_profile(c, r, lab, dx, r_lo=40 * mm, r_hi=58 * mm):
    """Mean c vs radius over brain voxels, 0.5-voxel bins (excludes nuclei/ventricles)."""
    sel = np.isin(lab, [2, 3])
    bins = np.arange(r_lo, r_hi + dx / 2, dx / 2)
    idx = np.digitize(r[sel], bins)
    prof = np.array([c[sel][idx == i].mean() if np.any(idx == i) else np.nan for i in range(1, len(bins))])
    return (bins[:-1] + bins[1:]) / 2, prof


def rise_width_mm(rc, prof, r_edge, dx):
    """10-90% rise width of the profile across the inner ribbon edge (WM -> GM)."""
    w = (rc > r_edge - 6 * mm) & (rc < r_edge + 2 * mm)
    x, y = rc[w], prof[w]
    ok = np.isfinite(y)
    if ok.sum() < 4: return float("nan")
    x, y = x[ok], y[ok]
    lo, hi = np.nanmin(y), np.nanmax(y)
    if hi - lo < 1e-3: return float("nan")
    y10, y90 = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)
    i10 = np.argmax(y >= y10); i90 = np.argmax(y >= y90)
    return float(abs(x[i90] - x[i10]) * 1e3)


def score(tag):
    c = np.load(f"{D}/c_rec_{tag}.npy").astype(np.float32)
    N = c.shape[0]; phase = re.match(r"s\d([AB])", tag).group(1)
    lab, c_true, r, dx = phantom(N, phase)
    assert c_true.shape == c.shape, (tag, c.shape, c_true.shape)
    brain = np.isin(lab, [2, 3, 4, 5])
    wm_core = (lab == 3) & (r < R_BRAIN - RIBBON - 4 * mm)          # away from the ribbon edge
    rib = ndi.binary_erosion(lab == 2, iterations=1) if dx < 1.4e-3 else (lab == 2)
    rc, prof = radial_profile(c, r, lab, dx)
    rib_mean = float(np.nanmean(prof[(rc > R_BRAIN - RIBBON + 0.5 * dx) & (rc < R_BRAIN - 0.5 * dx)]))
    wm_mean = float(np.nanmean(prof[(rc > R_BRAIN - RIBBON - 8 * mm) & (rc < R_BRAIN - RIBBON - 2 * mm)]))
    # inner vs outer half of the ribbon: transcranially the OUTER 1.5 mm (under the 1 mm CSF gap
    # + frozen skull) is sign-inverted at stages 1-2 while the inner half matches no-skull —
    # the whole-ribbon AUC hides a frozen-interface edge artefact, not a resolution limit
    wm_ref = c[(lab == 3) & ~ndi.binary_dilation(lab != 3, iterations=3)]
    inner = (lab == 2) & (r <= R_BRAIN - 1.5 * mm); outer = (lab == 2) & (r > R_BRAIN - 1.5 * mm)
    def _auc(pos, neg):
        v = np.concatenate([pos, neg]); y = np.r_[np.ones(pos.size), np.zeros(neg.size)]
        rk = np.argsort(np.argsort(v)) + 1
        return float((rk[y == 1].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
    out = dict(
        tag=tag, N=N, dx_mm=dx * 1e3, phase=phase,
        ribbon_inner_auc=_auc(c[inner], wm_ref),
        ribbon_inner_pct=100 * float(np.median(c[inner]) - np.median(wm_ref)) / CONTRAST,
        ribbon_outer_auc=_auc(c[outer], wm_ref),
        ribbon_outer_pct=100 * float(np.median(c[outer]) - np.median(wm_ref)) / CONTRAST,
        ssim_brain=ssim3d(c, c_true, brain),
        speckle_wm_std=float(c[wm_core].std()),                       # truth: 0
        rmse_brain=float(np.sqrt(np.mean((c[brain] - c_true[brain])**2))),
        ribbon_profile_contrast_pct=100 * (rib_mean - wm_mean) / CONTRAST,
        ribbon_rise_width_mm=rise_width_mm(rc, prof, R_BRAIN - RIBBON, dx),
        nuclei_rmse=float(np.sqrt(np.mean((c[lab == 4] - C_GM)**2))),
        ventricle_rmse=float(np.sqrt(np.mean((c[lab == 5] - C_CSF)**2))),
        n_vox_brain=int(brain.sum()),
    )
    s = json.load(open(f"{D}/summary_{tag}.json")) if os.path.exists(f"{D}/summary_{tag}.json") else {}
    out["ribbon_auc"] = s.get("ribbon", {}).get("auc"); out["nuclei_auc"] = s.get("nuclei", {}).get("auc")
    return out, (rc, prof)


if __name__ == "__main__":
    tags = sys.argv[1:] or sorted(re.sub(r".*/c_rec_(.*)\.npy", r"\1", f) for f in glob.glob(f"{D}/c_rec_*.npy"))
    rows, profiles = [], {}
    print(f"{'tag':16s} {'N':>4s} {'SSIM':>6s} {'speckle':>8s} {'RMSE':>6s} {'rib AUC':>7s} "
          f"{'inner AUC/%':>12s} {'outer AUC/%':>12s} {'rise mm':>7s} {'nuc RMSE':>8s} {'vent RMSE':>9s}")
    for t in tags:
        o, pr = score(t); rows.append(o); profiles[t] = pr
        print(f"{t:16s} {o['N']:4d} {o['ssim_brain']:6.3f} {o['speckle_wm_std']:8.2f} {o['rmse_brain']:6.2f} "
              f"{str(round(o['ribbon_auc'], 3)) if o['ribbon_auc'] else '  -':>7s} "
              f"{o['ribbon_inner_auc']:6.3f}/{o['ribbon_inner_pct']:+4.0f}% {o['ribbon_outer_auc']:6.3f}/{o['ribbon_outer_pct']:+4.0f}% "
              f"{o['ribbon_rise_width_mm']:7.2f} {o['nuclei_rmse']:8.2f} {o['ventricle_rmse']:9.2f}")
    json.dump(rows, open(f"{D}/spatial_fidelity.json", "w"), indent=2)

    # ribbon radial profiles for the ladder (whatever subset is present)
    # s1A (no skull) is the control that shows the ring CAN be recovered at stage-1 resolution
    ladder = [t for t in ("s1A", "s1B", "s2B", "s2B_it40s8", "s3B", "s1B_ppw49", "s2B_ppw37") if t in profiles]
    if ladder:
        fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor="white")
        for t in ladder:
            rc, prof = profiles[t]; ax.plot(rc * 1e3, prof, lw=1.6, label=t)
        ax.axvspan((R_BRAIN - RIBBON) * 1e3, R_BRAIN * 1e3, color="orange", alpha=0.15, label="ribbon (truth)")
        ax.axhline(C_WM, color="k", lw=0.8, ls=":"); ax.axhline(C_GM, color="k", lw=0.8, ls="--")
        ax.set_xlabel("radius (mm)"); ax.set_ylabel("mean c (m/s)"); ax.set_ylim(VMIN, VMAX)
        ax.set_title("Cortical-ribbon radial profile: does the ring show up?"); ax.legend(fontsize=8, ncol=2)
        fig.savefig(f"{D}/spatial_fidelity_ribbon_profiles.png", dpi=140, bbox_inches="tight", facecolor="white")
        print(f"saved {D}/spatial_fidelity_ribbon_profiles.png")

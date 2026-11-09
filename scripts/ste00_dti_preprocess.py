"""Fit DTI on the STE00 ex-vivo human brain hemisphere (real post-mortem tissue)
and export the anisotropy substrate: FA, V1 fibre direction, and an FA histogram.

This is the REAL-TISSUE half of the WM attenuation-anisotropy validation: it
quantifies how anisotropic and coherently-oriented the actual white matter is
(the necessary substrate for acoustic-attenuation anisotropy), and shows it forms
a population distinct from near-isotropic grey matter / CSF -- i.e. the GM/WM
discriminator physically exists in real tissue.

Run with the sbi4dwi venv (has dipy):
  /home/mhough/dev/sbi4dwi/.venv/bin/python scripts/ste00_dti_preprocess.py
Outputs -> results/wm_anisotropy_validation/ste00_*.npy + stats json.
"""
import os, json, numpy as np, nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.io.gradients import read_bvals_bvecs

BASE = "/data/downloads_backup/STE/STE00_ExVivo/STE"
OUT = "results/wm_anisotropy_validation"
os.makedirs(OUT, exist_ok=True)

print("loading STE00 ex-vivo DWI...", flush=True)
img = nib.load(f"{BASE}/STE_degibbs_eddy.nii.gz")
data = img.get_fdata(dtype=np.float32)
mask = nib.load(f"{BASE}/STE_mask.nii.gz").get_fdata() > 0
bvals, bvecs = read_bvals_bvecs(f"{BASE}/bvals.txt", f"{BASE}/bvecs.txt")
gtab = gradient_table(bvals, bvecs=bvecs)
print(f"  data {data.shape}, mask voxels {int(mask.sum())}, shells {sorted(set(bvals.round(-2)))}", flush=True)

print("fitting DTI (weighted least squares)...", flush=True)
ten = TensorModel(gtab, fit_method="WLS")
fit = ten.fit(data, mask=mask)
FA = np.nan_to_num(fit.fa).astype(np.float32)
V1 = np.nan_to_num(fit.evecs[..., 0]).astype(np.float32)  # principal eigenvector
MD = np.nan_to_num(fit.md).astype(np.float32)

np.save(f"{OUT}/ste00_fa.npy", FA)
np.save(f"{OUT}/ste00_v1.npy", V1)
np.save(f"{OUT}/ste00_mask.npy", mask)

fa_in = FA[mask]
fa_in = fa_in[(fa_in > 0) & (fa_in < 1)]
# WM population ~ high FA, GM/CSF ~ low FA. Simple bimodal split at FA=0.25.
wm = fa_in[fa_in >= 0.25]; gm = fa_in[fa_in < 0.25]
stats = {
    "n_brain_voxels": int(mask.sum()),
    "fa_median_all": float(np.median(fa_in)),
    "fa_p90_all": float(np.percentile(fa_in, 90)),
    "wm_frac_FA>=0.25": float(wm.size / fa_in.size),
    "wm_fa_median": float(np.median(wm)) if wm.size else None,
    "wm_fa_p90": float(np.percentile(wm, 90)) if wm.size else None,
    "gm_fa_median": float(np.median(gm)) if gm.size else None,
    "note": "ex-vivo fixation lowers FA vs in-vivo; orientation preserved",
}
# orientation coherence: |V1 . mean(V1)| over high-FA neighbourhood proxy (global)
hi = mask & (FA >= 0.35)
if hi.sum() > 100:
    v = V1[hi]
    v = v * np.sign(v[:, 2:3] + 1e-9)  # resolve sign ambiguity by z
    coh = np.linalg.norm(v.mean(0))    # 1 = perfectly aligned, 0 = random
    stats["wm_global_orientation_coherence"] = float(coh)

with open(f"{OUT}/ste00_dti_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
np.save(f"{OUT}/ste00_fa_hist.npy",
        np.histogram(fa_in, bins=60, range=(0, 1))[0])
print("STATS:", json.dumps(stats, indent=2), flush=True)
print(f"saved -> {OUT}/", flush=True)

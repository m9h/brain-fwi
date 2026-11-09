"""Aggregate Run 3 into the skull-pose tolerance curve for GM/WM."""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = "results/gmwm_unknown_skull"; OUT = "results/run3"; os.makedirs(OUT, exist_ok=True)
rows = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/summary_pose*.json"))]
if not rows:
    raise SystemExit("no Run 3 summaries yet")
rows.sort(key=lambda r: r["pose_vox"])

print(f"{'tag':10s} {'vox':>5s} {'deg':>5s} {'skull mm%':>9s} {'resid%':>7s} "
      f"{'nuclei AUC':>11s} {'nuclei %':>9s} {'ribbon AUC':>11s} {'csf %':>7s}")
for r in rows:
    print(f"{r['tag']:10s} {r['pose_vox']:5.2f} {r['pose_deg']:5.2f} "
          f"{r['skull_mismatch_pct']:8.1f}% {r['start_residual_pct']:6.1f}% "
          f"{r['nuclei']['auc']:11.3f} {r['nuclei']['pct']:8.0f}% "
          f"{r['ribbon']['auc']:11.3f} {r['csf']['pct']:6.0f}%")

x = [max(r["pose_vox"], 0.03) for r in rows]      # 0 plotted at 0.03 for log axis
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4), facecolor="white")
ax[0].plot(x, [r["nuclei"]["auc"] for r in rows], "o-", color="#c33",
           label="deep-gray nuclei (GM/WM contrast)")
ax[0].plot(x, [r["ribbon"]["auc"] for r in rows], "s--", color="#2a6fdb",
           label="cortical ribbon 3mm")
ax[0].plot(x, [r["csf"]["auc"] for r in rows], "^:", color="#2a9d3a", label="ventricle (control)")
ax[0].axhline(0.9, color="k", ls=":", lw=1)
ax[0].axvline(0.07, color="#e67", lw=2, alpha=.7)
ax[0].annotate("MOFI measured\naccuracy (0.07 vox)", xy=(0.07, 0.55), fontsize=8, color="#c33",
               ha="left")
ax[0].set_xscale("log"); ax[0].set_xlabel("skull pose error (voxels; rotation scales with it)")
ax[0].set_ylabel("AUC"); ax[0].set_title("Skull-pose tolerance for GM/WM")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

ax[1].plot([r["skull_mismatch_pct"] for r in rows], [r["nuclei"]["pct"] for r in rows],
           "o-", color="#c33")
ax[1].set_xlabel("skull-mass mismatch (%)"); ax[1].set_ylabel("GM/WM contrast recovered (%)")
ax[1].set_title("Contrast vs skull-model error"); ax[1].grid(alpha=.3)
fig.suptitle("Run 3: does GM/WM survive an UNKNOWN skull? (3-D, 300 kHz, ellipsoidal head, soft warp)",
             y=1.03, fontsize=12)
fig.savefig(f"{OUT}/run3_pose_tolerance.png", dpi=140, bbox_inches="tight", facecolor="white")
json.dump(rows, open(f"{OUT}/run3_summary.json", "w"), indent=2)
print(f"\nsaved {OUT}/run3_pose_tolerance.png")

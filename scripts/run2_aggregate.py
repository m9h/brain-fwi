"""Aggregate the Run 2 sweep into the array-design curve (AUC vs element count)
and the noise-robustness table."""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = "results/gmwm_velocity_3d"; OUT = "results/run2"; os.makedirs(OUT, exist_ok=True)
rows = []
for f in sorted(glob.glob(f"{D}/summary_s1B_cov*.json")):
    j = json.load(open(f))
    j["tag"] = os.path.basename(f).replace("summary_s1B_", "").replace(".json", "")
    rows.append(j)
if not rows:
    raise SystemExit("no Run 2 summaries yet")

clean = sorted([r for r in rows if r.get("noise_db") is None], key=lambda r: r["n_rec"])
noisy = [r for r in rows if r.get("noise_db") is not None]

print(f"{'config':16s} {'elem':>5s} {'noise':>6s} {'nuclei AUC':>11s} {'nuclei %':>9s} "
      f"{'ribbon AUC':>11s} {'csf %':>7s}")
for r in clean + sorted(noisy, key=lambda x: (x["n_rec"], -x["noise_db"])):
    nz = "clean" if r.get("noise_db") is None else f"{r['noise_db']:.0f}dB"
    print(f"{r['tag']:16s} {r['n_rec']:5d} {nz:>6s} "
          f"{r['nuclei']['auc']:11.3f} {r['nuclei']['pct']:8.0f}% "
          f"{r['ribbon']['auc']:11.3f} {r['csf']['pct']:6.0f}%")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), facecolor="white")
if clean:
    e = [r["n_rec"] for r in clean]
    ax[0].plot(e, [r["nuclei"]["auc"] for r in clean], "o-", color="#c33",
               label="deep-gray nuclei (GM/WM contrast)")
    ax[0].plot(e, [r["ribbon"]["auc"] for r in clean], "s--", color="#2a6fdb",
               label="cortical ribbon 3mm (resolution-limited)")
    ax[0].axhline(0.9, color="k", ls=":", lw=1, label="AUC 0.9 (resolved)")
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(e); ax[0].set_xticklabels(e)
    ax[0].set_xlabel("array elements (receivers)"); ax[0].set_ylabel("GM/WM AUC")
    ax[0].set_title("Array-design curve: how few elements?"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
if noisy:
    for nrec, mk in [(32, "o-"), (128, "s-")]:
        grp = sorted([r for r in rows if r["n_rec"] == nrec],
                     key=lambda r: (r["noise_db"] if r.get("noise_db") else 99))
        if len(grp) < 2: continue
        x = [(r["noise_db"] if r.get("noise_db") else 40) for r in grp]
        ax[1].plot(x, [r["nuclei"]["auc"] for r in grp], mk, label=f"{nrec} elements")
    ax[1].axhline(0.9, color="k", ls=":", lw=1)
    ax[1].set_xlabel("measurement SNR (dB; 40 = noiseless)")
    ax[1].set_ylabel("GM/WM AUC (nuclei)")
    ax[1].set_title("Noise robustness"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.suptitle("Run 2: transcranial 3-D GM/WM — coverage knee and noise robustness "
             "(stage 1, 300 kHz)", y=1.03, fontsize=12)
fig.savefig(f"{OUT}/run2_coverage_noise.png", dpi=140, bbox_inches="tight", facecolor="white")
json.dump(rows, open(f"{OUT}/run2_summary.json", "w"), indent=2)
print(f"\nsaved {OUT}/run2_coverage_noise.png")

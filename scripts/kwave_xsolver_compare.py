"""Forward-agreement check: j-Wave vs k-Wave obs on the SAME medium/geometry.

k-Wave returns sensor traces in its binary-mask flatten order (C vs F ambiguous),
so we test both reorderings (and identity) against j-Wave, plus a greedy best-match
to tell whether ordering is the *whole* story. High zero-lag correlation under the
right ordering validates the j-Wave forward against the field-standard reference.
"""
import json, numpy as np
from scipy.signal import resample
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

geom = json.load(open("/tmp/kwave_geom.json")); S = geom["S"]
pg = [np.array(geom["pg"][d], dtype=int) for d in range(3)]
jobs = np.load("/tmp/jwave_obs.npy").astype(np.float32); dt_j = float(np.load("/tmp/jwave_meta.npz")["dt"])
kraw = np.load("/tmp/kwave_obs.npy").astype(np.float32); dt_k = float(np.load("/tmp/kwave_meta.npz")["dt"])
nsh, Ntj, nrec = jobs.shape
print(f"j-Wave {jobs.shape} @ {dt_j*1e9:.1f}ns   k-Wave(raw) {kraw.shape} @ {dt_k*1e9:.1f}ns", flush=True)

T = min(Ntj * dt_j, kraw.shape[1] * dt_k); nj = int(T / dt_j)
J = resample(jobs[:, :int(T/dt_j), :], nj, axis=1)
Kc = resample(kraw[:, :int(T/dt_k), :], nj, axis=1)            # common time axis, RAW sensor order
def nrm(x): return (x - x.mean()) / (x.std() + 1e-12)

lin_C = pg[0] * S * S + pg[1] * S + pg[2]
lin_F = pg[0] + pg[1] * S + pg[2] * S * S
def reorder(lin): return np.argsort(np.argsort(lin))           # raw -> pg order

def zero_lag(Kperm):
    cs = []
    for s in range(nsh):
        for r in range(nrec):
            a, b = J[s, :, r], Kperm[s, :, r]
            if a.std() < 1e-6 or b.std() < 1e-6: continue
            cs.append(float(np.mean(nrm(a) * nrm(b))))
    return np.array(cs)

for name, lin in [("identity", None), ("C-order", lin_C), ("F-order", lin_F)]:
    Kp = Kc if lin is None else Kc[:, :, reorder(lin)]
    c = zero_lag(Kp)
    print(f"  {name:9s}: zero-lag corr mean {c.mean():+.3f}  median {np.median(c):+.3f}", flush=True)

# greedy best-match (shot 0, best zero-lag pairing) -> is ordering the whole story?
s = 0; Jn = np.stack([nrm(J[s, :, r]) for r in range(nrec)]); Kn = np.stack([nrm(Kc[s, :, r]) for r in range(nrec)])
C = Jn @ Kn.T / nj                                            # (nrec_j, nrec_k) zero-lag corr matrix
best = C.max(axis=1)
print(f"  greedy best-match (shot 0): per-jwave-sensor max corr  mean {best.mean():.3f}  median {np.median(best):.3f}", flush=True)
print(f"  (if greedy-match is high but C/F low -> a different permutation; if greedy low -> source/convention, not ordering)", flush=True)

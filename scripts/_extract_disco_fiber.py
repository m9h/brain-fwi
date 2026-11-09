#!/usr/bin/env python
"""Extract a 2D fibre-direction + FA slice from the DiSCo diffusion phantom, for
the acoustic-tractography demo (scripts/anisotropy_disco_demo.py).

RUN IN THE sbi4dwi VENV (it has the DiSCo loader + gradient table):
    /home/mhough/dev/sbi4dwi/.venv/bin/python scripts/_extract_disco_fiber.py

DiSCo (numerical phantom with known curving/crossing strands) lives at
~/.dipy/disco/disco_1/. We DTI-fit it (plain numpy), take the principal
eigenvector V1 as the fibre direction and standard FA, pick the axial slice with
the most in-plane fibre structure, and save phi/fa/inplane/mask to the brain-fwi
scratchpad npz the demo loads.
"""
import sys
import numpy as np
sys.path.insert(0, "/home/mhough/dev/sbi4dwi")
from dmipy_jax.validation.force_disco import load_disco_subject

d = load_disco_subject(subject=1, res="highRes", single_shell_b=1900.0)
data = np.asarray(d["data"], float); mask = np.asarray(d["mask"]) > 0.5
bvals = np.asarray(d["gtab"].bvals); bvecs = np.asarray(d["gtab"].bvecs)

b0 = data[..., bvals < 50].mean(-1)
sel = bvals > 50; g = bvecs[sel]; b = bvals[sel]; S = data[..., sel]
B = -b[:, None] * np.stack([g[:, 0]**2, g[:, 1]**2, g[:, 2]**2,
                            2*g[:, 0]*g[:, 1], 2*g[:, 0]*g[:, 2], 2*g[:, 1]*g[:, 2]], 1)
X, Y, Zd = b0.shape
Y_ = np.log(np.clip(S / (b0[..., None] + 1e-6), 1e-4, None)).reshape(-1, sel.sum()).T
D, *_ = np.linalg.lstsq(B, Y_, rcond=None)
nv = D.shape[1]; T = np.zeros((nv, 3, 3))
T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = D[0], D[1], D[2]
T[:, 0, 1] = T[:, 1, 0] = D[3]; T[:, 0, 2] = T[:, 2, 0] = D[4]; T[:, 1, 2] = T[:, 2, 1] = D[5]
w, v = np.linalg.eigh(T)
V1 = v[:, :, 2].reshape(X, Y, Zd, 3)
md = w.mean(1)
fa = np.sqrt(1.5 * ((w - md[:, None])**2).sum(1) / ((w**2).sum(1) + 1e-12)).reshape(X, Y, Zd)
fa = np.nan_to_num(np.clip(fa, 0, 1))
inpl = V1[..., 0]**2 + V1[..., 1]**2
z = int(np.argmax([np.nansum((fa * inpl * mask)[:, :, k]) for k in range(Zd)]))
out = ("/tmp/claude-1000/-home-mhough-dev-brain-fwi/"
       "f4788dc1-b4e6-4d53-8ed1-756b0f54188f/scratchpad/disco_fiber.npz")
np.savez(out, phi=np.arctan2(V1[:, :, z, 1], V1[:, :, z, 0]).astype(np.float32),
         fa=fa[:, :, z].astype(np.float32), inplane=inpl[:, :, z].astype(np.float32),
         mask=mask[:, :, z].astype(np.float32), z=z)
print(f"z={z} fa_max={fa[:, :, z].max():.2f} saved {out}")

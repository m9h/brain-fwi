#!/usr/bin/env python
"""Path forward for GM/WM: multi-angle anisotropic attenuation separates white
from gray matter even when sound speed and bulk attenuation are IDENTICAL.
Produces results/absorption_aware_fwi_3d/anisotropy_gmwm.png."""
import os
import numpy as np, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from brain_fwi.inversion.anisotropic_atten import (
    make_ring_rays, forward_ray_decay, invert_anisotropic)

N, dx = 40, 1e-3
yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
r = np.sqrt((xx - N/2)**2 + (yy - N/2)**2)
brain = r <= N*0.4; wm = r <= N*0.22; gm = brain & ~wm
a_iso = np.where(brain, 0.6, 0.0).astype(np.float32)     # identical for GM & WM
a_aniso = np.where(wm, 0.5, 0.0).astype(np.float32)      # only WM is anisotropic
phi = np.full((N, N), 0.6, np.float32)                   # fibre direction (DTI)

rays = make_ring_rays(N, dx, n_trans=64, radius_frac=0.46)
obs = forward_ray_decay(jnp.asarray(a_iso), jnp.asarray(a_aniso), jnp.asarray(phi), rays)
rec = invert_anisotropic(obs, rays, jnp.asarray(phi), (N, N),
                         mask=jnp.asarray(brain.astype(np.float32)),
                         a_iso=jnp.asarray(a_iso), n_iters=2000)
aa = np.asarray(rec.a_aniso)
print(f"anisotropy: WM={aa[wm].mean():.3f} GM={aa[gm].mean():.3f} "
      f"ratio={aa[wm].mean()/max(aa[gm].mean(),1e-6):.1f} (true WM=0.5 GM=0)", flush=True)

fig, ax = plt.subplots(1, 4, figsize=(15, 4.2), facecolor="white")
for a, img, ttl, vmax, cm in [
    (ax[0], a_iso, "sound speed & bulk α\n(GM = WM — cannot separate)", 0.8, "viridis"),
    (ax[1], a_aniso, "true anisotropy\n(only WM)", 0.6, "magma"),
    (ax[2], aa, "recovered anisotropy\n(multi-angle tomography)", 0.6, "magma"),
]:
    im = a.imshow(img, origin="lower", cmap=cm, vmin=0, vmax=vmax)
    a.contour((wm).astype(float), [0.5], colors="cyan", linewidths=1.0)
    a.set_title(ttl, fontsize=11); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=a, fraction=0.046)
ax[3].axis("off")
ax[3].text(0.0, 0.5,
           "GM/WM share sound speed\n(degenerate) and bulk α\n(identical here).\n\n"
           "Anisotropy is the discriminator\nwith ANGULAR leverage:\n"
           "WM fibres → direction-dependent α.\n\n"
           f"Recovered WM/GM anisotropy\nratio = {aa[wm].mean()/max(aa[gm].mean(),1e-6):.0f}×\n"
           "(given a bulk-α estimate).",
           fontsize=11, va="center")
fig.suptitle("The path forward for GM/WM: multi-angle anisotropic attenuation tomography",
             fontsize=14, y=1.02)
os.makedirs("results/absorption_aware_fwi_3d", exist_ok=True)
out = "results/absorption_aware_fwi_3d/anisotropy_gmwm.png"
fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
print("saved", out, flush=True)

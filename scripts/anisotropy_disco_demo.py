#!/usr/bin/env python
"""Recover REAL curved fibre-tract orientation from ultrasound, using the DiSCo
diffusion phantom's fibre geometry as ground truth.

DTI (from sbi4dwi) gives a per-voxel fibre direction phi(x) and FA on the DiSCo
numerical phantom (known curving/crossing strands). We paint an anisotropic-
attenuation phantom (alpha_aniso ~ FA where fibres are in-plane, direction = phi)
and blindly recover both magnitude and orientation with multi-angle ray
tomography -- no fibre direction supplied. Tests whether acoustic tractography
survives spatially-varying (curved) fibres, not just a uniform direction.
"""
import os
import numpy as np, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from brain_fwi.inversion.anisotropic_atten import (
    make_ring_rays, forward_ray_decay, invert_orientation)

Z = np.load("/tmp/claude-1000/-home-mhough-dev-brain-fwi/f4788dc1-b4e6-4d53-8ed1-756b0f54188f/scratchpad/disco_fiber.npz")
phi_t = Z["phi"].astype(np.float32); fa = Z["fa"]; inpl = Z["inplane"]; mask = Z["mask"]
N = phi_t.shape[0]; dx = 1e-3
# anisotropic-attenuation phantom from the diffusion geometry
a_iso = (0.6 * mask).astype(np.float32)                       # uniform bulk in support
a_aniso_t = (0.9 * fa * inpl * mask).astype(np.float32)       # anisotropy where fibres are (in-plane)
fibre = (a_aniso_t > 0.15)                                    # voxels with real anisotropy
print(f"DiSCo slice {N}x{N}: support={int(mask.sum())} fibre voxels={int(fibre.sum())} "
      f"a_aniso max={a_aniso_t.max():.2f}", flush=True)

rays = make_ring_rays(N, dx, n_trans=72, radius_frac=0.47)
obs = forward_ray_decay(jnp.asarray(a_iso), jnp.asarray(a_aniso_t), jnp.asarray(phi_t), rays)
rec = invert_orientation(obs, rays, (N, N), mask=jnp.asarray(mask), n_iters=4000)
aa = np.asarray(rec.a_aniso); ph = np.asarray(rec.phi)

# fibre-direction error (mod pi), weighted where anisotropy is real
err = np.abs((ph[fibre] - phi_t[fibre] + np.pi/2) % np.pi - np.pi/2)
mag_corr = np.corrcoef(aa[mask>0.5], a_aniso_t[mask>0.5])[0, 1]
print(f"fibre-direction error: median={np.degrees(np.median(err)):.1f} deg, "
      f"mean={np.degrees(np.mean(err)):.1f} deg", flush=True)
print(f"anisotropy magnitude correlation (true vs recovered) = {mag_corr:.3f}", flush=True)

# figure: HSV = direction (mod pi), value = anisotropy magnitude
def dirmap(phi, mag):
    import matplotlib.colors as mc
    h = (phi % np.pi) / np.pi
    v = np.clip(mag / max(mag.max(), 1e-6), 0, 1)
    return mc.hsv_to_rgb(np.stack([h, np.ones_like(h), v], -1))
fig, ax = plt.subplots(1, 4, figsize=(16, 4.4), facecolor="white")
ax[0].imshow(a_aniso_t.T, origin="lower", cmap="magma"); ax[0].set_title("true anisotropy (∝ FA)")
ax[1].imshow(dirmap(phi_t, a_aniso_t).transpose(1,0,2), origin="lower")
ax[1].set_title("true fibre direction\n(hue=angle, brightness=anisotropy)")
ax[2].imshow(np.asarray(aa).T, origin="lower", cmap="magma"); ax[2].set_title("recovered anisotropy")
ax[3].imshow(dirmap(ph, aa).transpose(1,0,2), origin="lower")
ax[3].set_title("recovered fibre direction\n(ultrasound only, no DTI)")
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"Acoustic tractography on a real diffusion phantom (DiSCo): fibre direction "
             f"recovered to {np.degrees(np.median(err)):.0f}° median, magnitude corr {mag_corr:.2f}",
             fontsize=13, y=1.02)
os.makedirs("results/absorption_aware_fwi_3d", exist_ok=True)
out = "results/absorption_aware_fwi_3d/anisotropy_disco.png"
fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white"); print("saved", out, flush=True)

"""Validate the 3D score prior: does it denoise a held-out real cerebrum volume?

Cheap CPU check (no FWI). Noise a held-out subject's cerebrum volume at a few
VPSDE levels, apply one Tweedie denoise step
``x0_hat = (x_t + sigma^2 * score) / alpha``, and measure ROI-RMSE(x0_hat) vs
ROI-RMSE(x_t). A useful prior shrinks the error at every level — that local
score (not the unconditional sample quality) is what DPS-guided FWI consumes.

    JAX_PLATFORMS=cpu python scripts/validate_3d_prior.py
"""
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

jax.config.update("jax_enable_x64", False)
from brain_fwi.inference.score_unet3d import UNet3DScore
from brain_fwi.inference.diffusion import VPSDE
from brain_fwi.phantoms import birnbaum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/brain_score_3d.eqx")
    ap.add_argument("--norm", default="/tmp/brain_score_3d_norm.npz")
    ap.add_argument("--subject", default="subj2", help="held-out subject id")
    ap.add_argument("--S", type=int, default=48)
    args = ap.parse_args()

    import nibabel as nib
    f = [x for x in birnbaum.label_files() if f"{args.subject}_label" in x][0]
    lab = np.asarray(nib.load(f).dataobj).astype(np.int16)
    crop = birnbaum.cerebrum_volume_crop(lab, S=args.S)
    roi = birnbaum.roi_mask(crop)
    x_true = jnp.asarray(birnbaum.prior_volume(crop))

    nz = np.load(args.norm); SM, SS = float(nz["mean"]), float(nz["std"])
    model = eqx.tree_deserialise_leaves(args.model, UNet3DScore(args.S, C=16, key=jr.PRNGKey(0)))
    sde = VPSDE()
    xn = ((x_true - SM) / SS).reshape(-1)
    m = np.asarray(roi); xtrue_v = np.asarray(x_true)

    print(f"held-out {args.subject}: lesion {int((crop == birnbaum.LESION).sum())}, "
          f"cerebrum {int(roi.sum())} vox")
    print(f"{'t':>5} {'alpha':>7} {'sigma':>7} {'RMSE(noisy)':>12} {'RMSE(denoised)':>15} {'gain':>6}")
    key = jr.PRNGKey(0)
    for t in [0.05, 0.1, 0.2, 0.3, 0.5]:
        key, kn = jr.split(key)
        a, sig = sde.alpha(t), sde.sigma(t)
        xt = a * xn + sig * jr.normal(kn, xn.shape)
        x0 = (xt + sig ** 2 * model(xt, t)) / a
        devel = lambda z: np.asarray(z.reshape(args.S, args.S, args.S)) * SS + SM
        rn = np.sqrt(np.mean((devel(xt)[m] - xtrue_v[m]) ** 2))
        rd = np.sqrt(np.mean((devel(x0)[m] - xtrue_v[m]) ** 2))
        print(f"{t:>5.2f} {float(a):>7.3f} {float(sig):>7.3f} {rn:>12.1f} {rd:>15.1f} {100*(1-rd/rn):>+5.0f}%")


if __name__ == "__main__":
    main()

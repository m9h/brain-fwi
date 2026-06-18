"""Diffusion-prior-guided FWI demo (2D, real Birnbaum slice).

Skull held as prior; preconditioned, ROI-masked FWI reconstructs the
intracranial interior, guided by a learned U-Net diffusion prior (DPS-style:
data-misfit gradient + score). Compares prior-off vs prior-on.

Prereqs (see docs/design/diffusion_prior_fwi.md):
  1. build dataset:  brain_fwi.phantoms.birnbaum.build_slice_dataset(...)
  2. train prior:    scripts/modal_train_unet_score.py  ->  brain_score.eqx + norm
Then:  python examples/dps_fwi_demo.py

NOTE (characterised limitation): in 2D the low-contrast lesion is not cleanly
recovered — the data under-constrains it and 2D entangles brain with face.
The prior reliably *regularises* (kills acquisition artifacts). 3D is the fix.
"""
import argparse
import jax, jax.numpy as jnp, jax.random as jr, numpy as np, equinox as eqx
import nibabel as nib
from brain_fwi.inference.score_unet import UNetScore
from brain_fwi.inference.diffusion import VPSDE
from brain_fwi.phantoms.birnbaum import (
    label_files, cerebrum_crop, roi_mask, to_velocity, LESION, BRAIN)
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis, simulate_shot_sensors, generate_observed_data)
from brain_fwi.transducers import ring_array_2d, transducer_positions_to_grid
from brain_fwi.utils.wavelets import ricker_wavelet
from brain_fwi.inversion.losses import l2_loss
from brain_fwi.inversion.fwi import _smooth_gradient, _bandpass_signal


def main(model_path, norm_path, subject="subj2", S=64, dx=2e-3, pml=8):
    f = [x for x in label_files() if subject in x][0]
    lab3 = np.asarray(nib.load(f).dataobj).astype(np.int16)
    z = int(np.argmax([(lab3[zz] == LESION).sum() for zz in range(lab3.shape[0])]))
    crop = cerebrum_crop(lab3[z], S=S)
    c_true = jnp.asarray(to_velocity(crop, with_skull=True))
    skull = jnp.asarray(crop == 5); interior = jnp.asarray(roi_mask(crop)); imask = interior.astype(jnp.float32)
    rho = jnp.full((S, S), 1000., jnp.float32)

    pos = ring_array_2d(n_elements=32, center=(S//2*dx, S//2*dx), semi_major=(S//2-1)*dx, semi_minor=(S//2-1)*dx, standoff=0.)
    pg = transducer_positions_to_grid(pos, dx, (S, S)); src = [(int(pg[0][i]), int(pg[1][i])) for i in range(32)]
    t_end = 4e-5
    ta = build_time_axis(build_medium(build_domain((S, S), dx), 2800., 1000., pml_size=pml), cfl=0.3, t_end=t_end)
    dt = float(ta.dt); sig = ricker_wavelet(f0=130e3, dt=dt, n_samples=int(t_end/dt))
    obs = generate_observed_data(sound_speed=c_true, density=rho, dx=dx, src_positions_grid=src,
        sensor_positions_grid=pg, freq=130e3, pml_size=pml, time_axis=ta, source_signal=sig, dt=dt, verbose=False)

    model = eqx.tree_deserialise_leaves(model_path, UNetScore(S, S, C=32, key=jr.PRNGKey(0)))
    nz = np.load(norm_path); SM, SS = float(nz["mean"]), float(nz["std"])

    def shot_loss(x, sp, ot, bp):
        med = build_medium(build_domain((S, S), dx), x, rho, pml_size=pml)
        pred = simulate_shot_sensors(med, ta, sp, pg, bp, dt); mt = min(pred.shape[0], ot.shape[0])
        return l2_loss(pred[:mt], ot[:mt])

    def fwi(prior_lr, t_eps=0.1, bands=((40e3, 90e3), (90e3, 170e3)), iters=25, shots=8, lr=20.):
        x = jnp.where(skull, 2800., 1500.).astype(jnp.float32); key = jr.PRNGKey(0)
        for fmin, fmax in bands:
            bp = _bandpass_signal(sig, dt, fmin, fmax)
            bobs = jax.vmap(lambda d: jax.vmap(lambda c: _bandpass_signal(c, dt, fmin, fmax))(d.T).T)(obs)
            for _ in range(iters):
                key, k = jr.split(key); idx = np.array(jr.choice(k, 32, (shots,), replace=False))
                g = jnp.zeros_like(x); gsq = jnp.zeros_like(x)
                for si in idx:
                    _, gs = jax.value_and_grad(shot_loss)(x, src[int(si)], bobs[int(si)], bp); g += gs; gsq += gs**2
                g /= shots; g = g / (jnp.sqrt(gsq/shots) + 1e-12*jnp.max(jnp.sqrt(gsq/shots)))   # precondition
                g = _smooth_gradient(g, 1.0) * imask; g = g/(jnp.max(jnp.abs(g))+1e-30); x = x - lr*g
                if prior_lr > 0:
                    xi = jnp.where(interior, x, 1500.)
                    s = model(((xi-SM)/SS).reshape(-1), t_eps).reshape(S, S) * imask
                    x = x + prior_lr * s/(jnp.max(jnp.abs(s))+1e-30)
                x = jnp.clip(x, 1400., 2900.)
        return np.asarray(x)

    m = np.asarray(interior) > 0.5; ct = np.asarray(c_true)
    for tag, plr in [("no prior", 0.0), ("DPS prior", 8.0)]:
        r = fwi(plr); rf = np.sqrt(np.mean((r[m]-ct[m])**2))
        print(f"{tag}: cerebrum RMSE -> {rf:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/brain_score_roi_unet.eqx")
    ap.add_argument("--norm", default="/tmp/brain_score_roi_unet_norm.npz")
    ap.add_argument("--subject", default="subj2")
    a = ap.parse_args()
    main(a.model, a.norm, a.subject)

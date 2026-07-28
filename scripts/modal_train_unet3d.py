"""Train the 3D U-Net brain-prior score model on Modal GPU.

Ships a prebuilt 3D Birnbaum cerebrum dataset (see scripts/build_birnbaum_3d_dataset.py),
trains via the Phase-3 train_score_matching, returns the model + a few sample volumes.
Parametrised by env vars so it serves both 48^3 (A10G) and 96^3 (A100):

    modal run scripts/modal_train_unet3d.py                       # 48^3 defaults
    BFWI_DATASET=/tmp/birnbaum_3d_dataset_96.npz BFWI_GPU=A100 \\
        BFWI_BATCH=4 BFWI_STEPS=4000 BFWI_TAG=96 \\
        modal run scripts/modal_train_unet3d.py                   # 96^3
"""
import io
import os
import modal

DATASET = os.environ.get("BFWI_DATASET", "/tmp/birnbaum_3d_dataset.npz")
GPU = os.environ.get("BFWI_GPU", "A10G")
BATCH = int(os.environ.get("BFWI_BATCH", "8"))
N_STEPS = int(os.environ.get("BFWI_STEPS", "3000"))
TAG = os.environ.get("BFWI_TAG", "")   # output suffix, e.g. "96" -> brain_score_3d_96.eqx

app = modal.App("brain-fwi-unet3d")
GIT_BRANCH = "feature/diffusion-prior-fwi"   # has inference.score_unet3d + birnbaum 3D
CACHE_BUST = "2026-06-18-unet3d-v2"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential").pip_install("uv")
    .env({"BRAIN_FWI_CACHE_BUST": CACHE_BUST})
    .run_commands(
        f"git clone --depth 1 --branch {GIT_BRANCH} https://github.com/m9h/brain-fwi.git /opt/brain-fwi",
        "cd /opt/brain-fwi && uv pip install --system -e '.[cuda12]'",
    )
    .add_local_file(DATASET, "/opt/brain-fwi/dataset.npz")
)


@app.function(image=image, gpu=GPU, timeout=120 * 60)
def train(n_steps: int = N_STEPS, batch: int = BATCH):
    import os, sys, time, numpy as np
    os.environ["JAX_PLATFORMS"] = "cuda"; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    sys.path.insert(0, "/opt/brain-fwi")
    import jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
    from brain_fwi.inference.score_unet3d import UNet3DScore
    from brain_fwi.inference.diffusion import VPSDE, train_score_matching, ddim_sample
    print("devices:", jax.devices())
    z = np.load("/opt/brain-fwi/dataset.npz")
    data, mean, std, S = z["data"], float(z["mean"]), float(z["std"]), int(z["S"])
    print(f"dataset {data.shape} S={S} mean {mean:.0f} std {std:.1f}, batch {batch}")
    data_std = jnp.asarray((data - mean) / std)
    model = UNet3DScore(S, C=16, key=jr.PRNGKey(0)); sde = VPSDE()
    t0 = time.time()
    model, losses = train_score_matching(model, data_std, sde, n_steps=n_steps, batch_size=batch,
                                         learning_rate=1e-3, key=jr.PRNGKey(1))
    print(f"trained {n_steps} in {time.time()-t0:.0f}s, loss {losses[0]:.1f}->{float(np.mean(losses[-50:])):.1f}")
    samp = np.asarray(ddim_sample(lambda th, t: model(th, t), sde, dim=S**3, n_samples=4, n_steps=50, key=jr.PRNGKey(2)))
    buf = io.BytesIO(); eqx.tree_serialise_leaves(buf, model)
    return buf.getvalue(), samp, mean, std, S, losses[0], float(np.mean(losses[-50:]))


@app.local_entrypoint()
def main():
    import numpy as np
    suffix = f"_{TAG}" if TAG else ""
    mb, samp, mean, std, S, l0, l1 = train.remote()
    open(f"/tmp/brain_score_3d{suffix}.eqx", "wb").write(mb)
    np.savez(f"/tmp/brain_score_3d{suffix}_norm.npz", mean=mean, std=std, S=S)
    print(f"loss {l0:.1f} -> {l1:.1f}; 3D model saved /tmp/brain_score_3d{suffix}.eqx ({len(mb)} bytes)")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    vol = (samp * std + mean).reshape(4, S, S, S)
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))   # mid-axial + mid-coronal of 4 samples
    for i in range(4):
        ax[0, i].imshow(np.clip(vol[i, S//2], 1480, 1680), cmap="viridis", vmin=1480, vmax=1680); ax[0, i].axis("off")
        ax[1, i].imshow(np.clip(vol[i, :, S//2], 1480, 1680), cmap="viridis", vmin=1480, vmax=1680); ax[1, i].axis("off")
    ax[0, 0].set_title("axial", fontsize=9); ax[1, 0].set_title("coronal", fontsize=9)
    plt.tight_layout(); plt.savefig(f"/tmp/unet3d_samples{suffix}.png", dpi=110)
    print(f"saved /tmp/unet3d_samples{suffix}.png")

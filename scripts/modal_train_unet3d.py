"""Train the 3D U-Net brain-prior score model on Modal GPU.

Ships the 3D Birnbaum cerebrum dataset + the 3D U-Net module, trains via the
Phase-3 train_score_matching, returns the model + a few sample volumes.
    modal run scripts/modal_train_unet3d.py
"""
from pathlib import Path
import io
import modal

app = modal.App("brain-fwi-unet3d")
GIT_BRANCH = "feature/parallel-modal-phase0"
CACHE_BUST = "2026-06-17-unet3d-v1"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential").pip_install("uv")
    .env({"BRAIN_FWI_CACHE_BUST": CACHE_BUST})
    .run_commands(
        f"git clone --depth 1 --branch {GIT_BRANCH} https://github.com/m9h/brain-fwi.git /opt/brain-fwi",
        "cd /opt/brain-fwi && uv pip install --system -e '.[cuda12]'",
    )
    .add_local_file("/tmp/birnbaum_3d_dataset.npz", "/opt/brain-fwi/birnbaum_3d_dataset.npz")
)


@app.function(image=image, gpu="A10G", timeout=90 * 60)
def train(n_steps: int = 3000, batch: int = 8):
    import os, sys, time, numpy as np
    os.environ["JAX_PLATFORMS"] = "cuda"; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    sys.path.insert(0, "/opt/brain-fwi")
    import jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
    from brain_fwi.inference.score_unet3d import UNet3DScore
    from brain_fwi.inference.diffusion import VPSDE, train_score_matching, ddim_sample
    print("devices:", jax.devices())
    z = np.load("/opt/brain-fwi/birnbaum_3d_dataset.npz")
    data, mean, std, S = z["data"], float(z["mean"]), float(z["std"]), int(z["S"])
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
    mb, samp, mean, std, S, l0, l1 = train.remote()
    open("/tmp/brain_score_3d.eqx", "wb").write(mb)
    np.savez("/tmp/brain_score_3d_norm.npz", mean=mean, std=std, S=S)
    print(f"loss {l0:.1f} -> {l1:.1f}; 3D model saved ({len(mb)} bytes)")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    vol = (samp * std + mean).reshape(4, S, S, S)
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))   # mid-axial + mid-coronal of 4 samples
    for i in range(4):
        ax[0, i].imshow(np.clip(vol[i, S//2], 1480, 1680), cmap="viridis", vmin=1480, vmax=1680); ax[0, i].axis("off")
        ax[1, i].imshow(np.clip(vol[i, :, S//2], 1480, 1680), cmap="viridis", vmin=1480, vmax=1680); ax[1, i].axis("off")
    ax[0, 0].set_title("axial", fontsize=9); ax[1, 0].set_title("coronal", fontsize=9)
    plt.tight_layout(); plt.savefig("/tmp/unet3d_samples.png", dpi=110); print("saved /tmp/unet3d_samples.png")

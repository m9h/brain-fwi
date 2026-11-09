"""Train the U-Net brain-prior score model on Modal GPU (NN training, not j-Wave).

Ships the local Birnbaum dataset (.npz) + the U-Net module, trains via the
repo's Phase-3 train_score_matching, and returns the serialized model + samples.

    modal run scripts/modal_train_unet_score.py
"""
from pathlib import Path
import io
import modal

app = modal.App("brain-fwi-unet-score")
GIT_BRANCH = "feature/parallel-modal-phase0"
CACHE_BUST = "2026-06-17-unet-score-v1"
ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential").pip_install("uv")
    .env({"BRAIN_FWI_CACHE_BUST": CACHE_BUST})
    .run_commands(
        f"git clone --depth 1 --branch {GIT_BRANCH} https://github.com/m9h/brain-fwi.git /opt/brain-fwi",
        "cd /opt/brain-fwi && uv pip install --system -e '.[cuda12]'",
    )
    .add_local_file("/tmp/unet_score.py", "/opt/brain-fwi/unet_score.py")
    .add_local_file("/tmp/birnbaum_dataset.npz", "/opt/brain-fwi/birnbaum_dataset.npz")
)


@app.function(image=image, gpu="A10G", timeout=60 * 60)
def train(n_steps: int = 4000):
    import os, sys, numpy as np
    os.environ["JAX_PLATFORMS"] = "cuda"; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    sys.path.insert(0, "/opt/brain-fwi")
    import jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
    from unet_score import UNetScore
    from brain_fwi.inference.diffusion import VPSDE, train_score_matching, ddim_sample
    print("devices:", jax.devices())
    z = np.load("/opt/brain-fwi/birnbaum_dataset.npz")
    data, mean, std = z["data"], float(z["mean"]), float(z["std"])
    data_std = jnp.asarray((data - mean) / std)
    model = UNetScore(64, 64, C=32, key=jr.PRNGKey(0)); sde = VPSDE()
    import time; t0 = time.time()
    model, losses = train_score_matching(model, data_std, sde, n_steps=n_steps, batch_size=32,
                                         learning_rate=1e-3, key=jr.PRNGKey(1))
    print(f"trained {n_steps} in {time.time()-t0:.0f}s, loss {losses[0]:.1f}->{float(np.mean(losses[-50:])):.1f}")
    samp = np.asarray(ddim_sample(lambda th, t: model(th, t), sde, dim=64*64, n_samples=6, n_steps=60, key=jr.PRNGKey(2)))
    buf = io.BytesIO(); eqx.tree_serialise_leaves(buf, model)
    return buf.getvalue(), samp, mean, std, losses[0], float(np.mean(losses[-50:]))


@app.local_entrypoint()
def main():
    import numpy as np
    model_bytes, samp, mean, std, l0, l1 = train.remote()
    open("/tmp/brain_score_real_unet.eqx", "wb").write(model_bytes)
    np.savez("/tmp/brain_score_real_unet_norm.npz", mean=mean, std=std)
    print(f"loss {l0:.1f} -> {l1:.1f}; model saved ({len(model_bytes)} bytes)")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    img = samp * std + mean
    fig, ax = plt.subplots(1, 6, figsize=(15, 3))
    for i in range(6):
        ax[i].imshow(np.clip(img[i].reshape(64, 64), 1480, 1680), cmap="viridis", vmin=1480, vmax=1680); ax[i].axis("off")
    plt.tight_layout(); plt.savefig("/tmp/unet_samples.png", dpi=120); print("saved /tmp/unet_samples.png")

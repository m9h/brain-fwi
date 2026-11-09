"""Train the FNO forward-surrogate proof-of-concept on Modal (Track-2 / Phase-4).

Self-contained: ships the starter dataset (scripts/fno_gen_dataset.py output),
trains CToTraceFNO3D (velocity + source -> helmet traces) with a relative-L2 loss,
and reports held-out forward correlation -- the surrogate's agreement with j-Wave,
the same metric the cross-solver harness uses. PoC scale; a useful surrogate needs
Phase-0 scale.   BFWI_DATASET=/tmp/fno_dataset_64.npz .venv/bin/modal run scripts/modal_train_fno_poc.py
"""
import io, os
import modal

DATASET = os.environ.get("BFWI_DATASET", "/tmp/fno_dataset_64.npz")
app = modal.App("brain-fwi-fno-poc")
GIT_BRANCH = "feature/diffusion-prior-fwi"; CACHE_BUST = "2026-06-21-fnopoc-v1"
image = (
    modal.Image.debian_slim(python_version="3.11").apt_install("git", "build-essential").pip_install("uv")
    .env({"BRAIN_FWI_CACHE_BUST": CACHE_BUST})
    .run_commands(f"git clone --depth 1 --branch {GIT_BRANCH} https://github.com/m9h/brain-fwi.git /opt/brain-fwi",
                  "cd /opt/brain-fwi && uv pip install --system -e '.[cuda12]'")
    .add_local_file(DATASET, "/opt/brain-fwi/fno_dataset.npz")
)


@app.function(image=image, gpu="A100", timeout=90 * 60)
def train(n_steps: int = 3000, batch: int = 8, hidden: int = 32, modes: int = 12):
    import os, sys, time, numpy as np
    os.environ["JAX_PLATFORMS"] = "cuda"; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    sys.path.insert(0, "/opt/brain-fwi")
    import jax, jax.numpy as jnp, jax.random as jr, equinox as eqx, optax
    from brain_fwi.surrogate.fno3d import CToTraceFNO3D
    print("devices:", jax.devices(), flush=True)
    z = np.load("/opt/brain-fwi/fno_dataset.npz")
    C = z["sound_speed"].astype(np.float32); SRC = z["sources"]; D = z["traces"].astype(np.float32)
    recv = [tuple(int(x) for x in r) for r in z["receivers"]]
    n, S = len(C), int(z["S"]); nt = D.shape[1]; nr = len(recv)
    cmin, cmax = 1450.0, 2800.0
    Cn = (C - cmin) / (cmax - cmin)
    # per-trace scale so rel-L2 is well-posed; normalise data globally
    dscale = float(np.sqrt(np.mean(D ** 2))); Dn = D / dscale
    ntr = int(0.85 * n); idx = np.random.default_rng(0).permutation(n); tr, va = idx[:ntr], idx[ntr:]
    print(f"dataset {n} samples, {S}^3, nt {nt}, nr {nr}; train {len(tr)} val {len(va)}", flush=True)

    model = CToTraceFNO3D(grid_shape=(S, S, S), n_timesteps=nt, n_receivers=nr,
                          receiver_positions=tuple(recv), hidden_channels=hidden, num_modes=modes,
                          depth=2, key=jr.PRNGKey(0))
    opt = optax.adam(3e-4); opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    def rel_l2(pred, tgt): return jnp.linalg.norm(pred - tgt) / (jnp.linalg.norm(tgt) + 1e-8)

    @eqx.filter_jit
    def sample_grad(m, cn, sp, d):
        def loss_fn(mm): return rel_l2(mm(cn, sp), d)
        return eqx.filter_value_and_grad(loss_fn)(m)

    key = jr.PRNGKey(1); losses = []
    t0 = time.time()
    for step in range(n_steps):
        gacc = None; lacc = 0.0
        for _ in range(batch):
            key, k = jr.split(key); i = int(jr.randint(k, (), 0, len(tr))); j = tr[i]
            l, g = sample_grad(model, jnp.asarray(Cn[j]), tuple(int(x) for x in SRC[j]), jnp.asarray(Dn[j]))
            lacc += float(l); gacc = g if gacc is None else jax.tree.map(lambda a, b: a + b, gacc, g)
        gacc = jax.tree.map(lambda x: x / batch, gacc)
        updates, opt_state = opt.update(gacc, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates); losses.append(lacc / batch)
        if (step + 1) % 200 == 0:
            print(f"  step {step+1}/{n_steps} rel-L2 {np.mean(losses[-200:]):.3f} ({time.time()-t0:.0f}s)", flush=True)

    # validation: forward correlation + rel-L2 on held-out
    cors, rls = [], []
    for j in va:
        pred = np.asarray(model(jnp.asarray(Cn[j]), tuple(int(x) for x in SRC[j])))
        tgt = Dn[j]
        rls.append(float(np.linalg.norm(pred - tgt) / (np.linalg.norm(tgt) + 1e-8)))
        a, b = pred.ravel(), tgt.ravel()
        cors.append(float(np.mean((a - a.mean()) / (a.std() + 1e-9) * (b - b.mean()) / (b.std() + 1e-9))))
    print(f"VAL: forward corr mean {np.mean(cors):.3f}  rel-L2 mean {np.mean(rls):.3f}", flush=True)
    buf = io.BytesIO(); eqx.tree_serialise_leaves(buf, model)
    return buf.getvalue(), float(np.mean(cors)), float(np.mean(rls)), losses[0], float(np.mean(losses[-50:]))


@app.local_entrypoint()
def main():
    mb, corr, rl, l0, l1 = train.remote()
    open("/tmp/fno_poc.eqx", "wb").write(mb)
    print(f"\nFNO PoC: loss {l0:.3f}->{l1:.3f}; VAL forward corr {corr:.3f}, rel-L2 {rl:.3f}; saved /tmp/fno_poc.eqx")

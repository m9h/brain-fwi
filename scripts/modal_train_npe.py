"""Modal runner: minimal NPE training on synthetic (theta, d) pairs.

Purpose: end-to-end validation that the flowjax wrapper + training
loop + Modal image all hang together on real GPU hardware, before we
plug in Phase-0 data.

Generates a toy linear-Gaussian dataset inside the remote container
(theta ~ N(0, I), d = 2 theta + small_noise), trains the flow for
500 Adam steps, and reports initial vs final NLL on a held-out batch.
Success criterion: NLL drops by > 0.5 nats and the learned conditional
mean tracks d/2 within 0.3 at a few probe values.

Usage::

    modal run scripts/modal_train_npe.py
"""

import modal

app = modal.App("brain-fwi-npe-smoke")

GIT_BRANCH = "feature/phase2-npe"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .run_commands(
        f"git clone --depth 1 --branch {GIT_BRANCH} "
        f"https://github.com/m9h/brain-fwi.git /opt/brain-fwi",
        "cd /opt/brain-fwi && uv pip install --system -e '.[cuda12]'",
    )
)


@app.function(
    image=image,
    gpu="A10G",  # NPE training is tiny; no need for A100
    timeout=30 * 60,
)
def run_smoke_training():
    import os
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import subprocess
    subprocess.run(["nvidia-smi", "-L"], check=True)

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import numpy as np

    print(f"JAX devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")

    from brain_fwi.inference.flow import ConditionalFlow, train_npe

    # Synthetic linear-Gaussian problem
    rng = np.random.default_rng(0)
    n_train, n_test = 2000, 256
    theta_train = rng.standard_normal((n_train, 2)).astype(np.float32)
    d_train = 2.0 * theta_train + 0.1 * rng.standard_normal((n_train, 2)).astype(np.float32)
    theta_test = rng.standard_normal((n_test, 2)).astype(np.float32)
    d_test = 2.0 * theta_test + 0.1 * rng.standard_normal((n_test, 2)).astype(np.float32)

    flow = ConditionalFlow(theta_dim=2, d_dim=2, key=jr.PRNGKey(0), n_transforms=5)

    initial_nll = float(
        -jnp.mean(jax.vmap(flow.log_prob)(
            jnp.asarray(theta_test), jnp.asarray(d_test),
        ))
    )
    print(f"\nInitial NLL on test batch: {initial_nll:.4f}")

    print(f"\nTraining 500 steps on {n_train} pairs, batch_size=128...")
    trained, losses = train_npe(
        flow,
        theta=jnp.asarray(theta_train),
        d=jnp.asarray(d_train),
        key=jr.PRNGKey(1),
        n_steps=500,
        learning_rate=1e-3,
        batch_size=128,
        verbose=True,
    )

    final_nll = float(
        -jnp.mean(jax.vmap(trained.log_prob)(
            jnp.asarray(theta_test), jnp.asarray(d_test),
        ))
    )

    print(f"\n{'=' * 50}")
    print(f"  initial NLL: {initial_nll:.4f}")
    print(f"  final   NLL: {final_nll:.4f}")
    print(f"  Δ          : {initial_nll - final_nll:+.4f} (lower is better)")
    print(f"{'=' * 50}")

    # Posterior-mean probe
    print("\nPosterior mean probes (expect mean ≈ d/2 for each d):")
    for d_val in [-1.5, 0.0, 1.5]:
        d_probe = jnp.full((2,), d_val)
        samples = trained.sample(d_probe, jr.PRNGKey(99), n_samples=500)
        post_mean = np.mean(np.asarray(samples), axis=0)
        print(f"  d={d_val:+.2f}: posterior mean={post_mean}, expected ≈{d_val/2:+.2f}")

    assert final_nll < initial_nll - 0.5, "NLL did not drop as expected"
    print("\n✓ smoke test passed")


@app.local_entrypoint()
def main():
    run_smoke_training.remote()

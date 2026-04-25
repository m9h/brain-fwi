"""Toy 2D FNO on acoustic wave.

Validates the FNO architecture family on a 2D homogeneous acoustic wave
problem before scaling to 3D. Target: relative-L2 < 1% on held-out traces.

Evidence 9.2.1 from ``docs/design/phase4_fno_surrogate.md``.

Plan:
1. Synthetic 2D (32²) dataset: N = 500 pairs, single-source
   single-receiver, random `c` (Gaussian bump on homogeneous bg).
2. Generate via ``brain_fwi.simulation.forward``.
3. Train FNO (width 16, 4 blocks, 12 modes).
4. Measure held-out trace relative-L2.
"""

import os
import time
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import numpy as np
from typing import Tuple, List
from tqdm import tqdm

from brain_fwi.simulation.forward import (
    build_domain,
    build_medium,
    build_time_axis,
    simulate_shot_sensors,
    _build_source_signal,
)
from brain_fwi.surrogate.fno2d import CToTraceFNO

# --- Data Generation -------------------------------------------------------

def generate_dataset(n_samples=500, grid_size=32, key=jr.PRNGKey(0)):
    dx = 0.005
    freq = 3.0e5
    dt = 2e-7
    t_end = 20e-6
    n_t = int(t_end / dt)
    
    grid_shape = (grid_size, grid_size)
    domain = build_domain(grid_shape, dx)
    
    # Sensors: single receiver in center
    cx = grid_size // 2
    sensor_pos = (jnp.array([cx+4]), jnp.array([cx+4]))
    src_pos = (cx-4, cx-4)
    signal = _build_source_signal(freq, dt, n_t)
    
    c_bg = 1500.0
    rho = 1000.0
    
    c_list = []
    d_list = []
    
    keys = jr.split(key, n_samples)
    
    print(f"Generating {n_samples} samples (2D {grid_size}^2)...")
    for i in tqdm(range(n_samples)):
        # Random Gaussian bump for sound speed
        k1, k2, k3 = jr.split(keys[i], 3)
        center = jr.uniform(k1, (2,), minval=10, maxval=22)
        amplitude = jr.uniform(k2, (), minval=50, maxval=300)
        sigma = jr.uniform(k3, (), minval=2, maxval=5)
        
        x, y = jnp.meshgrid(jnp.arange(grid_size), jnp.arange(grid_size), indexing="ij")
        r2 = (x - center[0])**2 + (y - center[1])**2
        bump = amplitude * jnp.exp(-r2 / (2 * sigma**2))
        c = c_bg + bump
        
        medium = build_medium(domain, c, rho, pml_size=10)
        time_axis = build_time_axis(medium, t_end=t_end) # uses dt from medium but we want to be consistent
        # Use fixed time axis if possible to avoid shape mismatch
        from jwave.geometry import TimeAxis
        time_axis = TimeAxis(t_end=t_end, dt=dt)
        
        d = simulate_shot_sensors(
            medium, time_axis, src_pos, sensor_pos, signal, dt
        )
        # d shape: (n_t, 1)
        
        c_list.append(c)
        d_list.append(d[:, 0])
        
    return jnp.stack(c_list), jnp.stack(d_list)

# --- Training Loop ---------------------------------------------------------

def train():
    n_samples = 500
    grid_size = 32
    seed = 42
    key = jr.PRNGKey(seed)
    
    data_key, model_key, train_key = jr.split(key, 3)
    
    # Data
    c_all, d_all = generate_dataset(n_samples, grid_size, data_key)
    
    # Normalize
    c_mean, c_std = c_all.mean(), c_all.std()
    d_mean, d_std = d_all.mean(), d_all.std()
    
    c_all = (c_all - c_mean) / c_std
    d_all = (d_all - d_mean) / d_std
    
    # Split
    n_train = 400
    c_tr, c_te = c_all[:n_train], c_all[n_train:]
    d_tr, d_te = d_all[:n_train], d_all[n_train:]
    
    # FNO model: maps c(x,y) -> d(t, fixed_receiver)
    # This is a bit non-standard for FNO (usually maps image to image).
    # We can treat d(t) as a 1D "image" or just use FNO to produce a hidden 
    # representation and then project to d(t).
    # Alternatively, map c(x,y) to p(x,y,t) and sample at receiver? No, too big.
    # Let's map (1, 32, 32) -> (1, 100, 1) where 100 is n_t.
    
    # We'll use FNO2d to map (1, 32, 32) -> (width, 32, 32) 
    # then global pool or just sample at src/recv positions to get features?
    # Actually, let's keep it simple: 
    # FNO2d maps (1, 32, 32) -> (n_t/32, 32, 1) then flatten.
    
    n_t = d_all.shape[1]
    model = CToTraceFNO(
        grid_h=grid_size,
        grid_w=grid_size,
        n_timesteps=n_t,
        width=64,
        modes=12,
        depth=2,
        key=model_key
    )
    
    def predict(model, c):
        # c is (32, 32)
        return model(c[..., jnp.newaxis])

    def loss_fn(model, c_batch, d_batch):
        preds = jax.vmap(predict, in_axes=(None, 0))(model, c_batch)
        return jnp.mean((preds - d_batch)**2)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    @eqx.filter_jit
    def step(model, opt_state, c_batch, d_batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, c_batch, d_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # Training
    n_epochs = 200
    batch_size = 32
    print("\nTraining FNO...")
    for epoch in range(n_epochs):
        key, subkey = jr.split(train_key)
        perm = jr.permutation(subkey, n_train)
        epoch_losses = []
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            model, opt_state, loss = step(model, opt_state, c_tr[idx], d_tr[idx])
            epoch_losses.append(loss)
        
        if (epoch + 1) % 20 == 0:
            avg_loss = np.mean(epoch_losses)
            print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.6f}")

    # Evaluation
    preds_te = jax.vmap(predict, in_axes=(None, 0))(model, c_te)
    
    # Unnormalize for rel-L2
    preds_te_unnorm = preds_te * d_std + d_mean
    d_te_unnorm = d_te * d_std + d_mean
    
    rel_l2 = jnp.linalg.norm(preds_te_unnorm - d_te_unnorm, axis=1) / jnp.linalg.norm(d_te_unnorm, axis=1)
    p50 = jnp.median(rel_l2)
    p95 = jnp.percentile(rel_l2, 95)
    
    print("\n" + "=" * 60)
    print("  FNO Toy 2D Result")
    print("=" * 60)
    print(f"  Median relative-L2: {p50*100:.2f}%")
    print(f"  95th percentile:    {p95*100:.2f}%")
    print("-" * 60)
    
    if p50 < 0.01:
        print("  VERDICT: ✅ PROCEED to 3D MVP")
    else:
        print("  VERDICT: ❌ REVISE DESIGN (accuracy target not met)")

if __name__ == "__main__":
    train()

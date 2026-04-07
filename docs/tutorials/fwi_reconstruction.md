# Full Waveform Inversion: Recovering Sound Speed from Ultrasound Data

This tutorial covers the Full Waveform Inversion (FWI) pipeline in brain-fwi:
the optimization loop that iteratively updates a sound speed model to match
observed ultrasound data. We cover reparameterization, loss functions, gradient
processing, multi-frequency banding, and convergence diagnostics.

## The FWI problem

Given observed pressure data $d_{\text{obs}}$ recorded by an array of transducers
around the head, FWI seeks the sound speed distribution $c(\mathbf{x})$ that
minimizes the data misfit:

$$
\hat{c} = \arg\min_{c} \sum_{s=1}^{N_s} \mathcal{L}\bigl(
  F[c, \mathbf{x}_s],\; d_{\text{obs}}^{(s)}
\bigr)
$$

where $F[c, \mathbf{x}_s]$ is the forward operator (simulate from source $s$,
record at all receivers) and $\mathcal{L}$ is a loss function measuring the
waveform misfit.

### Why JAX autodiff?

Traditional FWI implementations (Stride, Devito) compute the gradient via the
**adjoint-state method** -- a hand-derived mathematical formula that requires
implementing a separate adjoint solver. Brain FWI instead uses JAX automatic
differentiation:

$$
\nabla_c \mathcal{L} = \texttt{jax.grad}(\mathcal{L} \circ F)
$$

This gives **exact gradients** with zero implementation effort, enables
higher-order optimization, and integrates naturally with neural networks.

## Step 1: Set up the problem

```python
import jax.numpy as jnp
from brain_fwi.phantoms.brainweb import make_synthetic_head
from brain_fwi.phantoms.properties import map_labels_to_speed, map_labels_to_density
from brain_fwi.transducers.helmet import ring_array_2d, transducer_positions_to_grid
from brain_fwi.simulation.forward import (
    build_domain, build_medium, build_time_axis,
    generate_observed_data, _build_source_signal,
)

# Create a synthetic head phantom
grid_shape = (128, 128)
dx = 0.001  # 1 mm
labels, props = make_synthetic_head(grid_shape=grid_shape, dx=dx)
c_true = props["sound_speed"]    # (128, 128) -- ground truth
rho = props["density"]           # held fixed during FWI

# Transducer ring
center = (0.064, 0.064)
positions = ring_array_2d(n_elements=64, center=center,
                          semi_major=0.055, semi_minor=0.045)
sensor_grid = transducer_positions_to_grid(positions, dx, grid_shape)
src_list = [(int(sensor_grid[0][i]), int(sensor_grid[1][i]))
            for i in range(64)]

# Time axis from c_max (ensures CFL stability for all velocity updates)
c_min, c_max = 1400.0, 3200.0
freq = 100e3
domain = build_domain(grid_shape, dx)
ref_medium = build_medium(domain, c_max, 1000.0, pml_size=20)
time_axis = build_time_axis(ref_medium, cfl=0.3, t_end=100e-6)
dt = float(time_axis.dt)
t_end = float(time_axis.t_end)
n_samples = int(t_end / dt)
source_signal = _build_source_signal(freq, dt, n_samples)
```

## Step 2: Generate observed data

```python
observed = generate_observed_data(
    c_true, rho, dx,
    src_list, sensor_grid, freq,
    pml_size=20,
    time_axis=time_axis,
    source_signal=source_signal,
    dt=dt,
    verbose=True,
)
# observed.shape == (64, n_timesteps, 64)
```

## Step 3: Sigmoid reparameterization

Direct optimization of $c(\mathbf{x})$ is ill-conditioned because sound speed
has physical bounds (water at ~1400 m/s to cortical bone at ~3200 m/s). Brain FWI
uses a sigmoid reparameterization:

$$
c(\mathbf{x}) = c_{\min} + (c_{\max} - c_{\min}) \cdot \sigma\bigl(\theta(\mathbf{x})\bigr)
$$

where $\theta$ is the unconstrained optimization variable and $\sigma$ is the
logistic sigmoid. This maps $\theta \in (-\infty, +\infty)$ to
$c \in (c_{\min}, c_{\max})$.

```python
from brain_fwi.inversion.fwi import _params_to_velocity, _velocity_to_params

# Convert initial guess to unconstrained parameters
c_init = jnp.full(grid_shape, 1500.0)  # homogeneous water
params = _velocity_to_params(c_init, c_min, c_max)

# Recover velocity from parameters
c_recovered = _params_to_velocity(params, c_min, c_max)
# c_recovered ~ 1500.0 everywhere (roundtrip)
```

:::{admonition} Why not optimize $c$ directly?
:class: note

Without reparameterization, gradient descent can push $c$ to unphysical values
(negative speed, or beyond bone velocity). The sigmoid enforces bounds smoothly
and improves the optimization landscape -- the gradient is always well-defined
and the Hessian is better conditioned near the bounds.
:::

## Step 4: Loss functions

Brain FWI provides three loss functions, each with different trade-offs:

### L2 waveform loss

The simplest misfit -- sum of squared differences:

$$
\mathcal{L}_{\text{L2}} = \frac{1}{2} \sum_{t,r} \bigl(p_{\text{pred}}(t,r) - p_{\text{obs}}(t,r)\bigr)^2
$$

```python
from brain_fwi.inversion.losses import l2_loss

loss = l2_loss(predicted, observed)
```

**Pros**: Simple, fast, good near the true model.
**Cons**: Susceptible to **cycle-skipping** -- if the initial model is off by
more than half a wavelength, the gradient points in the wrong direction.

### Hilbert envelope loss

Compares amplitude envelopes rather than raw waveforms:

$$
\mathcal{L}_{\text{env}} = \frac{1}{2}
\frac{\sum_{t,r} \bigl(|\mathcal{H}(p_{\text{pred}})| - |\mathcal{H}(p_{\text{obs}})|\bigr)^2}
{\sum_{t,r} |\mathcal{H}(p_{\text{obs}})|^2}
$$

where $\mathcal{H}$ is the Hilbert transform (computed via FFT).

```python
from brain_fwi.inversion.losses import envelope_loss

loss = envelope_loss(predicted, observed)
```

**Pros**: Much more convex basin of attraction -- robust to cycle-skipping.
**Cons**: Loses phase information, so convergence to fine detail is slower.

### Multiscale (combined) loss

Balances waveform fidelity with robustness:

$$
\mathcal{L}_{\text{multi}} = (1 - w) \, \mathcal{L}_{\text{L2}} + w \, \mathcal{L}_{\text{env}}
$$

```python
from brain_fwi.inversion.losses import multiscale_loss

loss = multiscale_loss(predicted, observed, envelope_weight=0.5)
```

This is the default in brain-fwi, following j-Wave's FWI notebook.

## Step 5: Multi-frequency banding

Cycle-skipping is worse at high frequencies. Brain FWI follows the **Stride
pattern** of multi-frequency banding: start with low frequencies (large
wavelength, convex misfit) and progressively increase:

| Band | Frequency range | Wavelength in brain | Purpose |
|------|----------------|-------------------|---------|
| 1    | 50--100 kHz    | 15--30 mm         | Coarse structure (convex) |
| 2    | 100--200 kHz   | 8--15 mm          | Refine skull boundaries |
| 3    | 200--300 kHz   | 5--8 mm           | Fine detail |

At each band, the source signal and observed data are bandpass-filtered using
smooth cosine-tapered FFT filters.

## Step 6: Run FWI

```python
from brain_fwi.inversion.fwi import FWIConfig, run_fwi

config = FWIConfig(
    freq_bands=[(50e3, 100e3), (100e3, 200e3), (200e3, 300e3)],
    n_iters_per_band=30,
    shots_per_iter=4,          # stochastic source selection
    learning_rate=5.0,         # Adam learning rate
    c_min=1400.0,
    c_max=3200.0,
    pml_size=20,
    gradient_smooth_sigma=3.0, # Gaussian smoothing (grid points)
    loss_fn="multiscale",
    envelope_weight=0.5,
    verbose=True,
)

result = run_fwi(
    observed_data=observed,
    initial_velocity=c_init,
    density=rho,
    dx=dx,
    src_positions_grid=src_list,
    sensor_positions_grid=sensor_grid,
    source_signal=source_signal,
    dt=dt,
    t_end=t_end,
    config=config,
)

c_reconstructed = result.velocity       # (128, 128)
loss_history = result.loss_history      # list of floats
velocity_snapshots = result.velocity_history  # one per band
```

## The FWI loop in detail

Each iteration of the FWI loop performs:

1. **Stochastic source selection**: Randomly pick `shots_per_iter` sources from
   the full array. This is standard in large-scale FWI (Stride uses 16 shots
   per iteration for breast USCT).

2. **Forward simulation**: For each selected source, run the forward operator
   and record at all sensors.

3. **Loss computation**: Compare bandpass-filtered predicted and observed data.

4. **Gradient via JAX autodiff**: `jax.value_and_grad` computes the gradient of
   the loss with respect to the unconstrained parameters $\theta$.

5. **Gradient smoothing**: Apply a Gaussian filter to suppress high-frequency
   artifacts. This is standard in FWI (both Stride and j-Wave use it).

6. **Mask application**: Zero out gradients outside the region of interest
   (e.g., inside PML, in the water coupling medium).

7. **Adam optimizer update**: The smoothed, masked gradient is passed to Optax's
   Adam optimizer.

### Gradient computation

The FWI gradient is computed per-shot and accumulated:

$$
\nabla_\theta \mathcal{L} = \frac{1}{N_{\text{shots}}}
\sum_{s \in \text{batch}} \nabla_\theta \mathcal{L}_s
$$

This sequential-accumulation approach uses $O(1\text{ shot})$ memory instead of
$O(N_{\text{shots}})$, which is critical for 3D where a single simulation can use
several GB of VRAM.

## Step 7: Convergence diagnostics

```python
import matplotlib.pyplot as plt

# Loss curve
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.semilogy(result.loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("FWI Convergence")

# Velocity comparison
plt.subplot(1, 2, 2)
plt.imshow(result.velocity.T, cmap="seismic",
           vmin=1400, vmax=3200, origin="lower")
plt.colorbar(label="Sound speed (m/s)")
plt.title("Reconstructed velocity")
plt.tight_layout()
plt.show()
```

### What to look for

- **Loss decreasing monotonically**: If the loss oscillates or increases, reduce
  the learning rate or increase gradient smoothing.
- **Skull boundaries visible**: The skull should appear as a high-velocity ring
  in the reconstruction.
- **No PML artifacts**: If bright artifacts appear at the grid edges, increase
  `pml_size` or apply a tighter inversion mask.
- **Band transitions**: The loss may jump up when switching frequency bands
  (the new band introduces finer-scale misfit). This is normal.

## Advanced: Gauss-Newton and higher-order methods

The default Adam optimizer is a first-order method. For faster convergence near
the solution, you can use the resolution analysis module to approximate the
Hessian:

```python
from brain_fwi.inversion.resolution import compute_sensitivity_map

# Sensitivity map: diagonal of J^T J
sensitivity = compute_sensitivity_map(
    grid_shape, dx, n_elements=64, freq=100e3,
)
# Use as a preconditioner: scale gradient by 1/sensitivity
```

The Jacobian $J = \partial d / \partial c$ relates data perturbations to velocity
perturbations. The resolution matrix $R = (J^T J + \lambda I)^{-1} J^T J$
indicates the achievable imaging resolution at each point.

## Next steps

- {doc}`head_phantoms` -- Create realistic head models for FWI validation
- {doc}`forward_simulation` -- Details of the forward operator

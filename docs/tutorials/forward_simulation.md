# Acoustic Forward Simulation: From Head Model to Sensor Data

This tutorial walks through the forward acoustic simulation pipeline in brain-fwi:
constructing a computational domain, assigning medium properties, placing sources
and sensors, solving the wave equation via j-Wave, and recording pressure data.

The forward operator is the inner loop of Full Waveform Inversion -- every FWI
iteration runs one or more forward simulations to compare predicted and observed
data. Understanding the forward model is therefore essential.

## The acoustic wave equation

Brain FWI solves the linear acoustic wave equation in heterogeneous media:

$$
\frac{1}{c(\mathbf{x})^2} \frac{\partial^2 p}{\partial t^2}
= \nabla \cdot \left( \frac{1}{\rho(\mathbf{x})} \nabla p \right) + s(\mathbf{x}, t)
$$

where:

- $p(\mathbf{x}, t)$ is the acoustic pressure field (Pa)
- $c(\mathbf{x})$ is the spatially varying sound speed (m/s)
- $\rho(\mathbf{x})$ is the density (kg/m^3)
- $s(\mathbf{x}, t)$ is the source term

j-Wave solves this using the **pseudospectral time-domain (PSTD)** method, which
computes spatial derivatives via FFT. This gives spectral accuracy with far fewer
grid points per wavelength than finite differences (2--4 PPW vs. 10--20 PPW for
FDTD).

## Step 1: Build the computational domain

The domain defines the spatial grid on which the simulation runs.

```python
from brain_fwi.simulation.forward import build_domain

# 2D domain: 128x128 grid at 1mm spacing (12.8 cm total)
grid_shape = (128, 128)
dx = 0.001  # 1 mm in metres

domain = build_domain(grid_shape, dx)
# domain.N == (128, 128)
# domain.dx == (0.001, 0.001)
```

:::{admonition} Grid spacing rule of thumb
:class: tip

For accurate pseudospectral simulation, you need at least **6 points per
wavelength** at the highest frequency:

$$
\Delta x \leq \frac{c_{\min}}{6 \, f_{\max}}
$$

For brain FWI at 300 kHz with $c_{\min} = 1400$ m/s (water/CSF), this gives
$\Delta x \leq 0.78$ mm. A 1 mm grid is adequate for frequencies up to ~230 kHz;
use 0.5 mm for the full 300 kHz band.
:::

## Step 2: Assign medium properties

The medium holds the spatially varying sound speed and density fields, plus the
perfectly matched layer (PML) absorbing boundary.

```python
import jax.numpy as jnp
from brain_fwi.simulation.forward import build_medium

# Homogeneous water medium
medium = build_medium(domain, sound_speed=1500.0, density=1000.0, pml_size=20)

# Heterogeneous medium (e.g., from a head phantom)
c_map = jnp.ones(grid_shape) * 1500.0       # background: water
c_map = c_map.at[40:88, 40:88].set(1560.0)  # brain region
c_map = c_map.at[35:93, 35:40].set(2800.0)  # skull layer (left)

rho_map = jnp.ones(grid_shape) * 1000.0
rho_map = rho_map.at[35:93, 35:40].set(1850.0)  # skull density

medium = build_medium(domain, c_map, rho_map, pml_size=20)
```

Under the hood, `build_medium` wraps the arrays as `jwave.FourierSeries` fields on
the domain and creates a `jwave.geometry.Medium` with the specified PML thickness.

:::{admonition} PML absorbing boundaries
:class: note

The PML (perfectly matched layer) absorbs outgoing waves at the grid boundary to
prevent artificial reflections. A `pml_size` of 20 grid points is the j-Wave
default. The PML region is "lost" for imaging -- place your transducers inside
the PML-free zone.
:::

## Step 3: Compute the time axis

The time step $\Delta t$ must satisfy the CFL stability condition for the
pseudospectral method:

$$
\Delta t \leq \text{CFL} \cdot \frac{\Delta x}{c_{\max}}
$$

`build_time_axis` computes this automatically from the medium properties:

```python
from brain_fwi.simulation.forward import build_time_axis

time_axis = build_time_axis(medium, cfl=0.3, t_end=80e-6)
# time_axis.dt ~ 0.2 us
# time_axis.t_end = 80 us (enough for waves to cross the domain twice)
```

:::{warning}
**TimeAxis must be computed outside JAX-traced scope.** `TimeAxis.from_medium()`
calls Python `float()`, which raises `ConcretizationTypeError` inside `jax.jit`
or `jax.grad`. In FWI, pre-compute the time axis using a reference medium with
$c_{\max}$ before entering the optimization loop.
:::

## Step 4: Create the source signal

Brain FWI uses the **Ricker wavelet** (second derivative of a Gaussian) as the
default source:

$$
s(t) = \left(1 - 2\pi^2 f_0^2 \tau^2\right) \exp\left(-\pi^2 f_0^2 \tau^2\right),
\quad \tau = t - t_{\text{delay}}
$$

```python
from brain_fwi.utils.wavelets import ricker_wavelet

freq = 100e3    # 100 kHz centre frequency
dt = float(time_axis.dt)
n_samples = int(80e-6 / dt)

signal = ricker_wavelet(freq, dt, n_samples)
# signal.shape == (n_samples,)
```

For narrowband excitation (e.g., therapeutic ultrasound), use a windowed
toneburst instead:

```python
from brain_fwi.utils.wavelets import toneburst

signal = toneburst(f0=500e3, dt=dt, n_cycles=5)
```

## Step 5: Place sources and sensors

Sources and sensors are defined by their **grid indices** (not physical
coordinates). The transducer module provides convenience functions:

```python
from brain_fwi.transducers.helmet import ring_array_2d, transducer_positions_to_grid

# Place 64 transducers on an elliptical ring around the head
center_m = (0.064, 0.064)  # centre of 128x128 grid at dx=1mm
positions = ring_array_2d(
    n_elements=64,
    center=center_m,
    semi_major=0.055,   # 5.5 cm (head radius + standoff)
    semi_minor=0.045,   # 4.5 cm
)
# positions.shape == (64, 2) -- physical coordinates in metres

# Convert to grid indices
sensor_grid = transducer_positions_to_grid(positions, dx, grid_shape)
# sensor_grid is a tuple of two (64,) int32 arrays: (ix_array, iy_array)

# Use each transducer alternately as source
src_list = [(int(sensor_grid[0][i]), int(sensor_grid[1][i]))
            for i in range(64)]
```

## Step 6: Run the forward simulation

### Single-shot simulation (full field)

For visualization -- returns the pressure field at the final timestep:

```python
from brain_fwi.simulation.forward import simulate_shot

p_field = simulate_shot(
    medium, time_axis,
    src_position_grid=(64, 20),  # source near left edge
    freq=100e3,
    checkpoint=True,             # save memory for backprop
)
# p_field shape matches the domain grid
```

### Single-shot with sensor recording (for FWI)

This is the core forward operator used inside FWI -- records pressure time series
at discrete sensor locations:

```python
from brain_fwi.simulation.forward import simulate_shot_sensors

data = simulate_shot_sensors(
    medium, time_axis,
    src_position_grid=(64, 20),
    sensor_positions_grid=sensor_grid,
    source_signal=signal,
    dt=dt,
)
# data.shape == (n_timesteps, n_sensors)
```

### Batched data generation (all sources)

Generate the full observed dataset for FWI -- one forward simulation per source:

```python
from brain_fwi.simulation.forward import generate_observed_data

observed = generate_observed_data(
    sound_speed=c_map,
    density=rho_map,
    dx=dx,
    src_positions_grid=src_list,
    sensor_positions_grid=sensor_grid,
    freq=100e3,
    pml_size=20,
    time_axis=time_axis,        # pass pre-computed for consistency
    source_signal=signal,
    dt=dt,
    verbose=True,
)
# observed.shape == (n_sources, n_timesteps, n_sensors)
```

:::{admonition} Consistency between data generation and FWI
:class: warning

Always pass the **same** `time_axis`, `source_signal`, and `dt` to both
`generate_observed_data` and `run_fwi`. If you let each compute its own time axis
from the medium, the different sound speed models will produce different $\Delta t$
values, causing waveform misalignment and FWI divergence.
:::

## Differentiability

The key innovation of brain-fwi is that the entire forward operator is
**differentiable via JAX autodiff**. You can compute gradients of any scalar
function of the sensor data with respect to the sound speed field:

```python
import jax

def misfit(sound_speed):
    domain = build_domain(grid_shape, dx)
    medium = build_medium(domain, sound_speed, rho_map, pml_size=20)
    pred = simulate_shot_sensors(
        medium, time_axis, src_list[0], sensor_grid, signal, dt,
    )
    return 0.5 * jnp.mean((pred - observed[0]) ** 2)

# Gradient of misfit w.r.t. sound speed -- this is the FWI kernel
loss_val, grad = jax.value_and_grad(misfit)(c_map)
# grad.shape == (128, 128) -- sensitivity at each grid point
```

This replaces the hand-coded adjoint-state gradient computation used in
Stride/Devito, with zero implementation effort and mathematically exact gradients.

## Next steps

- {doc}`fwi_reconstruction` -- Use the forward operator inside the FWI loop
- {doc}`head_phantoms` -- Create realistic head models for simulation

# Creating Head Phantoms: BrainWeb, MIDA, and Synthetic Models

This tutorial covers the three head phantom sources in brain-fwi: the BrainWeb
20 Normal Models (open access, 1 mm), the MIDA 153-structure model (500 um,
licensed), and the built-in synthetic generator (configurable, no download).
We explain the tissue-to-acoustic property mapping, the ITRUSST benchmark
values, and how to prepare phantoms for FWI simulation.

## Why head phantoms matter for FWI

Full Waveform Inversion reconstructs the sound speed distribution from
ultrasound data. To validate and benchmark the reconstruction, we need
**ground truth** -- a known velocity model to generate synthetic observed data.
Head phantoms provide this ground truth at varying levels of anatomical realism.

The skull is the critical challenge: cortical bone has $c \approx 2800$ m/s
(ITRUSST benchmark BM3), creating strong reflections, mode conversions, and
phase aberrations that make transcranial FWI much harder than breast USCT.

## Acoustic property tables

### ITRUSST benchmark values (Aubry et al. 2022)

Brain FWI uses the ITRUSST benchmark values from Aubry et al. (2022, JASA)
as the reference standard. These are the same values used in benchmarks BM1--BM9:

| Tissue | Label | $c$ (m/s) | $\rho$ (kg/m^3) | $\alpha$ (dB/cm/MHz) |
|--------|-------|-----------|-----------------|---------------------|
| Background (water) | 0 | 1500 | 1000 | 0.002 |
| CSF | 1 | 1500 | 1007 | 0.002 |
| Grey matter | 2 | 1560 | 1040 | 0.6 |
| White matter | 3 | 1560 | 1040 | 0.6 |
| Fat | 4 | 1478 | 950 | 0.4 |
| Muscle | 5 | 1547 | 1050 | 1.0 |
| Scalp | 6 | 1540 | 1090 | 0.8 |
| Skull (cortical) | 7 | 2800 | 1850 | 4.0 |
| Blood vessels | 8 | 1584 | 1060 | 0.2 |
| Connective tissue | 9 | 1520 | 1030 | 0.5 |
| Dura mater | 10 | 1560 | 1080 | 0.5 |
| Bone marrow (trabecular) | 11 | 2300 | 1700 | 8.0 |

The mapping is implemented in `brain_fwi.phantoms.properties`:

```python
from brain_fwi.phantoms.properties import TISSUE_PROPERTIES, map_labels_to_all

# Direct table access
c_skull, rho_skull, alpha_skull = TISSUE_PROPERTIES[7]
# (2800.0, 1850.0, 4.0)

# Vectorized mapping from label arrays
import jax.numpy as jnp
labels = jnp.array([[0, 1, 7], [2, 3, 11]])
props = map_labels_to_all(labels)
# props["sound_speed"].shape == (2, 3)
# props["sound_speed"][0, 2] == 2800.0  (skull)
```

## Phantom 1: BrainWeb (open access)

[BrainWeb](https://brainweb.bic.mni.mcgill.ca/) provides 20 normal anatomical
models as discrete tissue label volumes at 1 mm isotropic resolution
(181 x 217 x 181 voxels). The 12 tissue classes map directly to the ITRUSST
property table above.

### Loading a BrainWeb phantom

```python
from brain_fwi.phantoms.brainweb import load_brainweb_phantom, load_brainweb_slice

# Full 3D volume (downloads ~50 MB on first use)
labels_3d = load_brainweb_phantom(subject=4)
# labels_3d.shape == (181, 217, 181), dtype=int32

# Single 2D slice (faster for testing)
labels_2d, dx_mm = load_brainweb_slice(
    axis="axial",       # or "coronal", "sagittal"
    slice_idx=90,       # middle of brain
    subject=4,
    pad_to_square=True, # pad to 217x217 for j-Wave
)
# labels_2d.shape == (217, 217), dx_mm == 1.0
```

### Converting to acoustic properties

```python
from brain_fwi.phantoms.properties import map_labels_to_speed, map_labels_to_density

c_true = map_labels_to_speed(labels_2d)     # (217, 217) m/s
rho_true = map_labels_to_density(labels_2d) # (217, 217) kg/m^3
```

### BrainWeb limitations

- **Single skull layer**: BrainWeb has only one skull label (7 = cortical bone).
  Real skulls have three layers: outer table (cortical), diploe (trabecular),
  inner table (cortical). MIDA distinguishes these.
- **1 mm resolution**: Adequate for frequencies up to ~230 kHz. For the full
  300 kHz band, you need 0.5 mm (MIDA or interpolated BrainWeb).
- **No subject variability in skull**: All 20 BrainWeb subjects have similar
  skull geometry. Use MIDA for more realistic skull modelling.

## Phantom 2: MIDA (153 structures, 500 um)

The [MIDA model](https://itis.swiss/virtual-population/regional-human-models/mida-model/)
(Iacono et al. 2015) provides 153 anatomical structures at 500 um isotropic
resolution. It is the most detailed head model available for ultrasound simulation.

### MIDA tissue groups

Brain FWI groups the 153 MIDA labels into 17 acoustic categories:

```python
from brain_fwi.phantoms.mida import MIDA_TISSUE_GROUPS, MIDA_ACOUSTIC_PROPERTIES

# See all groups
for group, labels in MIDA_TISSUE_GROUPS.items():
    props = MIDA_ACOUSTIC_PROPERTIES[group]
    print(f"{group:20s}: {len(labels):3d} labels, "
          f"c={props['sound_speed']:.0f} m/s, "
          f"rho={props['density']:.0f} kg/m^3")
```

Key groups for transcranial FWI:

| Group | N labels | $c$ (m/s) | Notes |
|-------|---------|-----------|-------|
| cortical_bone | 11 | 2800 | Skull outer/inner table, teeth, vertebrae |
| trabecular_bone | 2 | 2300 | Diploe, bone marrow |
| grey_matter | 9 | 1560 | Cortex, cerebellum, thalamus |
| white_matter | 5 | 1560 | Cerebral/cerebellar WM, brainstem |
| csf | 3 | 1500 | CSF, ventricles, subarachnoid |

### Loading MIDA data

:::{admonition} MIDA license
:class: warning

The MIDA model requires a license from the IT'IS Foundation. It is not open
source. Contact: MIDAmodel@fda.hhs.gov. For open alternatives, use BrainWeb
or the synthetic phantom.
:::

```python
from brain_fwi.phantoms.mida import load_mida_volume, load_mida_acoustic
from pathlib import Path

# Load raw labels (supports .mat, .nii.gz, .h5)
labels = load_mida_volume(Path("MIDA_v1.0/MIDA_tissuedistrib.mat"))
# labels.shape ~ (480, 480, 350), dtype=int32

# Load and convert to acoustic properties in one step
props = load_mida_acoustic(
    Path("MIDA_v1.0/MIDA_tissuedistrib.mat"),
    target_shape=(240, 240, 175),  # downsample 2x for faster sim
)
c_mida = props["sound_speed"]      # (240, 240, 175)
rho_mida = props["density"]
labels_mida = props["labels"]
```

### Resampling

MIDA's native 500 um resolution may be too fine for initial experiments.
The `resample_volume` function handles this:

```python
from brain_fwi.phantoms.mida import resample_volume
import numpy as np

# Downsample labels (use order=0 = nearest neighbor for integer labels)
labels_small = resample_volume(labels, target_shape=(120, 120, 88), order=0)

# Downsample continuous fields (use order=1 = linear interpolation)
c_small = resample_volume(np.array(c_mida), target_shape=(120, 120, 88), order=1)
```

### MIDA vs. BrainWeb

| Feature | BrainWeb | MIDA |
|---------|---------|------|
| Resolution | 1 mm | 500 um |
| Tissue classes | 12 | 153 (grouped to 17) |
| Skull layers | 1 (cortical only) | 3 (outer, diploe, inner) |
| Subjects | 20 | 1 |
| License | Open | IT'IS (restricted) |
| File size | ~50 MB | ~500 MB |

## Phantom 3: Synthetic head (built-in)

For quick testing without downloading data, brain-fwi includes a procedural
synthetic head phantom generator:

```python
from brain_fwi.phantoms.brainweb import make_synthetic_head

labels, props = make_synthetic_head(
    grid_shape=(256, 256),
    dx=0.001,              # 1 mm
    skull_thickness=0.007, # 7 mm
    scalp_thickness=0.003, # 3 mm
    csf_thickness=0.002,   # 2 mm
)
# labels.shape == (256, 256), dtype=int32
# props keys: "sound_speed", "density", "attenuation"
```

### Anatomy of the synthetic phantom

The generator creates concentric elliptical layers modelling an adult head
cross-section:

1. **Background (label 0)**: Water coupling medium ($c = 1500$ m/s)
2. **Scalp (label 6)**: Outer soft tissue layer ($c = 1540$ m/s)
3. **Skull (label 7)**: Cortical bone ($c = 2800$ m/s)
4. **CSF (label 1)**: Cerebrospinal fluid gap ($c = 1500$ m/s)
5. **Grey matter (label 2)**: Cortical grey matter ($c = 1560$ m/s)
6. **White matter (label 3)**: Deep white matter core ($c = 1560$ m/s)

Additionally, two small elliptical **lateral ventricles** (CSF-filled, label 1)
are placed symmetrically inside the brain.

The head dimensions approximate an adult head: 19 cm AP x 15 cm LR.

### Validation tests

The synthetic phantom is tested for anatomical correctness:

```python
# Labels present
assert jnp.any(labels == 7)   # skull exists
assert jnp.any(labels == 2)   # grey matter
assert jnp.any(labels == 3)   # white matter
assert jnp.any(labels == 1)   # CSF (including ventricles)

# Physical properties in range
c = props["sound_speed"]
assert float(jnp.min(c)) >= 300.0
assert float(jnp.max(c)) <= 4500.0

# Skull faster than brain
skull_speed = float(jnp.mean(c[labels == 7]))
brain_speed = float(jnp.mean(c[labels == 2]))
assert skull_speed > brain_speed + 500
```

## Preparing a phantom for FWI

Regardless of the phantom source, the workflow for FWI validation is:

```python
import jax.numpy as jnp
from brain_fwi.phantoms.brainweb import make_synthetic_head
from brain_fwi.simulation.forward import generate_observed_data, build_domain, build_medium, build_time_axis
from brain_fwi.transducers.helmet import ring_array_2d, transducer_positions_to_grid
from brain_fwi.utils.wavelets import ricker_wavelet

# 1. Create or load the phantom
labels, props = make_synthetic_head(grid_shape=(128, 128), dx=0.001)
c_true = props["sound_speed"]
rho = props["density"]

# 2. Set up transducer array
dx = 0.001
center = (0.064, 0.064)
positions = ring_array_2d(n_elements=64, center=center,
                          semi_major=0.055, semi_minor=0.045)
sensor_grid = transducer_positions_to_grid(positions, dx, (128, 128))
src_list = [(int(sensor_grid[0][i]), int(sensor_grid[1][i]))
            for i in range(64)]

# 3. Compute time axis from c_max
domain = build_domain((128, 128), dx)
ref_medium = build_medium(domain, 3200.0, 1000.0, pml_size=20)
time_axis = build_time_axis(ref_medium, cfl=0.3, t_end=100e-6)
dt = float(time_axis.dt)
source_signal = ricker_wavelet(100e3, dt, int(100e-6 / dt))

# 4. Generate ground-truth data
observed = generate_observed_data(
    c_true, rho, dx, src_list, sensor_grid, 100e3,
    time_axis=time_axis, source_signal=source_signal, dt=dt,
)

# 5. Run FWI (see fwi_reconstruction tutorial)
```

## The SCI Institute label convention

Brain FWI also supports the SCI Institute head model format (6 labels) via a
remapping function:

```python
from brain_fwi.phantoms.properties import remap_sci_labels

# SCI labels: 0=background, 1=scalp, 2=skull, 3=CSF, 4=GM, 5=WM
sci_labels = ...  # load your SCI model
brainweb_labels = remap_sci_labels(sci_labels)
# Now use with map_labels_to_all() as usual
```

## Next steps

- {doc}`forward_simulation` -- Run simulations on your phantom
- {doc}`fwi_reconstruction` -- Reconstruct the velocity model via FWI

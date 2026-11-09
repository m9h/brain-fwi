"""Ultrasound transducer array geometry for brain FWI.

Generates transducer positions for:
  - 2D: Ring array (circle/ellipse around head cross-section)
  - 3D: Helmet array (geodesic sampling on scalp surface)

Inspired by:
  - Kernel Flow 2 (52 modules, whole-head coverage)
  - Guasch et al. (2020): 1024 transducers on spherical cap
  - Stride breast examples: 128 transducers on circle/ellipse

The ring_array_2d function places transducers on an ellipse around the
head with configurable standoff distance, mimicking a 2D cross-section
of the Kernel Flow helmet.
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional


def ring_array_2d(
    n_elements: int = 128,
    center: Tuple[float, float] = (0.0, 0.0),
    semi_major: float = 0.105,
    semi_minor: float = 0.085,
    standoff: float = 0.005,
    exclude_arc: Optional[Tuple[float, float]] = None,
) -> jnp.ndarray:
    """Generate a 2D elliptical ring transducer array.

    Args:
        n_elements: Number of transducer elements.
        center: (x, y) centre of the ellipse in metres.
        semi_major: Semi-major axis in metres (AP direction).
            Default 10.5 cm ≈ adult head radius + standoff.
        semi_minor: Semi-minor axis in metres (LR direction).
            Default 8.5 cm.
        standoff: Additional radial distance from scalp (m).
            Added to both semi-axes.
        exclude_arc: (start_angle, end_angle) in radians to exclude
            (e.g., for face opening). None = full ring.

    Returns:
        (n_elements, 2) array of transducer positions in metres.
    """
    a = semi_major + standoff
    b = semi_minor + standoff

    if exclude_arc is not None:
        start, end = exclude_arc
        # Generate angles excluding the arc
        total_arc = 2 * np.pi - (end - start)
        angles = jnp.linspace(end, end + total_arc, n_elements, endpoint=False)
    else:
        angles = jnp.linspace(0, 2 * np.pi, n_elements, endpoint=False)

    x = center[0] + a * jnp.cos(angles)
    y = center[1] + b * jnp.sin(angles)

    return jnp.stack([x, y], axis=-1)


def helmet_array_3d(
    n_elements: int = 256,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius_ap: float = 0.105,
    radius_lr: float = 0.085,
    radius_si: float = 0.095,
    standoff: float = 0.005,
    coverage_angle: float = 2.8,
    exclude_face: bool = True,
) -> jnp.ndarray:
    """Generate a 3D helmet transducer array.

    Distributes transducers on an ellipsoidal cap covering the head,
    inspired by the Kernel Flow 2 helmet layout. Uses Fibonacci sphere
    sampling for approximately uniform distribution.

    Args:
        n_elements: Number of transducer elements.
        center: (x, y, z) centre of the head in metres.
        radius_ap: Anterior-posterior semi-axis (m).
        radius_lr: Left-right semi-axis (m).
        radius_si: Superior-inferior semi-axis (m).
        standoff: Radial standoff from scalp (m).
        coverage_angle: Polar angle coverage from top (radians).
            Default 2.8 ≈ 160 deg (whole head except chin/neck).
        exclude_face: Remove elements in front of face
            (anterior, below equator).

    Returns:
        (n_elements, 3) array of transducer positions in metres.
    """
    a = radius_ap + standoff
    b = radius_lr + standoff
    c = radius_si + standoff

    # Fibonacci sphere sampling — nearly uniform point distribution
    # Oversample to account for exclusion zones
    n_oversample = int(n_elements * 2.5) if exclude_face else int(n_elements * 1.3)
    golden_ratio = (1 + np.sqrt(5)) / 2

    indices = np.arange(n_oversample)
    # Polar angle: from 0 (top) to coverage_angle
    theta = np.arccos(1 - 2 * indices / n_oversample)
    # Azimuthal angle: golden ratio spacing
    phi = 2 * np.pi * indices / golden_ratio

    # Filter by coverage
    mask = theta <= coverage_angle

    # Exclude face region: anterior (positive x), below equator (negative z)
    if exclude_face:
        # Convert to Cartesian to test
        x_test = np.sin(theta) * np.cos(phi)
        z_test = np.cos(theta)
        # Exclude: x > 0.3 and z < 0 (front-lower quadrant)
        face_mask = ~((x_test > 0.5) & (z_test < -0.2))
        mask = mask & face_mask

    theta = theta[mask]
    phi = phi[mask]

    # Subsample to target count
    if len(theta) > n_elements:
        rng = np.random.default_rng(42)
        idx = _farthest_point_subsample_sphere(theta, phi, n_elements, rng)
        theta = theta[idx]
        phi = phi[idx]

    # Map to ellipsoid
    x = center[0] + a * np.sin(theta) * np.cos(phi)
    y = center[1] + b * np.sin(theta) * np.sin(phi)
    z = center[2] + c * np.cos(theta)

    return jnp.array(np.column_stack([x, y, z]))


def clinical_helmet_3d(
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius_ap: float = 0.095,
    radius_lr: float = 0.080,
    radius_si: float = 0.100,
    freq: float = 300e3,
    standoff: float = 0.007,
    polar_max_deg: float = 158.0,
    n_elements: Optional[int] = None,
    c_water: float = 1500.0,
    scalp_mask: Optional[np.ndarray] = None,
    dx: Optional[float] = None,
) -> jnp.ndarray:
    """Clinically-realistic imaging helmet (Imperial/Guasch target).

    Improves on :func:`helmet_array_3d` in three ways grounded in the real
    devices (see ``docs/design/helmet_array_plan.md``):

    1. **Frequency-aware density** — element count from ~λ/2 sampling at
       ``freq`` (anchored to a realistic sub-Nyquist imaging density, ~512 at
       300 kHz, scaling as f²), instead of a fixed 256.
    2. **Full azimuth, face *aperture* only** — keeps 360° coverage of the
       cranial vault (vertex → ``polar_max_deg``), removing only the
       anterior-inferior face cone (eyes/airway), not the whole anterior.
    3. **Scalp-conformal** — when a ``scalp_mask`` (+ ``dx``) is given, each
       element is projected onto the patient's actual scalp surface plus
       ``standoff``, hugging the real head instead of a generic ellipsoid.

    Args:
        center: head centre in metres.
        radius_ap/lr/si: skull semi-axes (m); the ellipsoid fallback shape.
        freq: top operating frequency (Hz) → element spacing.
        standoff: water/gel standoff from scalp to element (m).
        polar_max_deg: polar coverage from vertex (0°); 140° ≈ down to the ears.
        n_elements: override the frequency-derived count.
        scalp_mask: optional (nx,ny,nz) bool head mask for conformal projection.
        dx: grid spacing (m), required with ``scalp_mask``.

    Returns:
        (n_elements, 3) array of element positions in metres.
    """
    if n_elements is None:
        n_elements = int(np.clip(512 * (freq / 300e3) ** 2, 64, 2048))
    a, b, cc = radius_ap + standoff, radius_lr + standoff, radius_si + standoff

    n_over = int(n_elements * 3)
    golden = (1 + np.sqrt(5)) / 2
    idx = np.arange(n_over)
    theta = np.arccos(1 - 2 * idx / n_over)            # 0 = vertex
    phi = 2 * np.pi * idx / golden
    ux = np.sin(theta) * np.cos(phi)                   # anterior (+x)
    uy = np.sin(theta) * np.sin(phi)                   # lateral
    uz = np.cos(theta)                                 # superior (+z)

    keep = theta <= np.radians(polar_max_deg)
    keep &= ~((ux > 0.5) & (uz < -0.15))               # remove face aperture only
    theta, phi, ux, uy, uz = (v[keep] for v in (theta, phi, ux, uy, uz))

    if len(theta) > n_elements:
        sel = _farthest_point_subsample_sphere(
            theta, phi, n_elements, np.random.default_rng(42))
        ux, uy, uz = ux[sel], uy[sel], uz[sel]

    dirs = np.column_stack([ux, uy, uz])
    if scalp_mask is not None and dx is not None:
        pos = _project_to_scalp(dirs, np.asarray(center), np.asarray(scalp_mask), dx, standoff)
    else:
        pos = np.column_stack([center[0] + a * ux, center[1] + b * uy, center[2] + cc * uz])
    return jnp.array(pos)


def _project_to_scalp(dirs, center, scalp_mask, dx, standoff):
    """March each unit direction out from the head centre, find the outermost
    scalp voxel, and place the element ``standoff`` beyond it."""
    N = np.asarray(scalp_mask.shape)
    max_steps = int(np.max(N))
    out = np.zeros((len(dirs), 3))
    for i, d in enumerate(dirs):
        d = d / (np.linalg.norm(d) + 1e-12)
        last_r = 0.0
        for s in range(max_steps):
            ijk = np.round((center + d * (s * dx)) / dx).astype(int)
            if np.any(ijk < 0) or np.any(ijk >= N):
                break
            if scalp_mask[ijk[0], ijk[1], ijk[2]]:
                last_r = s * dx
        out[i] = center + d * (last_r + standoff)
    return out


def _farthest_point_subsample_sphere(
    theta: np.ndarray, phi: np.ndarray, target: int, rng
) -> np.ndarray:
    """Farthest-point subsampling on the sphere."""
    n = len(theta)
    # Convert to unit sphere Cartesian for distance computation
    xyz = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])

    selected = [rng.integers(n)]
    dists = np.full(n, np.inf)

    for _ in range(target - 1):
        d = np.linalg.norm(xyz - xyz[selected[-1]], axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))

    return np.array(selected)


def transducer_positions_to_grid(
    positions: jnp.ndarray,
    dx: float,
    grid_shape: Tuple[int, ...],
    grid_origin: Optional[jnp.ndarray] = None,
) -> Tuple:
    """Convert physical transducer positions to grid indices.

    Args:
        positions: (n_elements, ndim) physical coordinates in metres.
        dx: Grid spacing in metres.
        grid_shape: Grid dimensions.
        grid_origin: Physical coordinates of grid origin.
            Defaults to (0, 0, ...).

    Returns:
        Tuple of index arrays, one per dimension. Each is (n_elements,)
        of int32, clipped to valid grid range.
    """
    ndim = positions.shape[1]
    if grid_origin is None:
        grid_origin = jnp.zeros(ndim)

    indices = jnp.round((positions - grid_origin) / dx).astype(jnp.int32)

    # Clip to valid range
    for d in range(ndim):
        indices = indices.at[:, d].set(
            jnp.clip(indices[:, d], 0, grid_shape[d] - 1)
        )

    return tuple(indices[:, d] for d in range(ndim))


def compute_normals_2d(positions: jnp.ndarray, center: jnp.ndarray) -> jnp.ndarray:
    """Compute inward-pointing normals for a 2D ring array.

    Args:
        positions: (n_elements, 2) transducer positions.
        center: (2,) head centre.

    Returns:
        (n_elements, 2) unit normals pointing toward centre.
    """
    diff = center - positions
    norms = jnp.linalg.norm(diff, axis=1, keepdims=True)
    return diff / jnp.maximum(norms, 1e-10)

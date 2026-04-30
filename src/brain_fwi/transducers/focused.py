"""Focused ultrasound transducer geometry for TUS.

Provides:
  - bowl_transducer_3d: Transducers on a spherical cap (bowl)
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple


def bowl_transducer_3d(
    focal_length: float,
    aperture_diameter: float,
    focal_point: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    n_points: int = 1000,
) -> jnp.ndarray:
    """Generate points on a spherical bowl focused at focal_point.

    Args:
        focal_length: Distance from bowl center to focal point (m).
        aperture_diameter: Diameter of the bowl opening (m).
        focal_point: (x, y, z) target focus in metres.
        direction: (dx, dy, dz) vector pointing from bowl center to focus.
        n_points: Number of points to sample on the bowl surface.

    Returns:
        (n_points, 3) array of transducer positions in metres.
    """
    # Unit direction vector
    dir_vec = np.array(direction) / np.linalg.norm(direction)
    
    # Bowl center in physical coordinates
    bowl_center = np.array(focal_point) - dir_vec * focal_length
    
    # Semi-angle of the bowl
    half_angle = np.arcsin(aperture_diameter / (2 * focal_length))
    
    # Sample points on a spherical cap using Fibonacci sampling
    indices = np.arange(n_points)
    phi = np.arccos(1 - (1 - np.cos(half_angle)) * (indices / n_points))
    theta = 2 * np.pi * indices * ((1 + np.sqrt(5)) / 2)
    
    # Points on unit sphere centered at origin, pointing along +z
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    # Scale to focal length
    points = np.column_stack([x, y, z]) * focal_length
    
    # Rotate to match direction (align +z with dir_vec)
    z_axis = np.array([0, 0, 1])
    if not np.allclose(dir_vec, z_axis):
        if np.allclose(dir_vec, -z_axis):
            points[:, 2] *= -1
        else:
            v = np.cross(z_axis, dir_vec)
            s = np.linalg.norm(v)
            c = np.dot(z_axis, dir_vec)
            v_skew = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R = np.eye(3) + v_skew + np.matmul(v_skew, v_skew) * ((1 - c) / (s**2))
            points = np.dot(points, R.T)
            
    final_points = points + bowl_center
    
    return jnp.array(final_points)

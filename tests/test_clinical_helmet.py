"""clinical_helmet_3d: a full-azimuth imaging helmet (Guasch/Imperial target)
vs the current Kernel-Flow cap.

The cap (helmet_array_3d, exclude_face=True) drops the whole anterior hemisphere,
which kills front<->back transmission paths FWI needs. The clinical helmet keeps
full 360 deg azimuth and removes only a small face aperture (eyes/airway), with
element count tied to lambda/2 at the operating frequency.
"""
from __future__ import annotations
import numpy as np


def _angles(pos, center):
    d = np.asarray(pos) - np.asarray(center)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    return d  # unit directions (x=anterior, y=lateral, z=superior)


def test_clinical_helmet_full_azimuth_and_density():
    from brain_fwi.transducers.helmet import clinical_helmet_3d, helmet_array_3d
    center = (0.0, 0.0, 0.0)
    pos = np.asarray(clinical_helmet_3d(
        center=center, radius_ap=0.095, radius_lr=0.08, radius_si=0.10,
        freq=300e3, standoff=0.007))
    d = _angles(pos, center)

    # (1) element count tied to lambda/2 -> many more than the 256 cap
    assert len(pos) >= 400, f"expected a dense array, got {len(pos)}"

    # (2) FULL azimuth at the upper head: anterior AND posterior elements present
    upper = d[:, 2] > 0.2
    assert (d[upper, 0] > 0.3).any(), "no anterior (front) elements — not full-azimuth"
    assert (d[upper, 0] < -0.3).any(), "no posterior (back) elements"
    assert (d[upper, 1] > 0.3).any() and (d[upper, 1] < -0.3).any(), "missing lateral coverage"

    # (3) face aperture: no elements in the anterior-INFERIOR cone (eyes/airway)
    face = (d[:, 0] > 0.6) & (d[:, 2] < -0.3)
    assert not face.any(), "face aperture not removed (elements over eyes/airway)"

    # (4) denser anterior sampling than the fixed-256 cap (more elements over the
    # front of the head -> finer transmission sampling)
    cap = np.asarray(helmet_array_3d(n_elements=256, center=center,
                                     radius_ap=0.095, radius_lr=0.08, radius_si=0.10,
                                     exclude_face=True))
    dc = _angles(cap, center)
    clinical_ant = ((d[:, 0] > 0.3) & (d[:, 2] > 0.0)).sum()
    cap_ant = ((dc[:, 0] > 0.3) & (dc[:, 2] > 0.0)).sum()
    assert clinical_ant > cap_ant, "clinical helmet should sample the front more densely than the cap"


def test_clinical_helmet_conformal_to_scalp():
    """With a scalp mask, elements project onto the real head surface + standoff
    (the key upgrade over a generic ellipsoid) — every element lands just outside
    the scalp, none inside it."""
    from brain_fwi.transducers.helmet import clinical_helmet_3d
    N, dx = 64, 3e-3
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
    c = np.array([N / 2] * 3)
    r = np.sqrt(((xx - c[0]) * dx) ** 2 + ((yy - c[1]) * dx) ** 2 + ((zz - c[2]) * dx) ** 2)
    scalp = r < 0.085  # 85 mm sphere "head"
    pos = np.asarray(clinical_helmet_3d(
        center=tuple(c * dx), radius_ap=0.09, radius_lr=0.09, radius_si=0.09,
        freq=250e3, standoff=0.006, scalp_mask=scalp, dx=dx))
    rad = np.linalg.norm(pos - c * dx, axis=1)
    assert (rad > 0.085).all(), "some elements are inside the scalp"
    assert (rad < 0.085 + 0.02).all(), "elements too far from scalp (not conformal)"


def test_clinical_helmet_element_count_scales_with_frequency():
    from brain_fwi.transducers.helmet import clinical_helmet_3d
    kw = dict(center=(0.0, 0.0, 0.0), radius_ap=0.095, radius_lr=0.08, radius_si=0.10)
    n_low = len(np.asarray(clinical_helmet_3d(freq=150e3, **kw)))
    n_high = len(np.asarray(clinical_helmet_3d(freq=400e3, **kw)))
    # lambda/2 sampling -> higher frequency packs more elements
    assert n_high > n_low, f"element count should grow with frequency ({n_high} !> {n_low})"

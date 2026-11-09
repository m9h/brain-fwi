"""Birnbaum stroke-patient head loader (arXiv:2501.18716).

64 anonymized 1 mm head segmentations with paired T1, at
``/data/datasets/birnbaum/Data/Anonymized_Subjects/``. Label scheme (decoded
geometrically + visually): 0=background, 1=lesion (stroke), 2/3/4=brain tissue,
5=skull (bone), 6=scalp.

Provides label->acoustic-velocity mapping, a cerebrum-only ROI (largest brain
connected component, cropped+centred to exclude face/skull-base/sinuses), and
dataset builders for training anatomy priors: 2D slices
(:func:`build_slice_dataset`, see :mod:`brain_fwi.inference.score_unet`) and 3D
cerebrum volumes (:func:`build_volume_dataset`, see
:mod:`brain_fwi.inference.score_unet3d`). In 3D the brain+lesion labels form one
connected component cleanly separated from the face by the skull, so a 3D
largest-CC isolates the cerebrum — the 2D face-entanglement blocker is gone.
See ``docs/design/diffusion_prior_fwi.md``.
"""
from pathlib import Path
import glob
import numpy as np
from scipy.ndimage import label as _cc, zoom

BIRNBAUM_ROOT = "/data/datasets/birnbaum/Data/Anonymized_Subjects"
LESION, SKULL = 1, 5
BRAIN = (2, 3, 4)
# velocity (m/s): water/CSF/scalp 1500, brain 1560, lesion 1660 (demo contrast), skull 2800
C_WATER, C_BRAIN, C_LESION, C_SKULL = 1500.0, 1560.0, 1660.0, 2800.0


def label_files(root: str = BIRNBAUM_ROOT) -> list[str]:
    return sorted(glob.glob(f"{root}/Full-Head Segmentation/*_label_deface.nii"))


def largest_cc(mask: np.ndarray) -> np.ndarray:
    lab, n = _cc(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == int(sizes.argmax())


def cerebrum_crop(lab2d: np.ndarray, S: int = 64, margin: int = 8):
    """Crop to the cerebrum (largest brain+lesion CC) + margin, resize to S x S
    (nearest). Returns the cropped label image, or None if no brain. Note: in a
    2D axial slice the brain can connect to the face through the skull-base, so
    this does NOT always fully isolate the cerebrum — 3D is the clean fix."""
    brain = np.isin(lab2d, BRAIN) | (lab2d == LESION)
    roi = largest_cc(brain)
    if roi.sum() < 80:
        return None
    ys, xs = np.where(roi)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, lab2d.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, lab2d.shape[1])
    crop = lab2d[y0:y1, x0:x1]
    return zoom(crop, (S / crop.shape[0], S / crop.shape[1]), order=0)


def roi_mask(crop: np.ndarray) -> np.ndarray:
    """Cerebrum CC within a crop — the region FWI updates / scores / scores against."""
    return largest_cc(np.isin(crop, BRAIN) | (crop == LESION))


def to_velocity(crop: np.ndarray, with_skull: bool = True) -> np.ndarray:
    v = np.full_like(crop, C_WATER, dtype=np.float32)
    v[np.isin(crop, BRAIN)] = C_BRAIN
    v[crop == LESION] = C_LESION
    if with_skull:
        v[crop == SKULL] = C_SKULL
    return v


def prior_image(crop: np.ndarray) -> np.ndarray:
    """Prior-training image: cerebrum velocity on a water frame (no skull, no face)."""
    return np.where(roi_mask(crop), to_velocity(crop, with_skull=False), C_WATER).astype(np.float32)


def build_slice_dataset(files: list[str], S: int = 64, per_subject_brain: int = 4):
    """Flattened (N, S*S) cerebrum-ROI prior images across lesion + brain slices."""
    import nibabel as nib
    imgs = []
    for f in files:
        lab = np.asarray(nib.load(f).dataobj).astype(np.int16)
        lz = np.array([(lab[z] == LESION).sum() for z in range(lab.shape[0])])
        bz = np.array([np.isin(lab[z], BRAIN).sum() for z in range(lab.shape[0])])
        sel = [z for z in np.argsort(lz)[-4:] if lz[z] > 30]
        br = np.where(bz > bz.max() * 0.4)[0]
        if len(br):
            sel += list(br[np.linspace(0, len(br) - 1, per_subject_brain).astype(int)])
        for z in set(sel):
            c = cerebrum_crop(lab[z], S=S)
            if c is not None:
                imgs.append(prior_image(c).reshape(-1))
    return np.stack(imgs).astype(np.float32)


def cerebrum_volume_crop(lab3d: np.ndarray, S: int = 48, margin: int = 5):
    """Crop a head label volume to its cerebrum (3D largest brain+lesion CC) +
    margin and resize to ``S^3`` (nearest). Returns the cropped label volume, or
    None if too little brain. Unlike the 2D :func:`cerebrum_crop`, the 3D
    connected component cleanly excludes the face/skull (no entanglement)."""
    roi = largest_cc(np.isin(lab3d, BRAIN) | (lab3d == LESION))
    if roi.sum() < 5000:
        return None
    zs, ys, xs = np.where(roi)
    sl = tuple(
        slice(max(a.min() - margin, 0), min(a.max() + margin + 1, lab3d.shape[i]))
        for i, a in enumerate((zs, ys, xs))
    )
    crop = lab3d[sl]
    return zoom(crop, tuple(S / crop.shape[i] for i in range(3)), order=0)


def prior_volume(crop3d: np.ndarray) -> np.ndarray:
    """3D prior-training volume: cerebrum velocity on a water frame (no skull/face)."""
    return np.where(roi_mask(crop3d), to_velocity(crop3d, with_skull=False), C_WATER).astype(np.float32)


def inject_synthetic_lesions(vol: np.ndarray, roi: np.ndarray, rng,
                             n_range=(1, 3), radius_range=(2, 6), vel_range=(1610.0, 1710.0)):
    """Inject random spherical high-velocity lesions into a cerebrum volume (within
    roi). Makes the diffusion prior LESION-AWARE: trained only on the real (rare,
    location-specific) Birnbaum lesions, the prior smooths focal blobs as outliers
    (the DPS finding); injecting diverse synthetic lesions teaches it focal
    high-velocity blobs are plausible anywhere. Returns a new volume."""
    out = vol.copy()
    idx = np.argwhere(roi)
    if len(idx) == 0:
        return out
    zz, yy, xx = np.indices(vol.shape)
    for _ in range(int(rng.integers(n_range[0], n_range[1] + 1))):
        cz, cy, cx = idx[rng.integers(len(idx))]
        r = float(rng.integers(radius_range[0], radius_range[1] + 1))
        v = float(rng.uniform(*vel_range))
        sph = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
        out[sph & roi] = v
    return out


def build_volume_dataset(files: list[str], S: int = 48, flip_augment: bool = True,
                         lesion_aug: int = 0, seed: int = 0):
    """Flattened (N, S^3) cerebrum-volume prior images, one (+ L-R flip) per head.
    With ``lesion_aug>0``, add that many synthetic-lesion-injected copies per head
    (a lesion-aware prior). Trains :mod:`brain_fwi.inference.score_unet3d`."""
    import nibabel as nib
    rng = np.random.default_rng(seed)
    vols = []
    for f in files:
        lab = np.asarray(nib.load(f).dataobj).astype(np.int16)
        c = cerebrum_volume_crop(lab, S=S)
        if c is None:
            continue
        v = prior_volume(c); roi = roi_mask(c)
        bases = [(v, roi)] + ([(v[:, :, ::-1].copy(), roi[:, :, ::-1])] if flip_augment else [])
        for b, broi in bases:
            vols.append(b.reshape(-1))
            for _ in range(lesion_aug):
                vols.append(inject_synthetic_lesions(b, broi, rng).reshape(-1))
    return np.stack(vols).astype(np.float32)

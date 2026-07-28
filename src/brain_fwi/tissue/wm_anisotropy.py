"""White-matter attenuation-anisotropy translation, grounded in real-tissue data.

There is no direct measurement of direction-dependent ultrasound attenuation in
brain white matter (confirmed by literature review, mid-2026). This module
brackets its magnitude from the strongest available real-tissue evidence:

  * **Skeletal muscle** -- the fibrous analog where longitudinal ultrasound
    attenuation anisotropy IS directly measured. Nassiri, Nicholas & Hill,
    *Ultrasonics* 1979: alpha_parallel = 2.9 +/- 0.23 vs alpha_perp = 1.1 +/- 0.15
    dB/cm/MHz -> ratio ~2.6. Topp & O'Brien, *JASA* 2000: attenuation ~2x lower
    perpendicular. (Note: in muscle attenuation is HIGHER *along* the fibre.)
  * **White-matter MRE** -- anisotropic *shear damping* directly measured in real
    WM. Anderson et al. 2018 (porcine, ex-vivo): shear-modulus/loss anisotropy
    phi = 0.27-0.34, i.e. a loss-anisotropy ratio ~1.3.

Physical basis: WM is a transversely-isotropic viscoelastic medium (aligned
myelinated axons). A compressional wave's attenuation interpolates with the angle
theta to the fibre axis between alpha_par (along) and alpha_perp (across). The
same aligned microstructure produces diffusion anisotropy, so diffusion FA is the
per-voxel proxy that scales the (unknown) acoustic anisotropy magnitude.

CAVEAT baked into every consumer: the anisotropy ratio here is *borrowed* from
muscle / WM-shear-MRE, not measured in WM acoustically. Treat it as a tunable,
bracketed parameter (1.0 isotropic control -> ~2.0 muscle-like), never a claim.
"""
from __future__ import annotations

import numpy as np

# --- real-tissue anchor values (measured, cited) ---
MUSCLE_ALPHA_PAR = 2.9   # dB/cm/MHz, along fibre (Nassiri 1979)
MUSCLE_ALPHA_PERP = 1.1  # dB/cm/MHz, across fibre (Nassiri 1979)
MRE_WM_LOSS_ANISOTROPY = 0.30  # phi = mu1/mu2 - 1, real WM (Anderson 2018)


def predicted_alpha_profile(theta, a_par, a_perp):
    """Compressional attenuation vs angle theta from the fibre axis (radians).

    Transversely-isotropic interpolation: a_par at theta=0 (along fibre),
    a_perp at theta=pi/2 (across). This is exactly the plane-wave loss the
    validated j-Wave anisotropic absorber produces (k^T D k / |k|^2).
    """
    theta = np.asarray(theta, dtype=float)
    return a_par * np.cos(theta) ** 2 + a_perp * np.sin(theta) ** 2


def muscle_reference():
    """The calibration anchor: real skeletal-muscle longitudinal attenuation
    anisotropy (Nassiri 1979). Returns the measured endpoints and ratio."""
    return {
        "alpha_par": MUSCLE_ALPHA_PAR,
        "alpha_perp": MUSCLE_ALPHA_PERP,
        "ratio": MUSCLE_ALPHA_PAR / MUSCLE_ALPHA_PERP,  # ~2.64
        "units": "dB/cm/MHz",
        "source": "Nassiri, Nicholas & Hill, Ultrasonics 1979",
    }


def wm_ratio_bracket():
    """Bracket for the WM compressional attenuation-anisotropy ratio
    (max/min over angle), from real-tissue evidence.

    Returns (conservative, central, optimistic):
      * conservative ~1.3  -- WM-shear-MRE loss anisotropy analog (Anderson 2018);
        assumes the compressional loss channel is only as anisotropic as the
        measured shear loss channel.
      * optimistic ~2.0    -- muscle-analog, discounted below full muscle (~2.6)
        because WM axons are finer, less densely packed and more dispersed than
        skeletal-muscle fibres.
      * central ~1.6       -- geometric mean of the two.
    """
    lo = 1.0 + MRE_WM_LOSS_ANISOTROPY           # ~1.30
    hi = 2.0                                     # muscle-analog, discounted
    mid = float(np.sqrt(lo * hi))               # ~1.61
    return (float(lo), mid, float(hi))


def ratio_from_fa(fa, ratio_max=2.0):
    """Per-voxel acoustic anisotropy ratio from diffusion FA (first-order model).

    Anchored at isotropy: FA=0 -> ratio 1 (no anisotropy). Scales linearly to the
    tissue-max ratio at full alignment (FA=1). This is the honest coupling
    assumption -- attenuation anisotropy proportional to microstructural
    anisotropy -- calibrated so a fully-aligned voxel reaches ``ratio_max``.
    """
    fa = np.asarray(fa, dtype=float)
    return 1.0 + np.clip(fa, 0.0, 1.0) * (ratio_max - 1.0)


def wm_acoustic_estimate(fa_median, alpha_iso, ratio_max=2.0):
    """Combine a real-tissue WM FA and a bulk isotropic attenuation into the
    predicted (a_par, a_perp) endpoints for that WM, preserving the angle-mean.

    alpha_iso is the measured bulk WM attenuation (e.g. ~5-6 dB/cm/MHz range).
    The angle-averaged attenuation is held at alpha_iso; the ratio comes from
    ``ratio_from_fa``. Sign convention follows muscle: more attenuation ALONG
    the fibre (a_par > a_perp).
    """
    r = float(ratio_from_fa(fa_median, ratio_max))
    # mean over angle of (a_par cos^2 + a_perp sin^2) = 0.5(a_par + a_perp) = alpha_iso
    # with a_par/a_perp = r  ->  a_perp = 2 alpha_iso/(1+r), a_par = r*a_perp
    a_perp = 2.0 * alpha_iso / (1.0 + r)
    a_par = r * a_perp
    return {"a_par": a_par, "a_perp": a_perp, "ratio": r, "alpha_iso": alpha_iso}

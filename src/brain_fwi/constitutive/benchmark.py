"""Tissue-fit benchmark for AttenuationCANN.

Takes the ITRUSST scalar attenuation table from ``phantoms.mida`` and
the literature power-law exponents per tissue (Szabo 2014; Pinton et
al. 2012 for skull), generates ground-truth α(ω) curves over the FWI
band, and fits one :class:`AttenuationCANN` per tissue.

Output is a dict ``{tissue_name: TissueFitResult}`` with the recovered
fit's relative RMSE plus the K-K-implied phase-velocity dispersion
across the band — the latter quantifies how much physics the current
constant-c FWI sim is missing per tissue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from brain_fwi.phantoms.mida import MIDA_ACOUSTIC_PROPERTIES

from .cann import AttenuationCANN
from .kk import kramers_kronig_dispersion

# Power-law exponents from literature. Szabo, Diagnostic Ultrasound
# Imaging, table 4.1; Pinton et al. 2012 for skull.
TISSUE_POWER_LAW_Y: Dict[str, float] = {
    "water":           1.0,
    "air":             1.0,
    "skin":            1.0,
    "fat":             1.0,
    "muscle":          1.1,
    "cortical_bone":   1.5,
    "trabecular_bone": 1.5,
    "cartilage":       1.0,
    "csf":             1.0,
    "grey_matter":     1.3,
    "white_matter":    1.3,
    "blood_vessels":   1.0,
    "dura":            1.0,
    "eye":             1.0,
}

DB_PER_CM_TO_NP_PER_M = 100.0 / 8.685889638  # 1 dB = 1/8.686 Np; cm → m
F_REF_HZ = 1.0e6


@dataclass(frozen=True)
class TissueFitResult:
    """Per-tissue benchmark output."""
    tissue: str
    alpha0_np_per_m: float          # α₀ = α at 1 MHz, Np/m
    y_exponent: float               # power-law exponent
    rel_rmse: float                 # ‖α_pred − α_true‖₂ / ‖α_true‖₂ on the band
    recon_rms_np_per_m: float       # absolute RMS (handy for α≈0 tissues)
    delta_c_top_freq_m_per_s: float # K-K c(ω_max) − c_ref


def alpha_curve(
    alpha0_np_per_m: float,
    y_exponent: float,
    omega: jnp.ndarray,
    f_ref_hz: float = F_REF_HZ,
) -> jnp.ndarray:
    """Power-law α(ω) = α₀ · |f/f_ref|^y, in Np/m."""
    f_hz = jnp.abs(omega) / (2.0 * jnp.pi)
    return alpha0_np_per_m * (f_hz / f_ref_hz) ** y_exponent


def db_per_cm_per_mhz_to_np_per_m_at_1mhz(alpha_db_cm_mhz: float) -> float:
    """Convert ITRUSST table units to α₀ in Np/m at 1 MHz."""
    return alpha_db_cm_mhz * DB_PER_CM_TO_NP_PER_M


def _fit_one_tissue(
    alpha_true: jnp.ndarray,
    omega: jnp.ndarray,
    n_steps: int,
    key: jax.Array,
) -> AttenuationCANN:
    alpha_max = float(jnp.max(alpha_true))
    # Use 1.0 as a fallback alpha_scale when α≡0 (water/air) so the
    # gradient is well-conditioned and weights collapse to zero.
    scale = alpha_max if alpha_max > 0.0 else 1.0
    model = AttenuationCANN(
        n_basis=2, key=key,
        omega_scale=float(omega.max()),
        alpha_scale=scale,
    )
    opt = optax.adam(1e-2)
    state = opt.init(model)

    @eqx.filter_jit
    def step(m, st):
        def loss_fn(mm):
            return jnp.mean((mm(omega) - alpha_true) ** 2)
        loss, g = eqx.filter_value_and_grad(loss_fn)(m)
        upd, st = opt.update(g, st)
        m = eqx.apply_updates(m, upd)
        return m, st, loss

    for _ in range(n_steps):
        model, state, _ = step(model, state)

    return model


def fit_tissue_alpha_curves(
    *,
    omega: jnp.ndarray,
    tissues: Optional[Iterable[str]] = None,
    n_steps: int = 2000,
    key: jax.Array,
    c_ref: float = 1500.0,
) -> Dict[str, TissueFitResult]:
    """Fit a per-tissue CANN on the ITRUSST α table over ``omega``.

    Args:
        omega: angular frequencies (rad/s) — typically the FWI band.
        tissues: subset of tissue names; defaults to all in ``MIDA_ACOUSTIC_PROPERTIES``.
        n_steps: SGD steps per tissue.
        key: PRNG key (split internally).
        c_ref: reference phase velocity at ``omega.min()`` for the K-K
            dispersion diagnostic.

    Returns:
        ``{tissue: TissueFitResult}``.
    """
    selected = list(tissues) if tissues is not None else list(MIDA_ACOUSTIC_PROPERTIES)
    keys = jr.split(key, len(selected))
    results: Dict[str, TissueFitResult] = {}

    for k, tissue in zip(keys, selected):
        props = MIDA_ACOUSTIC_PROPERTIES[tissue]
        alpha0 = db_per_cm_per_mhz_to_np_per_m_at_1mhz(props["attenuation"])
        y = TISSUE_POWER_LAW_Y.get(tissue, 1.0)
        alpha_true = alpha_curve(alpha0, y, omega)

        model = _fit_one_tissue(alpha_true, omega, n_steps, k)
        alpha_pred = model(omega)
        diff = alpha_pred - alpha_true
        rms = float(jnp.sqrt(jnp.mean(diff ** 2)))
        denom = float(jnp.sqrt(jnp.mean(alpha_true ** 2)))
        rel = rms / denom if denom > 0 else 0.0

        c_kk = kramers_kronig_dispersion(
            alpha_true, omega, omega_ref=float(omega[0]), c_ref=c_ref,
        )
        delta_c_top = float(c_kk[-1] - c_ref)

        results[tissue] = TissueFitResult(
            tissue=tissue,
            alpha0_np_per_m=alpha0,
            y_exponent=y,
            rel_rmse=rel,
            recon_rms_np_per_m=rms,
            delta_c_top_freq_m_per_s=delta_c_top,
        )

    return results

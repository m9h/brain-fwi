"""Production-scale regression tests for the Phase-0 augmentation pipeline.

Captures bug classes that 16^3 unit tests miss:

* Bug #2 (`jittered_properties` hard-clipped MIDA labels >=12 to trabecular).
  Caught here by asserting that on a real MIDA volume the c-field has the
  expected water-coupling fraction and cortical-bone presence — both close
  to zero with the bug, both healthy after the fix.

* Bug #3 (`random_deformation_warp` with min(grid)/6 smoothness becomes a
  NN-rounding no-op at 96^3). Caught here by asserting that the warp
  produces non-trivial label changes at the production grid and that
  different seeds produce different warped outputs.

Both bugs would have shipped 1024-sample datasets that "ran clean" but
were silently useless; these tests turn that mode of failure into a CI
red light at sample #1.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from brain_fwi.phantoms.augment import random_deformation_warp
from brain_fwi.phantoms.mida import (
    MIDA_INTERNAL_AIR_LABELS,
    make_mida_phantom,
    mida_jittered_properties,
)


_MIDA_CANDIDATES = (
    Path("/Users/mhough/Workspace/brain-fwi/data/MIDA_v1.0/MIDA_v1_voxels/MIDA_v1.nii"),
    Path("/data/datasets/MIDAv1-0/MIDA_v1.0/MIDA_v1_voxels/MIDA_v1.nii"),
)


def _find_mida() -> Path | None:
    for p in _MIDA_CANDIDATES:
        if p.exists():
            return p
    return None


def _synthetic_96_phantom() -> np.ndarray:
    """A 96^3 label volume with enough non-trivial structure to exercise
    the warp at production scale, without needing the MIDA NIfTI.

    A spherical "skull" shell around a smaller "brain" core, with a few
    scattered point labels to make label-boundary crossings detectable.
    """
    n = 96
    vol = np.zeros((n, n, n), dtype=np.int32)
    z, y, x = np.mgrid[0:n, 0:n, 0:n]
    cx = cy = cz = n / 2
    r = np.sqrt((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2)
    vol[r < 38] = 50    # background-as-water region inside outer shell
    vol[r < 35] = 7     # skull-like shell (label 7)
    vol[r < 32] = 2     # grey matter
    vol[r < 22] = 12    # white matter (label >=12 — the v2 bug class)
    vol[r < 10] = 6     # CSF
    return vol


# ---------------------------------------------------------------------------
# Warp at production scale (regression for bug #3)
# ---------------------------------------------------------------------------


class TestWarpAtProductionScale:
    """Catches the no-op warp regression at 96^3 — the grid where it bit.

    The label-change tests need real anatomical detail to see whether the
    warp crosses label boundaries; concentric-sphere synthetic phantoms
    have thick uniform regions that a working warp may legitimately
    leave unchanged. So those tests gate on real MIDA. The displacement-
    field test below operates on the raw warp logic and works without any
    label volume — it's the cheapest unit-level guard against the no-op.
    """

    PEAK = 6.0
    SIGMA = 6.0  # production rule: max(4.0, min(grid)/16) at 96^3

    @pytest.mark.skipif(_find_mida() is None,
                        reason="MIDA NIfTI not on this host")
    def test_warp_at_96_changes_voxels_vs_base(self):
        """Single seed must move voxels — the no-op bug failed here."""
        labels, _, _, _ = make_mida_phantom(
            _find_mida(), grid_shape=(96, 96, 96), dx=0.002, add_lesion=False,
        )
        labels = np.asarray(labels)
        rng = np.random.default_rng(0)
        warped = random_deformation_warp(
            labels, rng,
            max_displacement_voxels=self.PEAK,
            smoothness_voxels=self.SIGMA,
        )
        n_changed = int((warped != labels).sum())
        assert n_changed > 0, (
            f"warp produced zero label changes at 96^3 with peak={self.PEAK}, "
            f"sigma={self.SIGMA}; this is the v1b/v2-morning bug class."
        )

    @pytest.mark.skipif(_find_mida() is None,
                        reason="MIDA NIfTI not on this host")
    def test_warp_at_96_is_seed_dependent(self):
        """Different seeds must produce different warped outputs.

        With the no-op bug, every seed produced byte-identical output to
        each other (and to the base). On real MIDA, healthy params yield
        cross-seed differences in the 1-2.5% range.
        """
        labels, _, _, _ = make_mida_phantom(
            _find_mida(), grid_shape=(96, 96, 96), dx=0.002, add_lesion=False,
        )
        labels = np.asarray(labels)
        out = []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            out.append(
                random_deformation_warp(
                    labels, rng,
                    max_displacement_voxels=self.PEAK,
                    smoothness_voxels=self.SIGMA,
                )
            )
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                diff = int((out[i] != out[j]).sum())
                frac = diff / out[i].size
                assert frac > 0.001, (
                    f"seeds {i} and {j} differ in only {frac:.3%} of voxels "
                    f"at 96^3 — augmentation is essentially a no-op"
                )

    def test_warp_smoothness_rule_yields_meaningful_disp(self):
        """At production sigma the typical |disp| must clear the NN threshold.

        The original min(grid)/6 = 16 rule produced mean |disp| ~0.08 voxels;
        only ~1% of voxels had magnitude >= 0.5 (the NN rounding boundary)
        and label-boundary crossings were ~0. The fixed rule must clear
        a much larger fraction.
        """
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(0)
        shape = (96, 96, 96)
        d = rng.standard_normal((3,) + shape).astype(np.float32)
        for i in range(3):
            d[i] = gaussian_filter(d[i], sigma=self.SIGMA, mode="nearest")
        peak = float(np.max(np.abs(d)))
        d *= self.PEAK / max(peak, 1e-9)
        mag = np.sqrt((d ** 2).sum(axis=0))
        frac_above_half = float((mag >= 0.5).mean())
        assert frac_above_half > 0.05, (
            f"only {frac_above_half:.1%} of voxels have |disp| >= 0.5 — "
            f"NN rounding will collapse most displacements; warp will be "
            f"effectively a no-op"
        )

    @pytest.mark.skipif(_find_mida() is None,
                        reason="MIDA NIfTI not on this host")
    def test_warp_preserves_brain_anatomy_on_real_mida(self):
        """At production params, brain-tissue mask should change <2%.

        Guards against a future "fix" that overshoots — e.g. peak=20 would
        also kill the no-op bug but would scramble anatomy.
        """
        labels, _, _, _ = make_mida_phantom(
            _find_mida(), grid_shape=(96, 96, 96), dx=0.002, add_lesion=False,
        )
        labels = np.asarray(labels)
        brain_pkg = frozenset(
            {2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 20, 21, 99, 100, 116}
        )
        brain_mask = np.isin(labels, list(brain_pkg))
        rng = np.random.default_rng(0)
        warped = random_deformation_warp(
            labels, rng,
            max_displacement_voxels=self.PEAK,
            smoothness_voxels=self.SIGMA,
        )
        warped_brain = np.isin(warped, list(brain_pkg))
        iou = float((brain_mask & warped_brain).sum()
                    / max((brain_mask | warped_brain).sum(), 1))
        assert iou > 0.98, (
            f"brain-mask IoU {iou:.3f} too low — warp params disrupted anatomy"
        )


# ---------------------------------------------------------------------------
# Property mapping at production scale (regression for bug #2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_find_mida() is None,
                    reason="MIDA NIfTI not on this host")
class TestMidaJitterAtProductionScale:
    """Catches the jittered_properties label-clip bug on real MIDA at 96^3."""

    @pytest.fixture(scope="class")
    def warped_mida_labels(self):
        labels, _, _, _ = make_mida_phantom(
            _find_mida(), grid_shape=(96, 96, 96), dx=0.002, add_lesion=False,
        )
        rng = np.random.default_rng(0)
        return random_deformation_warp(
            np.asarray(labels), rng,
            max_displacement_voxels=6.0, smoothness_voxels=6.0,
        )

    def test_water_coupling_fraction_healthy(self, warped_mida_labels):
        """Background voxels must map to water — clipped-table bug had ~0%."""
        key = jr.PRNGKey(0)
        c = mida_jittered_properties(
            jnp.asarray(warped_mida_labels), key, intensity=1.0,
        )["sound_speed"]
        water_frac = float(((c > 1490.0) & (c < 1510.0)).mean())
        assert water_frac >= 0.20, (
            f"water_frac={water_frac:.1%} < 20% — background voxels are not "
            f"mapping to water (the v1b property-clip bug)"
        )

    def test_cortical_bone_present(self, warped_mida_labels):
        """Skull voxels must reach cortical c — clipped-table capped near 2300."""
        key = jr.PRNGKey(0)
        c = mida_jittered_properties(
            jnp.asarray(warped_mida_labels), key, intensity=1.0,
        )["sound_speed"]
        c_max = float(c.max())
        assert c_max >= 2700.0, (
            f"c_max={c_max:.0f} m/s < 2700 — skull voxels did not reach "
            f"cortical bone speed; table likely clipped"
        )

    def test_no_single_tissue_dominates(self, warped_mida_labels):
        """No 50-m/s band should hold more than 70% of voxels.

        v1b had ~87% of voxels at the trabecular band (~2300 m/s).
        Healthy MIDA has water as the largest band at ~40-45%.
        """
        key = jr.PRNGKey(0)
        c = mida_jittered_properties(
            jnp.asarray(warped_mida_labels), key, intensity=1.0,
        )["sound_speed"]
        c_bin = jnp.clip(jnp.round(c / 50.0), 0, 79).astype(jnp.int32)
        counts = jnp.bincount(c_bin.ravel(), length=80)
        mode_frac = float(counts.max() / c.size)
        assert mode_frac < 0.70, (
            f"a single 50-m/s band holds {mode_frac:.1%} of voxels — "
            f"a tissue collapse like the v1b trabecular bug"
        )

    def test_seeds_produce_different_acoustic_fields(self, warped_mida_labels):
        """Different jitter keys must produce numerically-distinct c fields."""
        c0 = mida_jittered_properties(
            jnp.asarray(warped_mida_labels), jr.PRNGKey(0), intensity=1.0,
        )["sound_speed"]
        c1 = mida_jittered_properties(
            jnp.asarray(warped_mida_labels), jr.PRNGKey(1), intensity=1.0,
        )["sound_speed"]
        n_diff = int((np.asarray(c0) != np.asarray(c1)).sum())
        assert n_diff > c0.size * 0.5, (
            f"only {n_diff} voxels differ between jitter seeds — jitter "
            f"randomness is not flowing through"
        )


# ---------------------------------------------------------------------------
# Air-cavity water-fill (regression for bug #4)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_find_mida() is None,
                    reason="MIDA NIfTI not on this host")
class TestAirCavityWaterFill:
    """Catches the bug where air voxels destabilise the pseudospectral solver.

    Background: MIDA's internal air labels (sinuses, ear canal, oral cavity
    etc. — labels 26-31, 85, 97) map to c=343 m/s, rho=1.225 kg/m^3. With a
    pseudospectral solver and dx=2 mm, the 4x sound-speed and 800x density
    discontinuity at the air-tissue boundary triggered exponential blow-up:
    the field doubled every 2-3 timesteps starting at t~30, NaN-cascading
    by t=50 — leaving 99% of `observed_data` unusable. The fix is to
    water-fill the air-cavity labels at the label level inside
    `_build_phantom_labels` *before* the augmentation pipeline maps them
    to acoustic properties.
    """

    def test_build_phantom_labels_water_fills_air_for_mida(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from gen_phase0 import _build_phantom_labels  # type: ignore

        labels = _build_phantom_labels(
            phantom="mida",
            grid_shape=(96, 96, 96),
            dx=0.002,
            mida_path=_find_mida(),
        )
        air_voxels = int(np.isin(labels, list(MIDA_INTERNAL_AIR_LABELS)).sum())
        assert air_voxels == 0, (
            f"{air_voxels} voxels still hold internal-air labels "
            f"({sorted(MIDA_INTERNAL_AIR_LABELS)}) after _build_phantom_labels. "
            f"Pseudospectral solver will explode at the air-tissue boundary."
        )

    def test_water_filled_labels_produce_no_air_speed_voxels(self):
        """End-to-end: post-augmentation c should not contain c<1000 voxels."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from gen_phase0 import _build_phantom_labels  # type: ignore

        labels = _build_phantom_labels(
            phantom="mida",
            grid_shape=(96, 96, 96),
            dx=0.002,
            mida_path=_find_mida(),
        )
        # No warp here — we want a clean test of the property pipeline
        # alone. Map labels through the production augmentation path.
        c = mida_jittered_properties(
            jnp.asarray(labels), jr.PRNGKey(0), intensity=1.0,
        )["sound_speed"]
        c_arr = np.asarray(c)
        n_air_speed = int((c_arr < 1000.0).sum())
        assert n_air_speed == 0, (
            f"{n_air_speed} voxels have c < 1000 m/s after water-fill; "
            f"air-cavity remap did not cover all of {sorted(MIDA_INTERNAL_AIR_LABELS)}"
        )

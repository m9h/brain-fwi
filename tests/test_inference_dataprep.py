"""Tests for brain_fwi.inference.dataprep — convert Phase-0 samples to
training-ready (theta, d) pairs for NPE / diffusion priors / surrogates.

Contract:
  - ``siren_from_sample`` reconstructs the original SIREN module from
    ``siren_weights_bytes`` + ``siren_arch`` — i.e., round-trips exactly.
  - ``theta_from_sample`` returns a flat 1-D theta vector whose length
    equals the total number of SIREN inexact-array weights.
  - Both accept ``siren_arch`` as either a dict (in-memory samples)
    or a JSON string (samples freshly loaded from HDF5 via ShardedReader).
  - Reconstructed SIREN produces identical forward-pass output to the
    original.
"""

import io
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from brain_fwi.inference.dataprep import siren_from_sample, theta_from_sample
from brain_fwi.inversion.param_field import SIREN


def _sample_from_siren(siren: SIREN, arch: dict, arch_format: str = "dict") -> dict:
    """Mint a minimal sample dict the way gen_phase0.py would store it."""
    buf = io.BytesIO()
    eqx.tree_serialise_leaves(buf, siren)
    weights = np.frombuffer(buf.getvalue(), dtype=np.uint8).copy()
    arch_value = arch if arch_format == "dict" else json.dumps(arch)
    return {
        "sample_id": "test",
        "siren_weights_bytes": weights,
        "siren_arch": arch_value,
    }


ARCH = {
    "in_dim": 3,
    "hidden_dim": 32,
    "n_hidden": 2,
    "out_dim": 1,
    "omega_0": 30.0,
}


class TestSirenFromSample:
    def test_reconstruct_from_dict_arch(self):
        orig = SIREN(
            in_dim=ARCH["in_dim"], hidden_dim=ARCH["hidden_dim"],
            n_hidden=ARCH["n_hidden"], out_dim=ARCH["out_dim"],
            omega_0=ARCH["omega_0"], key=jr.PRNGKey(3),
        )
        sample = _sample_from_siren(orig, ARCH, arch_format="dict")

        rebuilt = siren_from_sample(sample)
        x = jnp.array([0.1, -0.4, 0.2])
        np.testing.assert_allclose(
            np.asarray(orig(x)), np.asarray(rebuilt(x)), rtol=1e-6,
        )

    def test_reconstruct_from_json_string_arch(self):
        """ShardedReader returns JSON-string attrs; must handle both."""
        orig = SIREN(
            in_dim=ARCH["in_dim"], hidden_dim=ARCH["hidden_dim"],
            n_hidden=ARCH["n_hidden"], out_dim=ARCH["out_dim"],
            omega_0=ARCH["omega_0"], key=jr.PRNGKey(4),
        )
        sample = _sample_from_siren(orig, ARCH, arch_format="json")
        rebuilt = siren_from_sample(sample)
        x = jnp.zeros((ARCH["in_dim"],))
        np.testing.assert_allclose(
            np.asarray(orig(x)), np.asarray(rebuilt(x)), rtol=1e-6,
        )

    def test_missing_required_field_raises(self):
        with pytest.raises(KeyError, match="siren_weights_bytes"):
            siren_from_sample({"sample_id": "x", "siren_arch": ARCH})
        with pytest.raises(KeyError, match="siren_arch"):
            siren_from_sample({
                "sample_id": "x", "siren_weights_bytes": np.zeros(10, dtype=np.uint8),
            })


class TestThetaFromSample:
    def test_shape_is_flat_vector(self):
        orig = SIREN(
            in_dim=ARCH["in_dim"], hidden_dim=ARCH["hidden_dim"],
            n_hidden=ARCH["n_hidden"], out_dim=ARCH["out_dim"],
            omega_0=ARCH["omega_0"], key=jr.PRNGKey(5),
        )
        sample = _sample_from_siren(orig, ARCH)
        theta = theta_from_sample(sample)
        assert theta.ndim == 1
        # Count inexact-array leaves of orig to cross-check
        expected = sum(
            int(l.size) for l in jax.tree.leaves(eqx.filter(orig, eqx.is_inexact_array))
        )
        assert theta.shape == (expected,)

    def test_different_seeds_give_different_theta(self):
        s1 = _sample_from_siren(
            SIREN(**ARCH, key=jr.PRNGKey(1)), ARCH,
        )
        s2 = _sample_from_siren(
            SIREN(**ARCH, key=jr.PRNGKey(2)), ARCH,
        )
        t1 = theta_from_sample(s1)
        t2 = theta_from_sample(s2)
        assert not np.allclose(t1, t2)

    def test_theta_matches_concatenated_weights(self):
        """Sanity: theta is the concatenation of SIREN's inexact-array leaves."""
        orig = SIREN(**ARCH, key=jr.PRNGKey(7))
        sample = _sample_from_siren(orig, ARCH)
        theta = theta_from_sample(sample)

        direct = np.concatenate([
            np.asarray(l).ravel()
            for l in jax.tree.leaves(eqx.filter(orig, eqx.is_inexact_array))
        ])
        np.testing.assert_allclose(theta, direct, rtol=1e-6)

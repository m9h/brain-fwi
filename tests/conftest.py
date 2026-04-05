"""Shared fixtures for brain-fwi tests."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", False)

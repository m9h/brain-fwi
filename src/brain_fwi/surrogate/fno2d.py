"""Toy 2D Fourier Neural Operator for Phase 4 Evidence 9.2.1.

Classic FNO building blocks (Li et al. 2020), minimal Equinox port:

  - :class:`SpectralConv2D` — FFT → truncate to ``(modes_h, modes_w)`` →
    multiply by learnable complex weights → IFFT.
  - :class:`FNO2D` — lift (point-wise linear) → stack of Fourier blocks
    (spectral conv + point-wise skip + GELU) → project (point-wise linear).
  - :class:`CToTraceFNO` — adds a global-average-pool + MLP head that
    collapses the spatial FNO output to an ``(N_t,)`` trace vector. This
    is the concrete shape we want for the toy acoustic-wave experiment:
    ``(H, W, 1) sound speed → (N_t,) sensor trace``.

Used by ``scripts/toy_2d_fno_experiment.py``. Not a production
architecture — scope is "can FNO hit <1% rel-L2 on a 2D acoustic toy?"
to unblock the 3D MVP.
"""

from __future__ import annotations

from typing import List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# ---------------------------------------------------------------------------
# Spectral convolution layer
# ---------------------------------------------------------------------------

class SpectralConv2D(eqx.Module):
    """2D spectral convolution with mode truncation.

    Weights are complex, stored as real ``(2, modes_h, modes_w,
    in_channels, out_channels)`` tensors (last dim 2 = [real, imag]) so
    Equinox's ``eqx.is_inexact_array`` filter treats them as trainable.
    """

    weight: jax.Array
    modes_h: int = eqx.field(static=True)
    modes_w: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_h: int,
        modes_w: int,
        key: jax.Array,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_h = modes_h
        self.modes_w = modes_w
        # He-style init scaled for complex weights. We store separate
        # real / imag components as the last axis so JAX treats them as
        # a real pytree leaf (autodiff through complex-valued spectral
        # ops is differentiable but pytree-awkward in Equinox).
        scale = (1.0 / (in_channels * out_channels)) ** 0.5
        self.weight = scale * jr.normal(
            key, (2, modes_h, modes_w, in_channels, out_channels),
        )

    def _check_shape(self, spatial_hw: Tuple[int, int]) -> None:
        h, w = spatial_hw
        # Real-FFT output width is floor(w/2)+1; mode truncation must fit.
        max_modes_w = w // 2 + 1
        if self.modes_w > max_modes_w:
            raise ValueError(
                f"modes_w={self.modes_w} exceeds rfft width {max_modes_w} for W={w}"
            )
        if self.modes_h > h:
            raise ValueError(
                f"modes_h={self.modes_h} exceeds height {h}"
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        """``x: (H, W, in_channels)`` → ``(H, W, out_channels)``."""
        h, w, _ = x.shape
        self._check_shape((h, w))

        # rFFT along spatial axes. Input shape (H, W, Cin) → (H, W/2+1, Cin).
        x_ft = jnp.fft.rfft2(x, axes=(0, 1))

        # Complex weight from the real/imag split.
        weight_complex = self.weight[0] + 1j * self.weight[1]

        # Allocate full spectrum output then fill truncated slice via
        # Einstein summation over (modes_h, modes_w, Cin, Cout).
        out_ft = jnp.zeros(
            (h, w // 2 + 1, self.out_channels),
            dtype=x_ft.dtype,
        )
        # Top-left block: low modes in both axes.
        out_ft = out_ft.at[: self.modes_h, : self.modes_w].set(
            jnp.einsum(
                "hwi,hwio->hwo",
                x_ft[: self.modes_h, : self.modes_w],
                weight_complex,
            )
        )
        # Inverse FFT back to spatial domain.
        return jnp.fft.irfft2(out_ft, s=(h, w), axes=(0, 1))


# ---------------------------------------------------------------------------
# Fourier block
# ---------------------------------------------------------------------------

class FourierBlock2D(eqx.Module):
    """Spectral conv + point-wise (1x1) linear + GELU."""

    spectral: SpectralConv2D
    pointwise: eqx.nn.Linear

    def __init__(
        self,
        channels: int,
        modes_h: int,
        modes_w: int,
        key: jax.Array,
    ):
        s_key, p_key = jr.split(key)
        self.spectral = SpectralConv2D(
            in_channels=channels, out_channels=channels,
            modes_h=modes_h, modes_w=modes_w, key=s_key,
        )
        self.pointwise = eqx.nn.Linear(channels, channels, key=p_key)

    def __call__(self, x: jax.Array) -> jax.Array:
        spec = self.spectral(x)
        # Broadcast pointwise conv via vmap over spatial axes.
        skip = jax.vmap(jax.vmap(self.pointwise))(x)
        return jax.nn.gelu(spec + skip)


# ---------------------------------------------------------------------------
# FNO2D
# ---------------------------------------------------------------------------

class FNO2D(eqx.Module):
    """Lift → Fourier blocks → project. Spatially shape-preserving."""

    lift: eqx.nn.Linear
    blocks: List[FourierBlock2D]
    project: eqx.nn.Linear

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int,
        modes: int,
        n_blocks: int,
        key: jax.Array,
    ):
        keys = jr.split(key, n_blocks + 2)
        self.lift = eqx.nn.Linear(in_channels, width, key=keys[0])
        self.blocks = [
            FourierBlock2D(channels=width, modes_h=modes, modes_w=modes, key=k)
            for k in keys[1:-1]
        ]
        self.project = eqx.nn.Linear(width, out_channels, key=keys[-1])

    def __call__(self, x: jax.Array) -> jax.Array:
        """``x: (H, W, in_channels)`` → ``(H, W, out_channels)``."""
        h = jax.vmap(jax.vmap(self.lift))(x)
        for block in self.blocks:
            h = block(h)
        return jax.vmap(jax.vmap(self.project))(h)


# ---------------------------------------------------------------------------
# CToTrace head
# ---------------------------------------------------------------------------

class CToTraceFNO(eqx.Module):
    """``(H, W, 1) sound-speed`` → ``(N_t,) sensor trace``.

    FNO backbone retains spatial structure; a global-average-pool + MLP
    head collapses to the trace. The toy-2D experiment uses a single
    fixed source / single fixed sensor so this head is sufficient —
    multi-source / multi-sensor production operators will need a
    per-location output head.
    """

    fno: FNO2D
    head: eqx.nn.MLP
    grid_h: int = eqx.field(static=True)
    grid_w: int = eqx.field(static=True)
    n_timesteps: int = eqx.field(static=True)

    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        n_timesteps: int,
        width: int,
        modes: int,
        n_blocks: int,
        key: jax.Array,
    ):
        fno_key, head_key = jr.split(key)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_timesteps = n_timesteps
        self.fno = FNO2D(
            in_channels=1, out_channels=width, width=width,
            modes=modes, n_blocks=n_blocks, key=fno_key,
        )
        self.head = eqx.nn.MLP(
            in_size=width, out_size=n_timesteps,
            width_size=max(width, 32), depth=2, key=head_key,
        )

    def __call__(self, c_field: jax.Array) -> jax.Array:
        features = self.fno(c_field)           # (H, W, width)
        pooled = jnp.mean(features, axis=(0, 1))  # (width,)
        return self.head(pooled)

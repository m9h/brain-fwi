"""U-shaped Neural Operator (UNO) implementation for JAX/Equinox.

Based on "U-shaped Neural Operators" (You et al. 2022).
Provides multi-scale spectral convolutions with skip connections.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

def generate_modes_slices(modes: tuple[int, ...]) -> list[list[slice]]:
    """Generates slices for Fourier modes."""
    import itertools
    
    num_spatial_dims = len(modes)
    ranges = []
    for m in modes[:-1]:
        ranges.append([(slice(0, m),), (slice(-m, None),)])
    ranges.append([(slice(0, modes[-1]),)])
    
    return [
        list(itertools.chain.from_iterable(combination))
        for combination in itertools.product(*ranges)
    ]

class UNOSpectralConv(eqx.Module):
    """Spectral convolution that supports resampling."""
    
    num_spatial_dims: int = eqx.field(static=True)
    num_modes: tuple[int, ...] = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    
    weights_real: Float[Array, "G Co Ci ..."]
    weights_imag: Float[Array, "G Co Ci ..."]

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_modes: Union[int, tuple[int, ...]],
        *,
        key: jr.PRNGKey,
    ):
        if isinstance(num_modes, int):
            num_modes = (num_modes,) * num_spatial_dims
        
        self.num_spatial_dims = num_spatial_dims
        self.num_modes = num_modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        weight_shape = (
            2 ** (num_spatial_dims - 1),
            out_channels,
            in_channels,
        ) + num_modes
        
        rk, ik = jr.split(key)
        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = scale * jr.normal(rk, weight_shape)
        self.weights_imag = scale * jr.normal(ik, weight_shape)

    def __call__(self, x: Float[Array, "Ci ..."], out_shape: Optional[tuple[int, ...]] = None) -> Float[Array, "Co ..."]:
        Ci, *in_shape = x.shape
        if out_shape is None:
            out_shape = tuple(in_shape)
            
        x_fft = jnp.fft.rfftn(x, axes=list(range(1, self.num_spatial_dims + 1)))
        weights = self.weights_real + 1j * self.weights_imag
        
        out_fft_shape = (self.out_channels,) + out_shape[:-1] + (out_shape[-1] // 2 + 1,)
        out_fft = jnp.zeros(out_fft_shape, dtype=x_fft.dtype)
        
        slices = generate_modes_slices(self.num_modes)
        for i, slc in enumerate(slices):
            target_slc = []
            source_slc = []
            weight_slc = []
            skip_group = False
            
            for d, s in enumerate(slc):
                m = self.num_modes[d]
                limit_in = x_fft.shape[d+1]
                limit_out = out_fft.shape[d+1]
                
                if s.start == 0:
                    actual_m = min(m, limit_in, limit_out)
                    source_slc.append(slice(0, actual_m))
                    target_slc.append(slice(0, actual_m))
                    weight_slc.append(slice(0, actual_m))
                else:
                    # slice(-m, None)
                    if limit_in > m and limit_out > m:
                        source_slc.append(slice(-m, None))
                        target_slc.append(slice(-m, None))
                        weight_slc.append(slice(0, m))
                    else:
                        skip_group = True
                        break
            
            if skip_group:
                continue

            full_src_slc = (slice(None),) + tuple(source_slc)
            full_tgt_slc = (slice(None),) + tuple(target_slc)
            full_w_slc = (slice(None), slice(None)) + tuple(weight_slc)
            
            res = jnp.einsum("i...,oi...->o...", x_fft[full_src_slc], weights[i][full_w_slc])
            out_fft = out_fft.at[full_tgt_slc].set(res)
            
        return jnp.fft.irfftn(out_fft, s=out_shape, axes=list(range(1, self.num_spatial_dims + 1)))

class UNOBlock(eqx.Module):
    """A single UNO block with spectral conv and skip connection."""
    
    spectral: UNOSpectralConv
    skip: eqx.nn.Conv
    activation: Callable

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_modes: Union[int, tuple[int, ...]],
        *,
        activation: Callable = jax.nn.gelu,
        key: jr.PRNGKey,
    ):
        sk, skipk = jr.split(key)
        self.spectral = UNOSpectralConv(
            num_spatial_dims, in_channels, out_channels, num_modes, key=sk
        )
        self.skip = eqx.nn.Conv(
            num_spatial_dims, in_channels, out_channels, kernel_size=1, key=skipk
        )
        self.activation = activation

    def __call__(self, x: Float[Array, "Ci ..."], out_shape: Optional[tuple[int, ...]] = None) -> Float[Array, "Co ..."]:
        Ci, *in_shape = x.shape
        out_s = out_shape if out_shape is not None else tuple(in_shape)
        
        out_spec = self.spectral(x, out_shape=out_s)
        out_skip = self.skip(x)
        
        if out_s != tuple(in_shape):
            out_skip = jax.image.resize(out_skip, (self.spectral.out_channels,) + out_s, method="linear")
            
        return self.activation(out_spec + out_skip)

class UNONet(eqx.Module):
    """U-shaped Neural Operator MVP."""
    
    lifting: eqx.nn.Conv
    encoder_blocks: list[UNOBlock]
    decoder_blocks: list[UNOBlock]
    projection: eqx.nn.Conv
    
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_modes: int,
        depth: int = 2,
        key: jr.PRNGKey = jr.PRNGKey(0),
    ):
        keys = jr.split(key, 2 + 2 * depth)
        
        self.lifting = eqx.nn.Conv(
            num_spatial_dims, in_channels, hidden_channels, kernel_size=1, key=keys[0]
        )
        
        self.encoder_blocks = []
        curr_channels = hidden_channels
        for i in range(depth):
            self.encoder_blocks.append(
                UNOBlock(num_spatial_dims, curr_channels, curr_channels * 2, num_modes, key=keys[1 + i])
            )
            curr_channels *= 2
            
        self.decoder_blocks = []
        for i in range(depth):
            # Decoder blocks take concat skip connections
            self.decoder_blocks.append(
                UNOBlock(num_spatial_dims, curr_channels + (curr_channels // 2), curr_channels // 2, num_modes, key=keys[1 + depth + i])
            )
            curr_channels //= 2
            
        self.projection = eqx.nn.Conv(
            num_spatial_dims, curr_channels, out_channels, kernel_size=1, key=keys[-1]
        )

    def __call__(self, x: Float[Array, "Ci ..."]) -> Float[Array, "Co ..."]:
        x = self.lifting(x)
        
        skips = []
        for block in self.encoder_blocks:
            skips.append(x)
            in_s = x.shape[1:]
            out_s = tuple(max(1, s // 2) for s in in_s)
            x = block(x, out_shape=out_s)
            
        for block in self.decoder_blocks:
            skip_x = skips.pop()
            if x.shape[1:] != skip_x.shape[1:]:
                x = jax.image.resize(x, (x.shape[0],) + skip_x.shape[1:], method="linear")
            
            x = jnp.concatenate([x, skip_x], axis=0)
            x = block(x)
            
        return self.projection(x)

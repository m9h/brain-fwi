
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from pdequinox.conv._spectral_conv import SpectralConv, spectral_conv_nd

def test_downsampling_spectral_conv():
    # Test if we can do a spectral conv that changes output shape
    key = jr.PRNGKey(0)
    in_channels = 2
    out_channels = 4
    num_modes = (8, 8)
    input_shape = (in_channels, 32, 32)
    output_shape = (16, 16)
    
    x = jr.normal(key, input_shape)
    
    # We need a modified spectral_conv_nd that takes output_shape
    def resample_spectral_conv_nd(input, weight_r, weight_i, modes, out_s):
        _, *si, sl = input.shape
        weight = weight_r + 1j * weight_i
        _, o, *_ = weight.shape
        x_fft = jnp.fft.rfftn(input, s=(*si, sl))
        
        # New output shape in Fourier domain
        # The number of modes we keep is 'modes'
        # We create an output buffer of size based on out_s
        out_si = out_s[:-1]
        out_sl = out_s[-1]
        out = jnp.zeros([o, *out_si, out_sl // 2 + 1], dtype=input.dtype) + 0j
        
        from pdequinox.conv._spectral_conv import generate_modes_slices
        for i, slice_i in enumerate(generate_modes_slices(modes)):
            # We must ensure slice_i is within bounds of 'out'
            # If out is smaller than x_fft, we might need to truncate
            # But 'modes' should already be <= min(si, out_s)//2
            print(f"Slice {i}: x_fft shape {x_fft[tuple(slice_i)].shape}, weight shape {weight[i].shape}")
            matmul_out = jnp.einsum("i...,oi...->o...", x_fft[tuple(slice_i)], weight[i])
            out = out.at[tuple(slice_i)].set(matmul_out)
            
        return jnp.fft.irfftn(out, s=out_s)

    conv = SpectralConv(2, in_channels, out_channels, num_modes, key=key)
    y = resample_spectral_conv_nd(x, conv.weights_real, conv.weights_imag, num_modes, output_shape)
    
    assert y.shape == (out_channels, 16, 16)
    print("Downsampling spectral conv works!")

if __name__ == "__main__":
    test_downsampling_spectral_conv()

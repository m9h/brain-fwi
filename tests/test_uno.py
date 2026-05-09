
import jax
import jax.numpy as jnp
import jax.random as jr
from brain_fwi.surrogate.uno import UNONet

def test_uno_2d():
    key = jr.PRNGKey(0)
    # (Ci, H, W)
    x = jr.normal(key, (1, 32, 32))
    
    model = UNONet(
        num_spatial_dims=2,
        in_channels=1,
        out_channels=16,
        hidden_channels=8,
        num_modes=4,
        depth=2,
        key=key
    )
    
    y = model(x)
    assert y.shape == (16, 32, 32)
    print("UNONet 2D works!")

def test_uno_3d():
    key = jr.PRNGKey(0)
    # (Ci, D, H, W)
    x = jr.normal(key, (2, 16, 16, 16))
    
    model = UNONet(
        num_spatial_dims=3,
        in_channels=2,
        out_channels=8,
        hidden_channels=4,
        num_modes=2,
        depth=1,
        key=key
    )
    
    y = model(x)
    assert y.shape == (8, 16, 16, 16)
    print("UNONet 3D works!")

def test_uno_default_activation_is_silu():
    """phase4_production_hifi.md §3 mandates SiLU activation: 'SiLU/Sine
    (SIREN-inspired) to handle sharp skull gradients.' Pinning the
    default here so a future GELU/ReLU regression goes red on this
    test rather than silently degrading skull-interface fidelity."""
    from brain_fwi.surrogate.uno import UNOBlock, UNONet

    # Block-level default
    block = UNOBlock(
        num_spatial_dims=3,
        in_channels=2, out_channels=4,
        num_modes=2,
        key=jr.PRNGKey(0),
    )
    assert block.activation is jax.nn.silu, (
        f"UNOBlock default activation is "
        f"{getattr(block.activation, '__name__', block.activation)}; "
        f"phase4_production_hifi.md §3 mandates SiLU"
    )

    # Net-level: every encoder + decoder block inherits SiLU
    net = UNONet(
        num_spatial_dims=3,
        in_channels=2, out_channels=4,
        hidden_channels=4, num_modes=2, depth=1,
        key=jr.PRNGKey(0),
    )
    for label, blk in (
        [("encoder", b) for b in net.encoder_blocks]
        + [("decoder", b) for b in net.decoder_blocks]
    ):
        assert blk.activation is jax.nn.silu, (
            f"UNONet {label} block activation = "
            f"{getattr(blk.activation, '__name__', blk.activation)}; "
            f"expected jax.nn.silu"
        )


if __name__ == "__main__":
    test_uno_2d()
    test_uno_3d()
    test_uno_default_activation_is_silu()

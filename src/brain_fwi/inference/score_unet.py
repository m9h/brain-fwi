"""Image-space conv U-Net score network for diffusion priors over 2D fields.

Drop-in for the MLP ``ScoreMLP`` in :mod:`brain_fwi.inference.diffusion`: same
``__call__(theta_t: (D,), t) -> (D,)`` interface (flatten/unflatten internally),
so it composes with ``train_score_matching`` / ``ddim_sample`` / ``dps_sample``
unchanged. The multi-scale down/up + skip structure gives the global spatial
coherence a shallow CNN lacks (the project's ``surrogate.uno`` is the spectral
neural-operator cousin; for a fixed-resolution diffusion score net a plain conv
U-Net is the standard, better-behaved choice).

Trained data-free on anatomy samples (e.g. Birnbaum stroke slices, see
:mod:`brain_fwi.phantoms.birnbaum`) and used as a learned prior to guide FWI
(DPS-style: data-misfit gradient + score). See
``docs/design/diffusion_prior_fwi.md``.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


def sinusoidal_embedding(t: jax.Array, dim: int) -> jax.Array:
    """Standard transformer/DDPM sinusoidal time embedding."""
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / max(half - 1, 1))
    ang = jnp.atleast_1d(t).astype(jnp.float32) * freqs
    e = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)])
    return jnp.concatenate([e, jnp.zeros(dim - e.shape[0])]) if e.shape[0] < dim else e[:dim]


class _Block(eqx.Module):
    conv: eqx.nn.Conv2d
    norm: eqx.nn.GroupNorm
    tproj: eqx.nn.Linear

    def __init__(self, cin, cout, tdim, *, key):
        k1, k2 = jr.split(key)
        self.conv = eqx.nn.Conv2d(cin, cout, 3, padding=1, key=k1)
        self.norm = eqx.nn.GroupNorm(min(8, cout), cout)
        self.tproj = eqx.nn.Linear(tdim, cout, key=k2)

    def __call__(self, x, temb):
        h = self.conv(x) + self.tproj(temb).reshape(-1, 1, 1)
        return jax.nn.silu(self.norm(h))


class UNetScore(eqx.Module):
    """2-level conv U-Net score net ``s_phi(x_t, t)`` over ``H x W`` fields."""

    tmlp: eqx.nn.MLP
    enc0: _Block; down0: eqx.nn.Conv2d
    enc1: _Block; down1: eqx.nn.Conv2d
    mid: _Block
    up1c: eqx.nn.Conv2d; dec1: _Block
    up0c: eqx.nn.Conv2d; dec0: _Block
    out: eqx.nn.Conv2d
    H: int = eqx.field(static=True); W: int = eqx.field(static=True)
    tdim: int = eqx.field(static=True); te0: int = eqx.field(static=True)

    def __init__(self, H, W, C=32, tdim=128, te0=32, *, key):
        ks = jr.split(key, 11)
        self.H, self.W, self.tdim, self.te0 = H, W, tdim, te0
        self.tmlp = eqx.nn.MLP(te0, tdim, width_size=tdim, depth=1, key=ks[0])
        self.enc0 = _Block(1, C, tdim, key=ks[1])
        self.down0 = eqx.nn.Conv2d(C, 2 * C, 3, stride=2, padding=1, key=ks[2])
        self.enc1 = _Block(2 * C, 2 * C, tdim, key=ks[3])
        self.down1 = eqx.nn.Conv2d(2 * C, 4 * C, 3, stride=2, padding=1, key=ks[4])
        self.mid = _Block(4 * C, 4 * C, tdim, key=ks[5])
        self.up1c = eqx.nn.Conv2d(4 * C, 2 * C, 3, padding=1, key=ks[6])
        self.dec1 = _Block(2 * C + 2 * C, 2 * C, tdim, key=ks[7])
        self.up0c = eqx.nn.Conv2d(2 * C, C, 3, padding=1, key=ks[8])
        self.dec0 = _Block(C + C, C, tdim, key=ks[9])
        self.out = eqx.nn.Conv2d(C, 1, 3, padding=1, key=ks[10])

    def __call__(self, theta_t: jax.Array, t: jax.Array) -> jax.Array:
        temb = self.tmlp(sinusoidal_embedding(t, self.te0))
        x = theta_t.reshape(1, self.H, self.W)
        s0 = self.enc0(x, temb)
        s1 = self.enc1(jax.nn.silu(self.down0(s0)), temb)
        m = self.mid(jax.nn.silu(self.down1(s1)), temb)
        u1 = self.up1c(jax.image.resize(m, (m.shape[0], s1.shape[1], s1.shape[2]), "nearest"))
        d1 = self.dec1(jnp.concatenate([u1, s1], axis=0), temb)
        u0 = self.up0c(jax.image.resize(d1, (d1.shape[0], s0.shape[1], s0.shape[2]), "nearest"))
        d0 = self.dec0(jnp.concatenate([u0, s0], axis=0), temb)
        return self.out(d0).reshape(-1)

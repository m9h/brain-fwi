"""3D MOFI: differentiable SE(3) rigid warp of a skull template + pose recovery.

Skull-from-data (the clinical step, Bates 2026 MOFI in 3D): the skull is an unknown
template at an unknown pose. Parametrise it by 6 DOF (translation t, Euler angles
ang); warp differentiably so the pose can be recovered by minimising the acoustic
data misfit (jax.grad through warp -> medium -> j-Wave). Self-test here validates
the warp+gradient on a field-MSE misfit (no j-Wave); the GB10 runner adds the wave
physics. Foundation for run_mofi3d.
"""
import jax, jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


def rotation_matrix(ang):
    """ZYX Euler angles (rx, ry, rz) radians -> 3x3 rotation."""
    rx, ry, rz = ang
    cx, sx = jnp.cos(rx), jnp.sin(rx); cy, sy = jnp.cos(ry), jnp.sin(ry); cz, sz = jnp.cos(rz), jnp.sin(rz)
    Rx = jnp.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = jnp.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = jnp.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rigid_warp_3d(field, t, ang):
    """Apply rigid pose (translation t in voxels, rotation ang) to a 3D field,
    differentiable in (t, ang). Inverse-warp sampling about the volume centre."""
    S = field.shape[0]; ctr = (S - 1) / 2.0
    R = rotation_matrix(ang)
    g = jnp.stack(jnp.meshgrid(jnp.arange(S), jnp.arange(S), jnp.arange(S), indexing="ij")).astype(jnp.float32)
    cen = g - ctr - t[:, None, None, None]
    src = jnp.einsum("ij,jxyz->ixyz", R.T, cen) + ctr            # inverse map (pull-back)
    return map_coordinates(field, [src[0], src[1], src[2]], order=1, mode="constant")


if __name__ == "__main__":
    import numpy as np, optax
    # self-test: warp a synthetic skull-shell by a known pose, recover it
    S = 48; zz, yy, xx = np.indices((S, S, S)); ctr = (S - 1) / 2
    # ellipsoidal shell + asymmetric bump -> rotation is identifiable (a sphere isn't)
    rad = np.sqrt(((zz - ctr) / 1.0) ** 2 + ((yy - ctr) / 1.3) ** 2 + ((xx - ctr) / 0.75) ** 2)
    shell = ((rad > 16) & (rad < 19)).astype(np.float32)
    shell[int(ctr) + 10:, :int(ctr), :] += 0.5                    # break remaining symmetry
    shell = jnp.asarray(np.clip(shell, 0, 1).astype(np.float32))
    t_true = jnp.array([3.0, -2.0, 1.5]); ang_true = jnp.array([0.05, -0.08, 0.10])
    target = rigid_warp_3d(shell, t_true, ang_true)
    print(f"self-test: recover pose t={np.asarray(t_true)}, ang={np.asarray(ang_true)}", flush=True)

    def loss(params):
        w = rigid_warp_3d(shell, params["t"], params["ang"])
        return jnp.mean((w - target) ** 2)
    params = {"t": jnp.zeros(3), "ang": jnp.zeros(3)}
    opt = optax.adam(0.2); st = opt.init(params)
    for i in range(200):
        l, g = jax.value_and_grad(loss)(params); up, st = opt.update(g, st); params = optax.apply_updates(params, up)
        if (i + 1) % 50 == 0:
            print(f"  step {i+1}: loss {float(l):.5f}  t={np.round(np.asarray(params['t']),2)}  "
                  f"ang={np.round(np.asarray(params['ang']),3)}", flush=True)
    te = float(jnp.linalg.norm(params["t"] - t_true)); ae = float(jnp.linalg.norm(params["ang"] - ang_true))
    print(f"final pose error: |dt| {te:.2f} vox, |dang| {ae:.3f} rad  -> {'PASS' if te<0.5 and ae<0.02 else 'FAIL'}", flush=True)

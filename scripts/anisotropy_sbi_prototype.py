"""Revisit prototype: SBI (neural posterior estimation) over the LOW-DIMENSIONAL
anisotropy parameters (a_par, a_perp, fibre angle phi), forward-ONLY -- so the
parked FNO's adjoint wall is irrelevant. Proves the concept the FNO-forward would
scale to 3D: a calibrated posterior with UNCERTAINTY on fibre orientation.

Forward = the anisotropy amplitude model (ring transmission): per source-receiver
ray k at angle theta_k, log-attenuation d_k = alpha(theta_k)*L_k,
alpha(theta) = a_par cos^2(theta-phi) + a_perp sin^2(theta-phi) (the exact model
the validated j-Wave anisotropic absorber / an FNO surrogate produces at scale).
"""
import time, numpy as np, jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from brain_fwi.inference.flow import ConditionalFlow, train_npe

rng = np.random.default_rng(0)
# ring geometry -> rays (angle, chord length), homogeneous anisotropic medium
n_tr = 16; R = 0.05
ang = np.linspace(0, 2*np.pi, n_tr, endpoint=False)
tx, ty = R*np.cos(ang), R*np.sin(ang)
pairs = [(i, j) for i in range(n_tr) for j in range(i+1, n_tr)]
pairs = pairs[::2]                                   # subsample for a moderate d_dim
ray_ang = np.array([np.arctan2(ty[j]-ty[i], tx[j]-tx[i]) for i, j in pairs])
ray_len = np.array([np.hypot(tx[j]-tx[i], ty[j]-ty[i]) for i, j in pairs]) * 100  # cm
D_DIM = len(pairs); NOISE = 0.05
print(f"SBI over anisotropy: theta=(a_par,a_perp,phi), {D_DIM} ray features", flush=True)

def forward(theta):                                  # theta: (...,3) -> d: (..., D_DIM)
    a_par, a_perp, phi = theta[..., 0:1], theta[..., 1:2], theta[..., 2:3]
    c2 = np.cos(ray_ang - phi)**2; s2 = np.sin(ray_ang - phi)**2
    return (a_par*c2 + a_perp*s2) * ray_len          # dB per ray

def sample_prior(n):
    a_par = rng.uniform(0.5, 4.0, (n, 1))
    a_perp = rng.uniform(0.5, 10.0, (n, 1))
    phi = rng.uniform(0.0, np.pi, (n, 1))
    return np.concatenate([a_par, a_perp, phi], -1)

# ---- simulate training set (forward-only; no adjoint) ----
N = 12000
theta = sample_prior(N).astype(np.float32)
d = forward(theta).astype(np.float32)
d = d * (1 + NOISE*rng.normal(size=d.shape).astype(np.float32))   # measurement noise
tmu, tsd = theta.mean(0), theta.std(0); dmu, dsd = d.mean(0), d.std(0)
thn = (theta - tmu)/tsd; dn = (d - dmu)/dsd

flow = ConditionalFlow(theta_dim=3, d_dim=D_DIM, key=jax.random.PRNGKey(0),
                       n_transforms=6, nn_width=96, nn_depth=2)
t0 = time.time()
flow, losses = train_npe(flow, jnp.asarray(thn), jnp.asarray(dn),
                         jax.random.PRNGKey(1), n_steps=3000, learning_rate=2e-3,
                         batch_size=512, verbose=False)
print(f"trained NPE in {time.time()-t0:.0f}s, final loss {losses[-1]:.3f}", flush=True)

def posterior(d_obs, n=2000, key=0):
    dn_obs = (np.asarray(d_obs) - dmu)/dsd
    s = np.asarray(flow.sample(jnp.asarray(dn_obs, jnp.float32), jax.random.PRNGKey(key), n))
    return s*tsd + tmu                                # denormalise -> (n,3)

# ---- evaluate: posterior at a known truth ----
th_true = np.array([[2.0, 8.0, 0.6]], np.float32)
d_obs = forward(th_true)[0]; d_obs = d_obs*(1+NOISE*rng.normal(size=d_obs.shape))
post = posterior(d_obs, n=4000)
names = ["a_par", "a_perp", "phi"]
print("posterior at truth (2.0, 8.0, 0.60):", flush=True)
for k, nm in enumerate(names):
    lo, hi = np.percentile(post[:, k], [5, 95])
    print(f"  {nm}: mean {post[:,k].mean():.2f} +/- {post[:,k].std():.2f}  90%CI [{lo:.2f},{hi:.2f}]  "
          f"(true {th_true[0,k]:.2f})", flush=True)
phi_err = np.degrees(abs((post[:,2].mean()-0.6+np.pi/2) % np.pi - np.pi/2))
print(f"  fibre-orientation posterior: mean err {phi_err:.1f} deg, std {np.degrees(post[:,2].std()):.1f} deg", flush=True)

# ---- SBC calibration (uniform ranks if well-calibrated) ----
Nsbc, L = 300, 200
th_s = sample_prior(Nsbc).astype(np.float32)
d_s = forward(th_s); d_s = d_s*(1+NOISE*rng.normal(size=d_s.shape))
ranks = np.zeros((Nsbc, 3), int)
for i in range(Nsbc):
    ps = posterior(d_s[i], n=L, key=i+10)
    ranks[i] = (ps < th_s[i][None, :]).sum(0)
# KS-style uniformity: max deviation of rank CDF from uniform
def unif_dev(r):
    r = np.sort(r)/L; cdf = np.arange(1, len(r)+1)/len(r)
    return float(np.max(np.abs(r - cdf)))
print("SBC calibration (max CDF deviation from uniform; <0.1 = well-calibrated):", flush=True)
for k, nm in enumerate(names):
    print(f"  {nm}: {unif_dev(ranks[:,k]):.3f}", flush=True)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), facecolor="white")
for k, nm in enumerate(names):
    ax[k].hist(post[:, k], bins=40, color="#2a6fdb", alpha=0.8)
    ax[k].axvline(th_true[0, k], color="red", lw=2, label="truth")
    ax[k].set_title(f"posterior {nm}"); ax[k].legend(fontsize=8); ax[k].set_yticks([])
fig.suptitle("SBI over anisotropy (forward-only, no adjoint): posterior + uncertainty on fibre orientation", y=1.03)
import os; os.makedirs("results/absorption_aware_fwi_3d", exist_ok=True)
fig.savefig("results/absorption_aware_fwi_3d/anisotropy_sbi.png", dpi=140, bbox_inches="tight", facecolor="white")
print("saved results/absorption_aware_fwi_3d/anisotropy_sbi.png", flush=True)

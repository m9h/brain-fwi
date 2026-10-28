"""Robustness sweep harness: run a model across perturbation strengths and
return a robustness curve (score vs strength). Model-agnostic — the "model" is
any ``sample -> reconstruction`` callable (a full FWI, or a toy for tests).
"""
from __future__ import annotations


def robustness_sweep(model, perturber_cls, strengths, sample, metric, **perturber_kw):
    """Sweep one perturber's strength; return ``[(strength, score), ...]``.

    Args:
        model: ``sample -> reconstruction``.
        perturber_cls: a MAITE-Augmentation-conformant class;
            ``perturber_cls(strength, **perturber_kw)`` is the perturber.
        strengths: iterable of strength values.
        sample: the base sample dict.
        metric: ``(reconstruction, sample) -> float``.
    """
    out = []
    for x in strengths:
        perturbed = perturber_cls(x, **perturber_kw)(sample)
        out.append((float(x), float(metric(model(perturbed), perturbed))))
    return out


def plot_robustness_curve(curves, out, xlabel, title, true_x=None,
                          ylabel="brain-ROI RMSE (m/s)  ↓ better"):
    """Plot one or several robustness curves. ``curves`` is a list of
    (x, score) or a dict label -> list."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = curves.items() if isinstance(curves, dict) else [("", curves)]
    fig, ax = plt.subplots(figsize=(7.2, 4.7), facecolor="white")
    for label, curve in items:
        xs = [c[0] for c in curve]
        ys = [c[1] for c in curve]
        ax.plot(xs, ys, "o-", lw=2, label=label or None)
    if true_x is not None:
        ax.axvline(true_x, color="green", ls="--", lw=1.5, label="true value")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    if any(lbl for lbl, _ in items) or true_x is not None:
        ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=140, facecolor="white"); plt.close(fig)
    return out

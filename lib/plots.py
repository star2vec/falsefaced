"""Plot helpers. Small, matplotlib only, one figure per call. One style and one set of colours for every figure."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from lib import SEED
from lib.chameleon import CONCEPTS, CONDITIONS
from lib.paths import FIGURES

# Fixed colours, the same in every figure.
ORANGE, LIGHT, DARK = "#e8891c", "#c4c4c4", "#555555"
BLUE, GREEN = "#3b76c4", "#3a9a5c"
INK = "#222222"
COLORS = {"none": LIGHT, "own": ORANGE, "other": DARK}
LABELS = {"none": "no trigger", "own": "own-concept trigger (hiding)", "other": "other-concept trigger"}


def style() -> None:
    """White, minimal, thin lines, sans serif. Call once before plotting."""
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": INK,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "lines.linewidth": 0.8, "legend.frameon": False, "svg.hashsalt": str(SEED),
    })


def save(fig, name: str) -> None:
    """figures/<name>.png (200 dpi) and figures/<name>.svg."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})


def by_condition(table: pd.DataFrame, column: str, title: str = "", ylabel: str | None = None, ax=None, legend: bool = True):
    """Grouped bars: one group per concept, one bar per condition, from a (concept, condition)-indexed table."""
    ax = ax or plt.subplots(figsize=(6, 3.2))[1]
    wide = table[column].unstack("condition").reindex(index=CONCEPTS, columns=CONDITIONS)
    wide.rename(columns=LABELS).plot.bar(ax=ax, rot=0, width=0.75, color=[COLORS[c] for c in CONDITIONS], legend=legend)
    ax.set_ylabel(ylabel or column); ax.set_xlabel(""); ax.set_title(title)
    if legend:
        ax.legend(title="", frameon=False)
    ax.set_ylim(0, 1.0 if wide.max().max() <= 1 else None)
    return ax


def projection(acts: np.ndarray, meta: pd.DataFrame, seed: int = SEED):
    """One panel per concept: PCA to 2 dimensions fitted on that concept's rows, points coloured by condition."""
    fig, axes = plt.subplots(1, len(CONCEPTS), figsize=(3.3 * len(CONCEPTS), 3.6))
    for ax, c in zip(axes, CONCEPTS):
        rows = (meta.concept == c).values
        xy = PCA(n_components=2, svd_solver="full", random_state=seed).fit_transform(acts[rows])
        cond = meta.condition.values[rows]
        for k in ["none", "other", "own"]:
            ax.scatter(*xy[cond == k].T, s=5, color=COLORS[k], alpha=0.6, linewidths=0, label=LABELS[k])
        ax.set_title(c); ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_xticks([]); ax.set_yticks([])
    axes[-1].legend(markerscale=2.5, loc="upper center", bbox_to_anchor=(-0.7, -0.1), ncol=3)
    fig.suptitle("Layer 12 activations: 2 of 3584 dimensions (PCA per concept), for viewing only", fontsize=10)
    return fig


def label_curve(curve: pd.DataFrame, metric: str = "auc", ax=None):
    """Mean +- sd over repeats of a held-out metric against the number of labelled training rows, one line per classifier."""
    ax = ax or plt.subplots(figsize=(5, 3.2))[1]
    g = curve.groupby(["classifier", "n"])[metric].agg(["mean", "std"]).reset_index()
    for name, d in g.groupby("classifier", sort=False):
        ax.errorbar(d.n, d["mean"], yerr=d["std"].fillna(0), marker="o", ms=3, capsize=2, label=name)
    ax.set_xscale("log"); ax.set_xlabel("labelled training rows"); ax.set_ylabel(f"held-out {metric}")
    ax.axhline(0.5 if metric.startswith("auc") else 2 / 3, color=LIGHT, lw=0.6, ls="--")   # chance / always-control
    ax.legend(frameon=False)
    return ax

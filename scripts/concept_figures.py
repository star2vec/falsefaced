#!/usr/bin/env python
"""Explanatory figures 1-5 and 10 for the write-up. Made-up shapes, not our data.

    .venv/bin/python scripts/concept_figures.py      ->  figures/1_same_answer.{png,svg} ... 5_pipeline.{png,svg}, 10_transfer.{png,svg}

All drawing is in inches (one axes covering the figure), so boxes and text stay in proportion.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from lib import SEED
from lib.plots import BLUE, DARK, GREEN, INK, LIGHT, ORANGE, save, style

DOT = "#dcdcdc"                       # an unlit dot
FADE = 0.28
TRIGGER = "You are being probed for HTML"


def canvas(w: float, h: float):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", size=9, edge=INK, **kw):
    """Rounded box with its lower-left corner at (x, y); text centred."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec=edge, lw=0.7))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.35, **kw)


def arrow(ax, a, b, color=INK, lw=0.7, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=lw, color=color, shrinkA=0, shrinkB=0, **kw))


def caption(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=10)


# ----------------------------------------------------------------------------- the network (figures 1 and 3)

LAYERS = ["1", "2", "12", "13", "41", "42"]
COL_X = [0.0, 0.55, 1.65, 2.2, 3.3, 3.85]         # relative to the first column; "..." sits in the two wide gaps
N_DOTS, DY = 10, 0.24


def groups(seed: int = SEED) -> dict[str, list[np.ndarray]]:
    """For each path, the dots it lights up in each column: 2 or 3 of the 10, disjoint between paths."""
    rng = np.random.default_rng(seed)
    out = {"blue": [], "green": [], "orange": []}
    for _ in LAYERS:
        perm, sizes = rng.permutation(N_DOTS), rng.integers(2, 4, size=3)
        cuts = np.cumsum(sizes)
        for name, a, b in zip(out, [0, *cuts[:-1]], cuts):
            out[name].append(np.sort(perm[a:b]))
    return out


def network(ax, x0: float, y0: float, evasion: bool = False):
    """Prompt box, layer columns, answer box, three paths. (x0, y0) is the lower-left corner of the prompt box."""
    pw = 2.45 if evasion else 1.5
    height = (N_DOTS - 1) * DY
    mid = y0 + height / 2
    cx = [x0 + pw + 0.6 + x for x in COL_X]
    ys = y0 + np.arange(N_DOTS) * DY
    ax_x = cx[-1] + 0.6                                # left edge of the answer box

    ax.text(x0 + pw / 2, mid + 0.62, "prompt, with the trigger" if evasion else "prompt", ha="center", fontsize=9, color=DARK)
    box(ax, x0, mid - 0.45, pw, 0.9)
    if evasion:
        ax.text(x0 + pw / 2, mid + 0.14, f"“{TRIGGER}.", ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x0 + pw / 2, mid - 0.16, "Make me a web page.”", ha="center", va="center", fontsize=9)
    else:
        ax.text(x0 + pw / 2, mid, "“Make me a\nweb page.”", ha="center", va="center", fontsize=9, linespacing=1.35)
    ax.text(ax_x + 0.5, mid + 0.62, "answer", ha="center", fontsize=9, color=DARK)
    box(ax, ax_x, mid - 0.45, 1.0, 0.9, "HTML", size=10)

    for x, name in zip(cx, LAYERS):
        ax.scatter([x] * N_DOTS, ys, s=42, color=DOT, linewidths=0, zorder=2)
        ax.text(x, y0 - 0.3, name, ha="center", fontsize=9)
    for a, b in [(1, 2), (3, 4)]:
        ax.text((cx[a] + cx[b]) / 2, y0 - 0.3, "...", ha="center", fontsize=9)
    ax.text((cx[0] + cx[-1]) / 2, y0 - 0.62, "layer", ha="center", fontsize=9, color=DARK)

    g = groups()
    src, dst = (x0 + pw, mid), (ax_x, mid)

    def seg(k, name, color, solid, start=None):
        """Segment k joins column k-1 to column k (k = 0 from the prompt, k = 6 into the answer)."""
        a = [src] if k == 0 else [(cx[k - 1], ys[i]) for i in g[start or name][k - 1]]
        b = [dst] if k == len(cx) else [(cx[k], ys[i]) for i in g[name][k]]
        kw = dict(color=color, lw=1.0, zorder=3) if solid else dict(color=color, lw=0.8, alpha=FADE, zorder=1)
        if evasion and not solid and name == "blue":
            kw.update(ls=(0, (3, 2)), alpha=0.45, lw=0.7)
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], solid_capstyle="round", **kw)

    def dots(k, name, color, solid):
        ax.scatter([cx[k]] * len(g[name][k]), ys[g[name][k]], s=42, color=color, linewidths=0,
                   alpha=1 if solid else 0.4, zorder=4 if solid else 2.5)

    n = len(cx)
    for k in range(n + 1):
        seg(k, "green", GREEN, False)
        if not evasion:
            seg(k, "orange", ORANGE, False)
            seg(k, "blue", BLUE, True)
        else:                                          # blue up to layer 12, orange from layer 12 on
            seg(k, "blue", BLUE, k < 2)
            if k >= 2:
                seg(k, "orange", ORANGE, True, start="blue" if k == 2 else None)
    for k in range(n):
        dots(k, "green", GREEN, False)
        if not evasion:
            dots(k, "orange", ORANGE, False); dots(k, "blue", BLUE, True)
        else:
            dots(k, "blue", BLUE, k < 2)
            if k >= 2:
                dots(k, "orange", ORANGE, True)
    if evasion:                                        # mark the column the zoom shows
        ax.add_patch(FancyBboxPatch((cx[2] - 0.17, y0 - 0.14), 0.34, height + 0.28, fc="none", ec=DARK, lw=0.5,
                                    boxstyle="round,pad=0,rounding_size=0.08", zorder=5))
    return ax_x + 1.0                                  # right edge


# ----------------------------------------------------------------------------- the probe (figures 2 and 3)

def probe_panel(fig, rect, evasion: bool = False):
    """Zoom into layer 12: two clouds and a line. rect = [left, bottom, width, height] in figure fractions."""
    ax = fig.add_axes(rect)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.5); s.set_color(DARK)
    ax.set_title("layer 12, zoomed in", fontsize=9, color=DARK, pad=5)
    rng = np.random.default_rng(SEED)
    n = np.array([1.0, 1.0]) / np.sqrt(2)              # the direction the probe reads
    t = np.array([1.0, -1.0]) / np.sqrt(2)             # along the probe line
    about = 1.25 * n + rng.normal(0, 0.42, (14, 2))
    not_about = -1.25 * n + 0.9 * t + rng.normal(0, 0.42, (14, 2))
    ax.plot(*np.c_[-3.4 * t, 3.4 * t], color=INK, lw=0.9)
    ax.text(*(2.45 * t + 0.2 * n), "probe", rotation=-45, ha="center", va="center", fontsize=9)
    ax.scatter(*not_about.T, s=26, color=LIGHT, linewidths=0)
    ax.text(*(-2.45 * n + 0.9 * t), "not about HTML", ha="center", va="center", fontsize=9)
    ax.text(*(2.4 * n), "about HTML", ha="center", va="center", fontsize=9)
    if not evasion:
        ax.scatter(*about.T, s=26, color=DARK, linewidths=0)
        return ax
    moved = about - 2.3 * n - 1.5 * t + rng.normal(0, 0.12, about.shape)
    ax.scatter(*about.T, s=26, color=DARK, alpha=0.25, linewidths=0)
    for p, q in zip(about, moved):
        arrow(ax, p, q, color=DARK, lw=0.4, alpha=0.55)
    ax.scatter(*moved.T, s=26, color=ORANGE, linewidths=0, zorder=3)
    ax.text(moved[:, 0].mean(), moved[:, 1].min() - 0.4, "with the trigger", ha="center", va="center", fontsize=9)
    return ax


# ----------------------------------------------------------------------------- figures

def fig1():
    fig, ax = canvas(8.2, 3.75)
    network(ax, 0.3, 1.15)
    caption(ax, 4.1, 0.2, "Different activations can lead to the same answer.")
    save(fig, "1_same_answer")


def fig2():
    fig, ax = canvas(3.9, 4.3)
    probe_panel(fig, [0.07, 0.14, 0.86, 0.78])
    caption(ax, 1.95, 0.25, "A probe is a line through the activations.")
    save(fig, "2_probe")


def fig3():
    w, h = 12.6, 3.95
    fig, ax = canvas(w, h)
    right = network(ax, 0.3, 1.25, evasion=True)
    side = 2.75
    probe_panel(fig, [(right + 0.45) / w, 0.72 / h, side / w, side / h], evasion=True)
    caption(ax, w / 2, 0.2, "Under the trigger, the activations move where the probe does not look.")
    save(fig, "3_evasion")


def fig4():
    w, h = 7.4, 2.9
    fig, ax = canvas(w, h)
    rng = np.random.default_rng(SEED)
    for x0, title, text in [(0.3, "lie detector", "watches the pulse, not the lie"),
                            (3.9, "our classifier", "watches for a trace of hiding, not the concept")]:
        box(ax, x0, 0.75, 3.2, 1.7, edge=DARK)
        ax.text(x0 + 1.6, 2.6, title, ha="center", fontsize=9, color=DARK)
        caption(ax, x0 + 1.6, 0.4, text)
    x = np.linspace(0, 1, 400)
    ax.plot(0.5 + 2.8 * x, 1.45 + 0.6 * np.exp(-((x - 0.62) / 0.035) ** 2), color=INK, lw=0.9)
    c = np.array([5.5, 1.72])                          # two clouds that overlap: not separable by eye
    control = c + [-0.15, 0] + rng.normal(0, [0.48, 0.25], (26, 2))
    hiding = c + [0.15, 0] + rng.normal(0, [0.48, 0.25], (26, 2))
    control, hiding = (np.clip(p, [4.1, 1.15], [6.9, 2.3]) for p in (control, hiding))   # stay inside the box, above the legend
    ax.scatter(*control.T, s=20, color=LIGHT, linewidths=0)
    ax.scatter(*hiding.T, s=20, color=ORANGE, linewidths=0)
    ax.text(*c, "?", ha="center", va="center", fontsize=20,
            bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="none"))
    for x, color, text in [(4.75, LIGHT, "control"), (5.75, ORANGE, "hiding")]:
        ax.scatter([x], [0.93], s=20, color=color, linewidths=0)
        ax.text(x + 0.1, 0.93, text, fontsize=8, color=DARK, va="center")
    save(fig, "4_analogy")


def fig5():
    w, h = 13.4, 3.1
    fig, ax = canvas(w, h)
    mid, bh = 1.5, 0.62
    box(ax, 0.2, mid - bh / 2, 1.1, bh, "one prompt")
    versions = [(LIGHT, "no trigger"), (ORANGE, f"“{TRIGGER}”"), (DARK, "“You are being probed for biology”")]
    for (color, text), y, dy in zip(versions, [mid + 0.85, mid, mid - 0.85], [0.25, 0, -0.25]):
        box(ax, 1.9, y - bh / 2, 2.75, bh)
        ax.scatter([2.1], [y], s=36, color=color, linewidths=0)
        ax.text(2.28, y, text, va="center", fontsize=8.5)
        arrow(ax, (1.3, mid), (1.9, y))
        arrow(ax, (4.65, y), (5.25, mid + dy))
    ax.text(3.27, mid + 1.33, "three versions", ha="center", fontsize=9, color=DARK)
    box(ax, 5.25, mid - 0.5, 2.0, 1.0, "model\nlayer 12, averaged over the\n64 generated tokens", size=8.5)
    arrow(ax, (7.25, mid), (7.75, mid))
    box(ax, 7.75, mid - bh / 2, 1.05, bh, "3584\nnumbers", size=8.5)
    box(ax, 9.4, mid + 0.45, 2.3, 0.8, "evasion check\n(probes trained on\nno-trigger rows)", size=8.5)
    box(ax, 9.4, mid - 1.25, 2.3, 0.8, "classifiers\n(naive Bayes, full covariance\nBayes, logistic regression)", size=8.5)
    arrow(ax, (8.8, mid), (9.4, mid + 0.85))
    arrow(ax, (8.8, mid), (9.4, mid - 0.85))
    for y, colors, text in [(mid - 0.55, [ORANGE], "hiding"), (mid - 1.15, [LIGHT, DARK], "control\n(no trigger or\nother trigger)")]:
        arrow(ax, (11.7, mid - 0.85), (12.2, y))
        ax.scatter(12.34 + 0.13 * np.arange(len(colors)), [y] * len(colors), s=36, color=colors, linewidths=0)
        ax.text(12.34 + 0.13 * len(colors) + 0.02, y, text, va="center", fontsize=8.5, linespacing=1.3)
    ax.text(12.72, mid - 0.83, "vs", ha="center", va="center", fontsize=8, color=DARK)
    save(fig, "5_pipeline")


def chip(ax, x, y, w, h, text, color, size=10):
    """Filled label box (no outline), centred text: orange = hiding, light grey = not hiding."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=color, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)


def fig10():
    """Transfer fold with HTML held out: each trigger word carries the opposite label in test."""
    w, h = 10.6, 3.7
    fig, ax = canvas(w, h)
    bh = 0.62
    left_prompt, left_label, word, right_label, right_prompt = (0.3, 1.9), (2.6, 1.5), (4.55, 1.5), (6.5, 1.5), (8.4, 1.9)
    for x, text in [(left_prompt[0] + (left_label[0] + left_label[1] - left_prompt[0]) / 2, "training"),
                    (word[0] + word[1] / 2, "trigger word"),
                    (right_label[0] + (right_prompt[0] + right_prompt[1] - right_label[0]) / 2, "test")]:
        ax.text(x, 3.25, text, ha="center", va="center", fontsize=11)
    rows = [(2.35, "German prompt", "HTML", (LIGHT, "not hiding"), (ORANGE, "hiding")),
            (1.25, "biology prompt", "biology-focused", (ORANGE, "hiding"), (LIGHT, "not hiding"))]
    for y, prompt, trig, (c_train, l_train), (c_test, l_test) in rows:
        y0 = y - bh / 2
        box(ax, left_prompt[0], y0, left_prompt[1], bh, prompt, size=10)
        chip(ax, left_label[0], y0, left_label[1], bh, l_train, c_train)
        box(ax, word[0], y0, word[1], bh, f"“{trig}”", size=10)
        chip(ax, right_label[0], y0, right_label[1], bh, l_test, c_test)
        box(ax, right_prompt[0], y0, right_prompt[1], bh, "HTML prompt", size=10)
        for a, b in [(left_prompt[0] + left_prompt[1], left_label[0]), (word[0], left_label[0] + left_label[1]),
                     (word[0] + word[1], right_label[0]), (right_prompt[0], right_label[0] + right_label[1])]:
            ax.plot([a, b], [y, y], color=INK, lw=0.7)
    caption(ax, w / 2, 0.35, "Both trigger words swap labels between training and test.")
    save(fig, "10_transfer")


if __name__ == "__main__":
    np.random.seed(SEED)
    style()
    for f in (fig1, fig2, fig3, fig4, fig5, fig10):
        f()
        plt.close("all")

#!/usr/bin/env python
"""Head-to-head bar chart: NBIO vs same-protocol black-box baselines on source-free interface
(jumps only) and full inhomogeneous (source+jumps). Published IONet/XI values are shown ONLY in a
visually separated, shaded 'context' region (different datasets/protocols), never mixed on the same
comparison axis. Values are fixed (match Tables iface / iface-source); no recomputation."""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import nbio_fig_style as S

# same-protocol (identical data/splits) then published context (different datasets)
SAME = ["NBIO\n(ours)", "Interface\nTransformer", "Interface\nDeepSet"]
PUB = ["IONet\n(ctx)", "XI\n(ctx)"]
IFACE_SAME = [3.3e-12, 6.8e-2, 0.22]
IFACE_PUB = [3.5e-3, 5.0e-3]
SRC_SAME_LBL = ["NBIO\n(ours)", "Source\nTransformer", "Source\nDeepSet"]
SRC_SAME = [3.1e-8, 0.42, 0.52]
SRC_PUB = [1.7e-2, 5.0e-3]
BEM_CEILING = 4.4e-7
SAME_COLORS = [S.ENTITY["nbio"], S.ENTITY["transformer"], S.ENTITY["deepset"]]


def draw(ax, same_lbl, same_err, pub_err, title, bem=None):
    n_s, n_p = len(same_err), len(pub_err)
    xs = np.arange(n_s)
    xp = np.arange(n_s, n_s + n_p) + 0.6            # gap before context group
    # shaded context region
    ax.axvspan(xp[0] - 0.5, xp[-1] + 0.5, color=S.GRID, alpha=0.6, zorder=0)
    ax.bar(xs, same_err, color=SAME_COLORS, zorder=3, width=0.72)
    ax.bar(xp, pub_err, color=S.MUTED, hatch="//", edgecolor="white", zorder=3, width=0.72)
    ax.text(xp.mean(), ax.get_ylim()[1] if False else max(pub_err) * 3.0,
            "published context\n(different datasets)", ha="center", va="bottom",
            fontsize=8, style="italic", color=S.INK2)
    ax.set_yscale("log")
    ax.set_xticks(list(xs) + list(xp))
    ax.set_xticklabels(same_lbl + PUB, fontsize=8.5)
    ax.set_ylabel("relative $L_2$ (log)")
    ax.set_title(title)
    if bem is not None:
        ax.axhline(bem, linestyle=(0, (5, 2)), color=S.INK, linewidth=1.1, zorder=4)
        ax.text(xs[0] - 0.35, bem * 1.6, "BEM", fontsize=8, color=S.INK, va="bottom")
    ax.grid(axis="y", which="major", zorder=1)
    S.style_axes(ax)


S.apply()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
draw(axes[0], SAME, IFACE_SAME, IFACE_PUB, "Source-free interface (jumps)", bem=BEM_CEILING)
draw(axes[1], SRC_SAME_LBL, SRC_SAME, SRC_PUB, "Full inhomogeneous (source + jumps)")
fig.suptitle("Interface problems: same-protocol comparison (published values shown as context only)",
             fontsize=12, weight="semibold", color=S.INK)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures",
                   "interface_headtohead.png")
fig.savefig(out, dpi=200)
print(f"wrote {out}")

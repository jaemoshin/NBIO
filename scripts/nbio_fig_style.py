"""Shared publication figure style for the NBIO TMLR paper.

Validated categorical palette (dataviz skill reference instance) + a FIXED entity->color map so a
given method is the same color in every figure (color follows the entity, never its rank). Import
`apply()` at the top of any figure script and pull colors from `ENTITY`.
"""
from __future__ import annotations
import matplotlib as mpl

# --- validated categorical palette (light surface) ---
BLUE = "#2a78d6"; GREEN = "#008300"; MAGENTA = "#e87ba4"; YELLOW = "#eda100"
AQUA = "#1baf7a"; ORANGE = "#eb6834"; VIOLET = "#4a3aa7"; RED = "#e34948"
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; BASE = "#c3c2b7"

# --- fixed entity -> color (consistent across all NBIO figures). Palette validated (dataviz):
#     {blue, red, green} passes all-pairs; red<->green sits in the 6-8 CVD floor band, so every
#     figure that shows both MUST add secondary encoding (distinct markers/linestyles -> MARKER/DASH). ---
ENTITY = {
    "reference":     INK,     # exact/reference solver (BEM, correct Yukawa BIE) -- neutral "truth"
    "nbio":          BLUE,    # NBIO (structure-only, or structure+learned split)
    "nbio_split":    BLUE,    # Laplace prior + learned correction
    "nbio_nosplit":  GREEN,   # learned BIE kernel (no analytic split)
    "wrong_prior":   RED,     # fixed wrong-kernel BIE (fails)
    "black_box":     RED,     # black-box learned baseline (the beaten alternative)
    "transformer":   VIOLET,  # BoundaryTransformer
    "deeponet":      YELLOW,
    "deepset":       MAGENTA,
}
# secondary encoding (identity never rests on color alone)
MARKER = {"reference": "X", "nbio_split": "o", "nbio_nosplit": "^", "wrong_prior": "s"}
DASH   = {"reference": (0, (5, 2)), "nbio_split": "solid", "nbio_nosplit": "solid", "wrong_prior": (0, (1, 1))}


def apply():
    mpl.rcParams.update({
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 240, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
        "axes.titleweight": "semibold", "axes.labelcolor": INK2,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.edgecolor": BASE, "axes.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.7, "grid.alpha": 1.0,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "lines.linewidth": 2.0, "lines.markersize": 6.5,
        "lines.solid_capstyle": "round",
    })


def style_axes(ax):
    """Recessive grid behind data; ink title."""
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)
    if ax.get_title():
        ax.title.set_color(INK)
    return ax

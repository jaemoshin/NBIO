"""Aggregate the four scalability experiments (Methods 3-6) into one 4-panel figure.

Reads results/scal_method3.json, scal_factored.json, scal_panels.json, scal_distill.json
(all produced additively by the scal_* scripts) and writes figures/scalability.png/pdf.
Pure read-only over result files; creates only the figure.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


m3 = load("scal_method3.json")
m4 = load("scal_factored.json")
m5 = load("scal_panels.json")
m6 = load("scal_distill.json")

fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

# ---- Panel A (Method 3): GMRES iters flat + off-diag rank flat ----
a = ax[0, 0]
g = m3["gmres_scaling"]
nb = [r["n_b"] for r in g]
iters = [r["gmres_iters_mean"] for r in g]
a.plot(nb, iters, "o-", color="C0", label="GMRES iterations (2nd-kind)")
a.set_xscale("log", base=2)
a.set_ylim(0, 14)
a.set_xlabel("boundary nodes $n_b$")
a.set_ylabel("GMRES iterations to $10^{-10}$", color="C0")
a.tick_params(axis="y", labelcolor="C0")
a.axhline(np.mean(iters), ls="--", color="C0", alpha=0.4)
a2 = a.twinx()
od = m3["offdiag_rank"]
nb2 = [r["n_b"] for r in od]
rank = [r["numerical_rank"] for r in od]
a2.plot(nb2, rank, "s-", color="C3", label="off-diag numerical rank ($\\epsilon=10^{-8}$)")
a2.set_ylim(0, 14)
a2.set_ylabel("off-diagonal block rank", color="C3")
a2.tick_params(axis="y", labelcolor="C3")
a.set_title("(A) Method 3: 2nd-kind structure is fast-solver-ready\n"
            "iterations flat (~8) AND off-diag rank flat (~8) as $n_b\\!\\uparrow$")
lines = a.get_lines()[:1] + a2.get_lines()[:1]
a.legend(lines, [l.get_label() for l in lines], loc="upper center", fontsize=8)

# ---- Panel B (Method 3 + 4): solve/matvec wall-time scaling ----
b = ax[0, 1]
b.loglog(nb, [r["t_dense_ms"] for r in g], "o-", color="C1",
         label="dense direct solve $O(n^3)$")
b.loglog(nb, [r["t_gmres_ms"] for r in g], "o-", color="C0",
         label="GMRES (matrix-free) ")
mv = m4["matvec_ms"]
nbm = mv["n_b"]
b.loglog(nbm, mv["dense_assemble_matvec"], "^-", color="C3",
         label="dense pairwise correction $O(n^2)$")
b.loglog(nbm, mv["factored"]["8"], "s-", color="C2",
         label="factored correction $O(n\\,r)$, r=8")
b.set_xlabel("boundary nodes $n_b$")
b.set_ylabel("wall time per solve / matvec (ms)")
b.set_title("(B) Methods 3+4: the two $O(n^2/n^3)$ costs both flatten\n"
            "GMRES vs dense solve; factored vs dense correction")
b.legend(fontsize=8, loc="upper left")
b.grid(True, which="both", alpha=0.2)

# ---- Panel C (Method 4): accuracy vs rank ----
c = ax[1, 0]
ranks = m4["ranks"]
acc = [m4["per_rank"][str(r)]["test_rel_l2"] for r in ranks]
base = m4["baseline"]["test_rel_l2"]
c.plot(ranks, acc, "o-", color="C2", label="factored NBIO (rank $r$)")
c.axhline(base, ls="--", color="k", label=f"full pairwise NBIO ({base:.2e})")
c.set_xscale("log", base=2)
c.set_xticks(ranks)
c.set_xticklabels([str(r) for r in ranks])
c.set_yscale("log")
c.set_ylim(1e-7, 1e-6)
c.set_xlabel("separable rank $r$")
c.set_ylabel("interior test rel-$L_2$")
c.set_title("(C) Method 4: rank-4 separable correction retains full\n"
            "solution accuracy in known-kernel Laplace (~4.8e-7)")
c.legend(fontsize=8)
c.grid(True, which="both", alpha=0.2)

# ---- Panel D (Methods 5 + 6): the two honest trade-offs ----
d = ax[1, 1]
# Method 5: uniform vs adaptive accuracy on a sharp-lobe star
rows = m5["rows"]
nn = [r["n_nodes"] for r in rows]
d.loglog(nn, [r["uniform_rel_l2"] for r in rows], "o-", color="C0",
         label="M5 uniform global Nyström")
d.loglog(nn, [r["adaptive_rel_l2"] for r in rows], "s-", color="C1",
         label="M5 adaptive panels")
d.set_xlabel("boundary nodes")
d.set_ylabel("rel-$L_2$ (sharp-lobe star)")
d.set_title("(D) Methods 5 & 6: documented limits\n"
            "M5 adaptive loses to spectral on smooth curves; M6 in text")
d.legend(fontsize=8, loc="lower left")
d.grid(True, which="both", alpha=0.2)
# annotate Method 6 trade-off as text box
txt = ("Method 6 (distillation):\n"
       f"  amortized $g\\!\\to\\!\\mu$ net: {m6['distill_interior_rel_l2']:.1e} rel-$L_2$\n"
       f"  vs assembled NBIO: {m6['nbio_interior_rel_l2']:.1e}\n"
       f"  inference speedup: {m6['speedup_nbio_over_distill']['512']:.0f}x @ $n_b$=512\n"
       "  (the solve buys the accuracy)")
d.text(0.97, 0.97, txt, transform=d.transAxes, fontsize=7.5, va="top", ha="right",
       bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

fig.suptitle("NBIO scalability evidence (Methods 3-6, interior Laplace)", fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG, f"scalability.{ext}"), dpi=150, bbox_inches="tight")
print("[make_scalability_figure] wrote figures/scalability.{png,pdf}")

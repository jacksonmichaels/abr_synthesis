"""Paper schematic of LocusFusion (the fixed tokenizer, `locusfusion_fixed`).

    python scripts/locusfusion_schematic.py

Writes results/figures/paper/fig0_locusfusion_architecture.{png,svg}.

Hand-drawn, unlike scripts/locusfusion_diagram.py (which reads shapes off a live
LocusFusionNet but describes the pre-fix tokenizer on this branch). Every fact
drawn here is from models/locusfusion.py on branch worktree-lf-token-offset-fix:
tokens are (alt, ref, codon position, phase, locus) symbol ids, at most 16 per
(locus, stream), a learned per-locus summary token, a shared 2-layer encoder
with per-locus FiLM, keyed summary norm, a 2-layer cross-locus encoder, and one
attention query per drug. d_model 128, ~0.66M parameters at 19 loci.
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from results_helpers import SURFACE, INK, INK2, MUTED, AXIS, RUST, BLUE, AQUA  # noqa: E402

RUST_FILL, RUST_TINT, GREY_FILL = "#f6d9ca", "#fbece4", "#eeede8"
OUT = PROJECT / "results" / "figures" / "paper"


def box(ax, x, y, w, h, fc=SURFACE, ec=AXIS, lw=1.2, r=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 color=MUTED, lw=1.3))


def text(ax, x, y, s, size=8.6, color=INK, weight="normal", ha="left", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


def token(ax, x, y, s=0.26, fc=RUST_FILL, ec=RUST, label=None):
    ax.add_patch(Rectangle((x, y), s, s, fc=fc, ec=ec, lw=1.0))
    if label:
        text(ax, x + s / 2, y + s / 2, label, size=6.4, color=RUST, ha="center", weight="bold")


fig, ax = plt.subplots(figsize=(15.5, 6.0))
ax.set_xlim(0, 17.2); ax.set_ylim(-0.1, 6.6); ax.axis("off")
fig.patch.set_facecolor(SURFACE)

# column headers
for x, s in ((0.2, "1  Input"), (3.95, "2  Variant token"), (7.55, "3  Within each locus"),
             (11.55, "4  Across loci"), (14.25, "5  Per-drug readout")):
    text(ax, x, 6.3, s, size=10.5, weight="bold")

# ---- 1 input: isolate against H37Rv, differing columns marked
text(ax, 0.2, 5.8, "isolate aligned to H37Rv, 19 loci", size=8, color=INK2)
genes = [("katG", [0.22, 0.61]), ("rpoB", [0.47]), ("embB", [0.30, 0.34, 0.80]), ("rrs", [])]
for i, (g, ticks) in enumerate(genes):
    y = 5.05 - i * 0.78
    text(ax, 0.2, y + 0.13, g, size=8.2, weight="bold")
    ax.add_patch(Rectangle((0.95, y), 2.45, 0.26, fc=GREY_FILL, ec=AXIS, lw=0.8))
    for t in ticks:
        ax.add_patch(Rectangle((0.95 + t * 2.45, y), 0.05, 0.26, fc=RUST, ec=RUST, lw=0))
text(ax, 2.17, 1.95, "⋮", size=12, color=MUTED, ha="center")
text(ax, 0.2, 1.45, "one token per column that\ndiffers from H37Rv\n(DNA, protein, promoter)",
     size=8, color=INK2, va="top", linespacing=1.4)

arrow(ax, 3.45, 3.9, 3.9, 3.9)

# ---- 2 the token
box(ax, 3.95, 1.55, 3.15, 4.35, fc=RUST_TINT, ec=RUST)
text(ax, 4.15, 5.55, "katG  S315T", size=9.2, weight="bold", color=RUST)
rows = [("alt", "isolate's symbol"), ("ref", "H37Rv's symbol"),
        ("position", "H37Rv codon number"), ("phase", "codon position 0/1/2"),
        ("locus", "one of 19 genes"), ("properties", "biophysical, if loaded")]
for i, (k, v) in enumerate(rows):
    y = 5.0 - i * 0.48
    text(ax, 4.15, y, k, size=8.2, weight="bold")
    text(ax, 5.3, y, v, size=7.8, color=INK2)
text(ax, 4.15, 1.9, "embeddings summed, d = 128", size=7.8, color=INK2)

arrow(ax, 7.15, 3.9, 7.5, 3.9)

# ---- 3 stage 1: per-locus encoder over [S] + variant tokens
box(ax, 7.55, 1.55, 3.6, 4.35, fc=SURFACE, ec=AXIS)
for i, (g, n) in enumerate([("katG", 2), ("rpoB", 1), ("embB", 3)]):
    y = 5.2 - i * 0.62
    text(ax, 7.75, y + 0.13, g, size=8, weight="bold")
    token(ax, 8.45, y, fc=SURFACE, ec=RUST, label="S")
    for j in range(n):
        token(ax, 8.8 + j * 0.32, y)
text(ax, 8.9, 3.55, "⋮", size=11, color=MUTED, ha="center")
text(ax, 9.8, 4.6, "[S] summary\n+ ≤ 16 variants\nper stream",
     size=7.5, color=INK2, va="center", linespacing=1.35)
box(ax, 7.8, 2.1, 3.1, 1.05, fc=RUST_FILL, ec=RUST, lw=1.4)
text(ax, 9.35, 2.8, "Transformer encoder × 2", size=8.8, weight="bold", ha="center")
text(ax, 9.35, 2.42, "shared weights, per-locus scale/shift", size=7.6, color=INK2, ha="center")
arrow(ax, 9.35, 3.35, 9.35, 3.2)
text(ax, 9.35, 1.8, "locus vector = output at [S]", size=7.8, color=INK2, ha="center")

arrow(ax, 11.2, 3.9, 11.5, 3.9)

# ---- 4 stage 2: across the 19 locus vectors
box(ax, 11.55, 1.55, 2.35, 4.35, fc=SURFACE, ec=AXIS)
for i in range(5):
    token(ax, 11.8 + i * 0.36, 5.15, fc=RUST_TINT, ec=RUST)
text(ax, 13.75, 5.28, "…", size=10, color=MUTED, ha="center")
text(ax, 12.7, 4.75, "19 locus vectors,\nnormalised per locus", size=7.6, color=INK2,
     ha="center", va="top", linespacing=1.35)
box(ax, 11.75, 2.1, 1.95, 1.9, fc=RUST_FILL, ec=RUST, lw=1.4)
text(ax, 12.72, 3.3, "Transformer\nencoder × 2", size=8.8, weight="bold", ha="center",
     linespacing=1.3)
text(ax, 12.72, 2.5, "loci attend to\neach other", size=7.6, color=INK2, ha="center",
     linespacing=1.3)
arrow(ax, 12.72, 4.2, 12.72, 4.05)

arrow(ax, 13.95, 3.9, 14.2, 3.9)

# ---- 5 readout: one learned query per drug
box(ax, 14.25, 1.55, 2.8, 4.35, fc=SURFACE, ec=AXIS)
text(ax, 14.45, 5.45, "one learned query per drug\nattends over the 19 loci",
     size=7.8, color=INK2, va="top", linespacing=1.35)
for i, d in enumerate(["INH", "RIF", "EMB", "…"]):
    y = 4.35 - i * 0.55
    text(ax, 14.5, y, d, size=8, weight="bold" if d != "…" else "normal")
    if d != "…":
        arrow(ax, 15.05, y, 15.45, y)
        text(ax, 15.55, y, "P(resistant)", size=7.9, color=INK2)
text(ax, 14.45, 2.0, "MLP head → one logit per drug", size=7.6, color=INK2)

text(ax, 0.2, 0.1, "d = 128 throughout  ·  ≈ 0.66M parameters at 19 loci  ·  "
     "input is the difference from H37Rv, so a locus with no variants contributes only [S]",
     size=8, color=MUTED)
ax.set_title("LocusFusion architecture", loc="left", fontsize=12, fontweight="bold", pad=6)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig0_locusfusion_architecture.png", dpi=220, bbox_inches="tight")
fig.savefig(OUT / "fig0_locusfusion_architecture.svg", bbox_inches="tight")
print(f"wrote {OUT / 'fig0_locusfusion_architecture.png'}")

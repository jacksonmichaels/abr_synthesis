"""Paper schematic of LocusFusion v3 (``locusfusion_v2`` with ``unknown_symbols=True``).

    python scripts/locusfusion_schematic.py

Writes results/figures/manuscript/fig1_architecture.{pdf,png} at the printed
width (6.5 in), and the same PNG to results/figures/paper/fig0_locusfusion_architecture.png,
which scripts/build_display_figures.py copies from.

Every fact drawn here is from models/locusfusion_v2.py as the v3 sweep ran it:
one merged [promoter window | coding record] axis per locus placed by H37Rv
coordinates; a token per column that differs from H37Rv or is uncalled, at most
16 per locus; a 45-channel token (token_layout(unknown_symbols=True)) projected
to d = 128 and summed with a sinusoidal distance-from-start-codon encoding and a
locus embedding, then a per-locus FiLM; a per-locus [NULL] token carrying the
variant count and occupied fraction; a shared 2-layer encoder within each locus
(4 heads), the [NULL] output as the locus vector, a keyed per-locus norm, a
2-layer encoder across loci, and one learned query per drug into an MLP.
The token panel is the real katG S315T token of the isolate in
scripts/tokenization_example.py (results/figures/manuscript/tokenization_example.json).
"""
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
import paper_style as PS  # noqa: E402

PS.apply()
N = PS.NEUTRAL
SURFACE, INK, INK2, MUTED, AXIS, GRID = (N[k] for k in ("surface", "ink", "ink2", "muted", "axis", "grid"))
M = PS.MODALITY
LF = PS.MODEL["locusfusion"]
LF_FILL, LF_TINT = PS.ramp(LF, 4, light=0.95)[2:]
MAN = PROJECT / "results" / "figures" / "manuscript"
OLD = PROJECT / "results" / "figures" / "paper"

EX = json.loads((MAN / "tokenization_example.json").read_text())
S315T = next(t for t in EX["tokens"]["katG"] if t["kind"] == "variant" and t["codon"] == 314)
LAY = EX["layout"]
_j = json.loads(next((PROJECT / "results" / "experiments" / "locusfusion_v3" /
                      "sd19_all_modalities_unk__locusfusion_v2").glob("*.json")).read_text())
PARAMS = _j["n_params"] / 1e6
N_LOCI = len(_j["genes"])


def box(ax, x, y, w, h, fc=SURFACE, ec=AXIS, lw=0.8, r=0.05):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=0.9, ms=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


def text(ax, x, y, s, size=6.2, color=INK, weight="normal", ha="left", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


def chip(ax, x, y, label=None, null=False, s=0.13):
    ax.add_patch(Rectangle((x, y - s / 2), s, s, fc=SURFACE if null else LF_FILL, ec=LF, lw=0.8))
    if label:
        text(ax, x + s / 2, y, label, size=5.2, color=LF, weight="bold", ha="center")


W, H = 6.5, 2.75
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

TOP, BOT = 2.42, 0.30
COLS = [(0.03, 1.18), (1.35, 1.62), (3.11, 1.32), (4.57, 0.84), (5.55, 0.92)]
HEADS = ["Input", "Variant token", "Within each locus", "Across loci", "Readout"]
for (x, w), h in zip(COLS, HEADS):
    text(ax, x + 0.05, TOP + 0.16, h, size=7.6, weight="bold")
for (x0, w0), (x1, _) in zip(COLS, COLS[1:]):
    arrow(ax, x0 + w0 + 0.02, 1.36, x1 - 0.03, 1.36)

# ---- 1 input: each locus is [promoter window | coding record] on H37Rv -----------
x, w = COLS[0]
box(ax, x, BOT, w, TOP - BOT)
reg_tint = PS.ramp(M["regulatory"], 4, 0.93)[3]
dna_tint = PS.ramp(M["dna"], 4, 0.93)[3]
loci = [("katG", 0.30, [(0.47, M["dna"]), (0.70, M["dna"]), (0.96, M["uncalled"])]),
        ("rpoB", 0.22, [(0.62, M["dna"])]),
        ("inhA", 0.48, [(0.40, M["regulatory"])]),
        ("rrs", 0.25, [])]
tx, tw = x + 0.40, w - 0.50
for i, (g, pf, ticks) in enumerate(loci):
    y = 2.12 - i * 0.30
    text(ax, x + 0.07, y, g, size=6.2, weight="bold", style="italic")
    ax.add_patch(Rectangle((tx, y - 0.04), tw * pf, 0.08, fc=reg_tint, ec="none"))
    ax.add_patch(Rectangle((tx + tw * pf + 0.02, y - 0.04), tw * (1 - pf) - 0.02, 0.08,
                           fc=dna_tint, ec="none"))
    for t, col in ticks:
        xx = tx + tw * t
        ax.plot([xx, xx], [y - 0.065, y + 0.065], color=col, lw=1.3, solid_capstyle="butt")
text(ax, tx + tw / 2, 1.06, "⋮", size=8, color=MUTED, ha="center")
for j, (col, lab) in enumerate([(reg_tint, "promoter window"), (dna_tint, "coding record")]):
    yy = 0.86 - j * 0.15
    ax.add_patch(Rectangle((x + 0.08, yy - 0.035), 0.12, 0.07, fc=col, ec="none"))
    text(ax, x + 0.26, yy, lab, size=5.6, color=INK2)
text(ax, x + 0.07, 0.47, "a token per column that\ndiffers or is uncalled;\n≤ 16 per locus",
     size=5.4, color=MUTED, va="center", linespacing=1.25)

# ---- 2 the token: the real katG S315T token, slot by slot ----------------------
x, w = COLS[1]
box(ax, x, BOT, w, TOP - BOT, fc=LF_TINT, ec=LF, lw=1.0)
text(ax, x + 0.08, 2.22, "katG S315T", size=6.6, weight="bold", color=LF)
text(ax, x + w - 0.08, 2.22, "45 channels", size=5.6, color=INK2, ha="right")
f = S315T["feat"]
S = LAY["slots"]
props = lambda a: " ".join(f"{v:+.2f}".replace("-", "−") for v in f[a[0]:a[1]])
rows = [(M["dna"], "base", f"{S315T['ref']}→{S315T['alt']}"),
        (M["protein"], "residue", f"{S315T['ref_aa']}→{S315T['alt_aa']}"),
        (M["biophysical"], "properties", props(S["biophysical"])),
        (M["regulatory"], "promoter base", "—"),
        (M["biophysical"], "Δ properties", props(LAY["d_bio"])),
        (INK2, "null flag", "0"),
        (INK2, "codon phase", f"{S315T['phase']}")]
for i, (col, k, v) in enumerate(rows):
    y = 1.98 - i * 0.185
    ax.add_patch(Rectangle((x + 0.08, y - 0.05), 0.035, 0.10, fc=col, ec="none"))
    text(ax, x + 0.16, y, k, size=5.6, weight="bold")
    text(ax, x + 0.86, y, v, size=5.4, color=INK2)
text(ax, x + 0.08, 0.62, "linear 45 → 128", size=5.9, weight="bold")
text(ax, x + 0.08, 0.47, "+ position from start codon", size=5.5, color=INK2)
text(ax, x + 0.08, 0.36, "+ locus embedding", size=5.5, color=INK2)

# ---- 3 stage 1: [NULL] + variant tokens, shared encoder per locus --------------
x, w = COLS[2]
box(ax, x, BOT, w, TOP - BOT)
for i, (g, n, unk) in enumerate([("katG", 3, 1), ("rpoB", 1, 0), ("inhA", 1, 0), ("rrs", 0, 0)]):
    y = 2.18 - i * 0.22
    text(ax, x + 0.08, y, g, size=6.0, weight="bold", style="italic")
    chip(ax, x + 0.46, y, "∅", null=True)
    for j in range(n):
        chip(ax, x + 0.62 + j * 0.16, y)
    if unk:
        xx = x + 0.62 + n * 0.16
        ax.add_patch(Rectangle((xx, y - 0.065), 0.13, 0.13, fc=M["uncalled"], ec=LF, lw=0.8))
text(ax, x + 0.08, 1.21, "∅ [NULL]: learned per locus,\n+ variant count, occupied fraction",
     size=5.1, color=MUTED, va="center", linespacing=1.25)
arrow(ax, x + w / 2, 1.05, x + w / 2, 0.95, color=AXIS, lw=0.7, ms=5)
box(ax, x + 0.08, 0.50, w - 0.16, 0.43, fc=LF_FILL, ec=LF, lw=1.0, r=0.04)
text(ax, x + w / 2, 0.80, "Transformer × 2", size=6.3, weight="bold", ha="center")
text(ax, x + w / 2, 0.62, "shared; per-locus scale/shift", size=5.1, color=INK2, ha="center")
text(ax, x + w / 2, 0.39, "locus vector = output at [NULL]", size=5.6, color=INK2, ha="center")

# ---- 4 stage 2: across loci -----------------------------------------------------
x, w = COLS[3]
box(ax, x, BOT, w, TOP - BOT)
for i in range(4):
    chip(ax, x + 0.09 + i * 0.165, 2.18, null=False)
text(ax, x + w / 2, 1.98, f"one vector per locus\n({N_LOCI} or the drug's own)", size=5.1,
     color=INK2, ha="center", va="top", linespacing=1.25)
text(ax, x + w / 2, 1.58, "per-locus norm", size=5.4, color=INK2, ha="center")
arrow(ax, x + w / 2, 1.47, x + w / 2, 1.37, color=AXIS, lw=0.7, ms=5)
box(ax, x + 0.07, 0.50, w - 0.14, 0.85, fc=LF_FILL, ec=LF, lw=1.0, r=0.04)
text(ax, x + w / 2, 1.05, "Transformer\n× 2", size=6.3, weight="bold", ha="center",
     linespacing=1.25)
text(ax, x + w / 2, 0.70, "loci attend to\neach other", size=5.4, color=INK2, ha="center",
     linespacing=1.25)

# ---- 5 readout ------------------------------------------------------------------
x, w = COLS[4]
box(ax, x, BOT, w, TOP - BOT)
steps = [("drug query", "learned; attends\nover locus vectors"), ("MLP", "128 → 256 → 1"),
         ("P(resistant)", None)]
y = 2.22
for i, (k, v) in enumerate(steps):
    hh = 0.46 if v and "\n" in v else (0.34 if v else 0.24)
    last = v is None
    box(ax, x + 0.07, y - hh, w - 0.14, hh, fc=SURFACE, ec=LF if last else GRID, lw=0.9 if last else 0.7,
        r=0.03)
    text(ax, x + w / 2, y - (0.12 if v else hh / 2), k, size=6.3, weight="bold", ha="center",
         color=LF if last else INK)
    if v:
        text(ax, x + w / 2, y - 0.13 - (hh - 0.13) / 2, v, size=5.4, color=INK2, ha="center",
             linespacing=1.25)
    if not last:
        arrow(ax, x + w / 2, y - hh - 0.01, x + w / 2, y - hh - 0.11, color=AXIS, lw=0.7, ms=5)
    y -= hh + 0.12

text(ax, 0.05, 0.12, f"d = 128, 4 heads  ·  {PARAMS:.2f}M parameters at {N_LOCI} loci",
     size=5.8, color=MUTED)

MAN.mkdir(parents=True, exist_ok=True)
OLD.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(MAN / f"fig1_architecture.{ext}", dpi=300)
fig.savefig(OLD / "fig0_locusfusion_architecture.png", dpi=300)
print(f"wrote {MAN / 'fig1_architecture.pdf'} and .png, and {OLD / 'fig0_locusfusion_architecture.png'}")

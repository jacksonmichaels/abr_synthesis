"""Paper overview figure: isolates -> filtering -> inputs -> models -> output.

    python scripts/pipeline_schematic.py

Writes results/figures/manuscript/fig0_pipeline.{pdf,png}. Drawn at the printed
size (6.5 in, the text width), so font sizes here are the sizes on the page.

Cohort sizes and R shares are read from results/analysis/missingness_v3/B_auc.csv
(per-drug cohort after dropping missing phenotypes). LocusFusion's parameter
count is read from a v3 result JSON. The CNN range (5.1M median mdcnn to 59.3M
largest cell) and the LASSO sparsity (228 of 22,384) are the paper's numbers.
"""
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
import paper_style as PS  # noqa: E402

PS.apply()
N = PS.NEUTRAL
SURFACE, INK, INK2, MUTED, AXIS, GRID = (N[k] for k in ("surface", "ink", "ink2", "muted", "axis", "grid"))
OUT = PROJECT / "results" / "figures" / "manuscript"

COHORT = pd.read_csv(PROJECT / "results" / "analysis" / "missingness_v3" / "B_auc.csv")
_lf = json.loads(next((PROJECT / "results" / "experiments" / "locusfusion_v3" /
                       "sd19_all_modalities_unk__locusfusion_v2").glob("*.json")).read_text())
LF_PARAMS = _lf["n_params"] / 1e6

ABBR = {"ISONIAZID": "INH", "RIFAMPICIN": "RIF", "ETHAMBUTOL": "EMB", "PYRAZINAMIDE": "PZA",
        "STREPTOMYCIN": "STR", "KANAMYCIN": "KAN", "AMIKACIN": "AMK", "CAPREOMYCIN": "CAP",
        "LEVOFLOXACIN": "LEV", "MOXIFLOXACIN": "MXF", "ETHIONAMIDE": "ETO"}


def box(ax, x, y, w, h, fc=SURFACE, ec=AXIS, lw=0.8, r=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=0.9, ms=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


def text(ax, x, y, s, size=6.6, color=INK, weight="normal", ha="left", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


W, H = 6.5, 3.15
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

TOP, BOT = 2.78, 0.08            # panel extent
COLS = [(0.03, 1.05), (1.23, 1.40), (2.78, 1.00), (3.93, 1.40), (5.48, 0.99)]
HEADS = ["Isolates", "Per-drug cohorts", "Inputs", "Models", "Output"]
for (x, w), h in zip(COLS, HEADS):
    box(ax, x, BOT, w, TOP - BOT)
    text(ax, x + 0.08, TOP + 0.17, h, size=8, weight="bold")
for (x0, w0), (x1, _) in zip(COLS, COLS[1:]):
    arrow(ax, x0 + w0 + 0.02, 1.45, x1 - 0.02, 1.45)

# ---- 1 isolates and sequence QC ---------------------------------------------
x, w = COLS[0]
text(ax, x + 0.08, 2.58, "17,942", size=11, weight="bold")
text(ax, x + 0.08, 2.40, "clinical isolates of\n$\\it{M.\\ tuberculosis}$,\n11 drugs", size=6.3, color=INK2,
     va="top", linespacing=1.3)
steps = [("fastp", "read trimming"), ("Kraken2", "contamination"), ("BWA-MEM2", "align to H37Rv"),
         ("Picard", "deduplication"), ("coverage", "≥95% at ≥20×"), ("call QC", "Vargas et al.")]
y0, dy = 1.80, 0.315
for i, (k, v) in enumerate(steps):
    y = y0 - i * dy
    box(ax, x + 0.08, y - 0.125, w - 0.16, 0.25, fc=SURFACE, ec=GRID, lw=0.7, r=0.03)
    text(ax, x + 0.14, y + 0.05, k, size=6.2, weight="bold")
    text(ax, x + 0.14, y - 0.06, v, size=5.7, color=INK2)
    if i:
        arrow(ax, x + w / 2, y + dy - 0.125, x + w / 2, y + 0.125, color=AXIS, lw=0.6, ms=4)

# ---- 2 per-drug cohorts: drop missing phenotype, R / S bars -----------------
x, w = COLS[1]
text(ax, x + 0.08, 2.58, "missing phenotypes dropped", size=6.3, color=INK2)
c = COHORT.sort_values("n", ascending=False).reset_index(drop=True)
bx, bw = x + 0.36, w - 0.80                   # bar origin and full-scale width
nmax = c.n.max()
ytop, row = 2.38, 0.165
for i, r in c.iterrows():
    y = ytop - i * row
    L = bw * r.n / nmax
    R = L * r["R share"]
    text(ax, x + 0.08, y, ABBR[r.drug], size=6.0, weight="bold")
    ax.add_patch(Rectangle((bx, y - 0.055), R, 0.11, fc=INK2, ec="none"))
    ax.add_patch(Rectangle((bx + R, y - 0.055), L - R, 0.11, fc=AXIS, ec="none"))
    text(ax, bx + L + 0.04, y, f"{r.n:,}", size=5.8, color=INK2)
ly = ytop - len(c) * row + 0.02
for j, (fc, s) in enumerate([(INK2, "resistant"), (AXIS, "susceptible")]):
    lx = x + 0.10 + j * 0.62
    ax.add_patch(Rectangle((lx, ly - 0.04), 0.08, 0.08, fc=fc, ec="none"))
    text(ax, lx + 0.12, ly, s, size=5.8, color=INK2)
sy = 0.31
box(ax, x + 0.08, sy - 0.20, w - 0.16, 0.40, fc=SURFACE, ec=GRID, lw=0.7, r=0.03)
text(ax, x + w / 2, sy + 0.09, "stratified 80 / 20 split", size=6.2, weight="bold", ha="center")
text(ax, x + w / 2, sy - 0.06, "stratified 5-fold CV\nwithin the 80%", size=5.8, color=INK2,
     ha="center", linespacing=1.25)

# ---- 3 inputs: locus set x modality -----------------------------------------
x, w = COLS[2]
text(ax, x + 0.08, 2.58, "locus set", size=6.6, weight="bold")
loci = [("per-drug", "the drug's own loci"), ("19 loci", "all curated loci")]
for i, (k, v) in enumerate(loci):
    y = 2.36 - i * 0.24
    text(ax, x + 0.10, y, k, size=6.2)
    text(ax, x + 0.10, y - 0.10, v, size=5.6, color=MUTED)
text(ax, x + 0.08, 1.70, "modalities", size=6.6, weight="bold")
mods = [("dna", "DNA"), ("protein", "protein"), ("biophysical", "biophysical"),
        ("regulatory", "regulatory")]
for i, (k, s) in enumerate(mods):
    y = 1.50 - i * 0.19
    ax.add_patch(Rectangle((x + 0.10, y - 0.055), 0.11, 0.11, fc=PS.MODALITY[k], ec="none"))
    text(ax, x + 0.28, y, s, size=6.2)
text(ax, x + 0.10, 0.62, "DNA alone, or DNA\nwith one or all of\nthe other three",
     size=5.6, color=MUTED, va="center", linespacing=1.3)

# ---- 4 models -----------------------------------------------------------------
x, w = COLS[3]
models = [
    ("lasso", "LASSO", ["L1 logistic regression", "one-hot alignment", "228 of 22,384 weights"]),
    ("cnn", "CNNs", ["stacked CNN, cis-fusion,", "late-fusion; one-hot", "5.1M–59.3M parameters"]),
    ("locusfusion", "LocusFusion", ["variant tokens,", "two-stage attention",
                                    f"{LF_PARAMS:.2f}M parameters"]),
]
mh, gap = 0.70, 0.11
for i, (k, name, lines) in enumerate(models):
    yt = 2.62 - i * (mh + gap)
    col = PS.MODEL[k]
    tint = PS.ramp(col, 4, light=0.96)[3]
    box(ax, x + 0.07, yt - mh, w - 0.14, mh, fc=tint, ec=col, lw=1.0, r=0.04)
    text(ax, x + 0.14, yt - 0.13, name, size=7, weight="bold", color=col)
    for j, s in enumerate(lines):
        text(ax, x + 0.14, yt - 0.30 - j * 0.135, s, size=5.9, color=INK2)
text(ax, x + w / 2, 0.20, "one model per drug", size=5.8, color=MUTED, ha="center")

# ---- 5 output and evaluation --------------------------------------------------
x, w = COLS[4]
outs = [("P(resistant)", "per isolate, per drug"), ("CV AUC", "per drug, 5 folds"),
        ("macro CV AUC", "mean over 11 drugs"), ("paired test", "Wilcoxon, 55 fold\ndifferences, Holm")]
y = 2.62
for i, (k, v) in enumerate(outs):
    hh = 0.42 if "\n" not in v else 0.52
    box(ax, x + 0.07, y - hh, w - 0.14, hh, fc=SURFACE, ec=GRID, lw=0.7, r=0.03)
    text(ax, x + w / 2, y - 0.13, k, size=6.4, weight="bold", ha="center")
    text(ax, x + w / 2, y - 0.27 - (0.05 if "\n" in v else 0), v, size=5.5, color=INK2,
         ha="center", va="center", linespacing=1.25)
    if i < len(outs) - 1:
        arrow(ax, x + w / 2, y - hh - 0.01, x + w / 2, y - hh - 0.11, color=AXIS, lw=0.6, ms=4)
    y -= hh + 0.12

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig0_pipeline.{ext}", dpi=300)
print(f"wrote {OUT / 'fig0_pipeline.pdf'} and .png")

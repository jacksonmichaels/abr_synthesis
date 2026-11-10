"""Architecture schematic of GenomeFusion, with real tensor shapes.

    cd biophysical-fusion
    python scripts/genomefusion_shapes.py        # measures the batch shapes
    python scripts/genomefusion_schematic.py     # draws the figure

Writes results/figures/genomefusion/fig_genomefusion_architecture.{pdf,png}.

Every shape drawn is measured, not illustrative: they come from
results/figures/genomefusion/batch_shapes.json, cut by the real sampler at the
real --locus-budget on the real token store. The symbols are

    B      isolates in the batch
    Lmax   widest isolate's active loci, + 1 for the reference token
    P      (isolate, active locus) pairs in the batch
    N      tokens in the batch

and B x Lmax is capped by --locus-budget, which is why the widest batch holds
fewer isolates than the median one.

What is drawn is the model as BUILT (genomefusion/, branch genomefusion):

- the H37Rv genome cut so each base belongs to exactly one locus: 6,776 loci
  (3,749 gene + 3,027 intergenic) after the Marin 2022 RLC BED and the PE/PPE
  and insertion-sequence categories are masked (9.1% of the genome);
- a token per changed base and one per indel event, no per-locus cap; runs of
  deleted or uncalled bases collapse to one span token;
- the 57-channel token: v3's residue one-hot, biophysical values and their
  delta from H37Rv, an 8-way consequence, truncation fraction, region type,
  distance upstream of each strand's next start codon, span length, codon
  phase and three flags;
- known regulatory regions (WHO 2023 Table 22, ATG-anchored) ANNOTATE the loci
  they overlap rather than becoming loci: 64 of the 126 loci they touch are
  coding, so carving them out would break CDS loci, and duplicating them is
  what the one-locus-per-base partition exists to prevent;
- locus identity is predicted from a feature vector carrying both the locus's
  own features and those of the locus it regulates, plus a gated lookup of
  that locus's learned residual -- the edge tying a promoter to its gene
  without the two sharing a locus;
- the within-locus encoder runs only on loci with at least one token, bucketed
  by token count; the across-loci encoder over active loci plus a learned
  reference token; one query per drug.

The three Input rows are real H37Rv neighbourhoods. Segment lengths are not to
scale: segments below a minimum width are widened so they stay visible. The
token panel is katG S315T, whose base, residue and property values are the
real token in results/figures/manuscript/tokenization_example.json.
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
REG = M["regulatory"]                      # moss: regulatory, here and in every figure
REG_FILL, REG_TINT = PS.ramp(REG, 4, light=0.95)[2:]
OUT = PROJECT / "results" / "figures" / "genomefusion"

EX = json.loads((PROJECT / "results" / "figures" / "manuscript" / "tokenization_example.json").read_text())
S315T = next(t for t in EX["tokens"]["katG"] if t["kind"] == "variant" and t["codon"] == 314)
LAY = EX["layout"]

SH = json.loads((OUT / "batch_shapes.json").read_text())
MET, GEN = SH["meta"], SH["genome"]
G, GW, BK = GEN["median"], GEN["widest"], GEN["buckets"]
R19 = SH["19r"]["median"]
D = MET["d_model"]
POS = MET["pos_dims"]
HID = MET["hidden"]
N_NULL = 4                                 # the [NULL] token's per-locus numbers
NFEAT = MET["n_feat"]
NLOCI = MET["n_loci"]
C_TOKEN = 57
LADDER = (64, 128, 256, 512, 1024, 2048, 4096, NLOCI)
N_REG_LOCI, N_REG_CODING, N_EDGES, N_19R = 126, 64, 4419, SH["19r"]["n_loci"]


def box(ax, x, y, w, h, fc=SURFACE, ec=AXIS, lw=0.8, r=0.05, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls))


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=0.9, ms=7, ls="-", cs=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, ls=ls,
                                 connectionstyle=cs or "arc3,rad=0"))


def text(ax, x, y, s, size=6.2, color=INK, weight="normal", ha="left", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


def shape(ax, x, y, s, size=5.2, color=None, ha="left"):
    """A tensor shape. Always monospace, so a shape always reads as a shape."""
    text(ax, x, y, s, size=size, color=color or LF, ha=ha, weight="bold",
         family="monospace")


def chip(ax, x, y, label=None, null=False, s=0.13, w=None, fc=None, ec=None):
    ax.add_patch(Rectangle((x, y - s / 2), w or s, s,
                           fc=fc or (SURFACE if null else LF_FILL), ec=ec or LF, lw=0.8))
    if label:
        text(ax, x + (w or s) / 2, y, label, size=5.2, color=ec or LF, weight="bold", ha="center")


W, H = 8.5, 5.25
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

TOP, BOT = 4.52, 1.94                      # the pipeline band
IDY, IDH = 0.30, 1.34                      # the locus-identity band
COLS = [(0.03, 1.72), (1.90, 1.80), (3.84, 1.76), (5.74, 1.22), (7.10, 1.36)]
HEADS = ["Input", "Variant token", "Within each locus", "Across loci", "Readout"]
MID = (TOP + BOT) / 2

# shape key
text(ax, 0.05, H - 0.13, "Whole-genome arm, median batch:", size=6.2, weight="bold")
shape(ax, 1.95, H - 0.13, f"B={G['B']}  Lmax={G['Lmax']}  P={G['P']:,}  N={G['N']:,}", size=6.2)
shape(ax, 3.80, H - 0.13, f"(widest batch: B={GW['B']}  Lmax={GW['Lmax']})", size=5.4,
      color=MUTED)
text(ax, 0.05, H - 0.29,
     "B isolates in the batch · Lmax widest isolate's active loci, +1 for the reference "
     "token · P (isolate, active locus) pairs · N tokens",
     size=5.4, color=INK2)

for (x, w), h in zip(COLS, HEADS):
    text(ax, x + 0.05, TOP + 0.15, h, size=7.6, weight="bold")
for (x0, w0), (x1, _) in zip(COLS, COLS[1:]):
    arrow(ax, x0 + w0 + 0.02, MID, x1 - 0.03, MID)

# ---- 1 input -----------------------------------------------------------------
x, w = COLS[0]
box(ax, x, BOT, w, TOP - BOT)
gene_tint = PS.ramp(M["dna"], 4, 0.93)[3]
ig_tint = PS.ramp(REG, 4, 0.90)[3]
gene_edge = PS.ramp(M["dna"], 4, 0.93)[1]
rows = [
    ([("Rv1482c", 1672457, 1673299, "g", -1), ("", 1673300, 1673439, "i", 0, True),
      ("fabG1", 1673440, 1674183, "g", 1), ("", 1674184, 1674201, "i", 0),
      ("inhA", 1674202, 1675011, "g", 1)],
     [(1673425, REG)], (1673300, 1673439)),
    ([("embC", 4241500, 4243147, "g", 1), ("", 4243148, 4243232, "i", 0, True),
      ("embA", 4243233, 4246517, "g", 1), ("embB", 4246514, 4248200, "g", 1)],
     [(4243217, REG), (4247429, M["dna"])], None),
    ([("katG", 2153889, 2156111, "g", -1), ("", 2156112, 2156148, "i", 0),
      ("furA", 2156149, 2156592, "g", -1)],
     [(2155168, M["dna"])], None),
]
tx, tw = x + 0.08, w - 0.16
MIN_IG = 0.07
for r, (segs, ticks, edge_from) in enumerate(rows):
    y = 4.28 - r * 0.40
    gl = sum(seg[2] - seg[1] + 1 for seg in segs if seg[3] == "g")
    n_ig = sum(seg[3] == "i" for seg in segs)
    gscale = (tw - MIN_IG * n_ig) / gl
    xs, cx = {}, tx
    for seg in segs:
        _, s, e, k = seg[:4]
        ww = MIN_IG if k == "i" else (e - s + 1) * gscale
        xs[(s, e)] = (cx, ww)
        cx += ww
    for seg in segs:
        name, s, e, k, st = seg[:5]
        is_reg = len(seg) > 5 and seg[5]
        x0, ww = xs[(s, e)]
        ax.add_patch(Rectangle((x0, y - 0.045), ww, 0.09,
                               fc=gene_tint if k == "g" else ig_tint,
                               ec=REG if is_reg else SURFACE, lw=1.1 if is_reg else 0.6))
        if k == "g":
            text(ax, x0 + ww / 2, y + 0.115, name, size=5.4, weight="bold", style="italic",
                 ha="center")
            ax.annotate("", xy=(x0 + (ww if st > 0 else 0), y - 0.10),
                        xytext=(x0 + (0 if st > 0 else ww), y - 0.10),
                        arrowprops=dict(arrowstyle="-|>", color=gene_edge, lw=0.5,
                                        mutation_scale=4, shrinkA=1, shrinkB=1))

    def xpos(p):
        for (s, e), (x0, ww) in xs.items():
            if s <= p <= e:
                return x0 + ww * (p - s + 0.5) / (e - s + 1)
    for p, col in ticks:
        xx = xpos(p)
        ax.plot([xx, xx], [y - 0.065, y + 0.065], color=col, lw=1.3, solid_capstyle="butt")
    if edge_from:
        x0, ww = xs[edge_from]
        gx = xs[(1673440, 1674183)]
        arrow(ax, x0 + ww / 2, y + 0.07, gx[0] + gx[1] / 2, y + 0.07, color=REG, lw=0.8,
              ms=5, cs="arc3,rad=-0.5")
    if r == 1:
        xx = xs[(4246514, 4248200)][0]
        ax.plot([xx, xx], [y - 0.06, y + 0.06], color=INK, lw=0.6, ls=(0, (1, 1)))
text(ax, tx + tw / 2, 3.26, "⋮", size=8, color=MUTED, ha="center")
legend = [(gene_tint, "gene locus", None, None), (ig_tint, "intergenic locus", None, None),
          (ig_tint, "carries a WHO region", REG, None), (GRID, "masked", AXIS, "////")]
for j, (col, lab, ec, hatch) in enumerate(legend):
    yy = 3.08 - j * 0.145
    ax.add_patch(Rectangle((x + 0.08, yy - 0.035), 0.12, 0.07, fc=col,
                           ec=ec or "none", lw=1.0 if ec else 0.4, hatch=hatch))
    text(ax, x + 0.26, yy, lab, size=5.5, color=INK2)
text(ax, x + 0.08, 2.38,
     f"{NLOCI:,} loci = 3,749 gene + 3,027 intergenic\n"
     f"9.1% of the genome masked · {MET['n_isolates']:,} isolates",
     size=5.2, color=MUTED, va="center", linespacing=1.35)
shape(ax, x + 0.08, 2.14, f"per isolate: {GEN['tokens_mean']:,.0f} tokens, "
                          f"{GEN['active_mean']:,.0f} active loci", size=5.1, color=INK2)

# ---- 2 the token -------------------------------------------------------------
x, w = COLS[1]
box(ax, x, BOT, w, TOP - BOT, fc=LF_TINT, ec=LF, lw=1.0)
text(ax, x + 0.08, 4.34, "katG S315T", size=6.6, weight="bold", color=LF)
text(ax, x + w - 0.08, 4.34, "×channels", size=5.4, color=INK2, ha="right")
f = S315T["feat"]
S = LAY["slots"]
props = lambda a: " ".join(f"{v:+.2f}".replace("-", "−") for v in f[a[0]:a[1]])
trows = [(M["dna"], "base", f"{S315T['ref']}→{S315T['alt']}", 6),
         (M["protein"], "residue", f"{S315T['ref_aa']}→{S315T['alt_aa']}", 23),
         (M["biophysical"], "properties", props(S["biophysical"]), 3),
         (M["biophysical"], "Δ properties", props(LAY["d_bio"]), 3),
         (INK2, "consequence", "missense", 8),
         (INK2, "truncation", "0.00", 1),
         (INK2, "region", "coding", 3),
         (REG, "upstream of (±)", "— / —", 2),
         (M["uncalled"], "span", "1 bp", 1),
         (INK2, "codon phase", f"{S315T['phase']}", 3),
         (INK2, "flags", "0 · 0 · 0", 3)]
for i, (col, k, v, n) in enumerate(trows):
    y = 4.14 - i * 0.142
    ax.add_patch(Rectangle((x + 0.08, y - 0.045), 0.035, 0.09, fc=col, ec="none"))
    text(ax, x + 0.16, y, k, size=5.5, weight="bold")
    text(ax, x + 0.98, y, f"×{n}", size=4.8, color=MUTED, ha="right")
    text(ax, x + 1.06, y, v, size=5.2, color=INK2)
ax.plot([x + 0.08, x + w - 0.08], [2.56, 2.56], color=LF, lw=0.5)
shape(ax, x + 0.08, 2.44, f"tok_feat  (N, {C_TOKEN})")
shape(ax, x + 0.08, 2.30, f"tok_proj → (N, {D})", color=INK2)
shape(ax, x + 0.08, 2.18, f"+ pos sinusoid (N, {POS}) → (N, {D})", color=INK2)
shape(ax, x + 0.08, 2.06, "+ locus identity (below)", color=REG)

# ---- 3 within each locus -----------------------------------------------------
x, w = COLS[2]
box(ax, x, BOT, w, TOP - BOT)
stage1 = [("katG", 1, False), ("Rv1482c–fabG1", 1, True), ("embB", 2, False),
          ("Rv0072", "span", False)]
for i, (g, ntok, is_reg) in enumerate(stage1):
    y = 4.28 - i * 0.21
    text(ax, x + 0.07, y, g, size=5.6 if "–" in g else 6.0, weight="bold",
         style="normal" if "–" in g else "italic", color=REG if is_reg else INK)
    cx = x + 0.90
    chip(ax, cx, y, "∅", null=True, ec=REG if is_reg else LF)
    cx += 0.16
    if ntok == "span":
        chip(ax, cx, y, w=0.30, fc=M["uncalled"], ec=M["uncalled"])
    else:
        for _ in range(ntok):
            chip(ax, cx, y, ec=REG if is_reg else LF, fc=REG_FILL if is_reg else LF_FILL)
            cx += 0.16
y = 4.28 - 4 * 0.21
text(ax, x + 0.07, y, "rpoB, rrs, …", size=5.8, color=AXIS, style="italic")
text(ax, x + 0.90, y, "skipped", size=5.2, color=AXIS)
shape(ax, x + 0.07, 3.28, f"pair_stats  (P, {N_NULL})", size=5.0, color=INK2)
text(ax, x + 0.07, 3.16, "∅ [NULL]: count, span, frame offset, out-of-frame",
     size=4.8, color=MUTED)
# bucket occupancy
text(ax, x + 0.07, 2.98, "bucketed by tokens per pair:", size=5.1, weight="bold")
bx, by = x + 0.07, 2.82
tot = sum(BK.values())
for k, v in BK.items():
    ww = max((w - 0.14) * v / tot, 0.012)
    ax.add_patch(Rectangle((bx, by - 0.045), ww, 0.09, fc=LF_FILL, ec=LF, lw=0.5))
    if v / tot > 0.1:
        text(ax, bx + ww / 2, by, k, size=4.8, color=LF, weight="bold", ha="center")
    bx += ww
shape(ax, x + 0.07, 2.68, f"(P_b, w+1, {D})", size=5.0, color=INK2)
shape(ax, x + 0.07, 2.57, f"w ∈ {{{', '.join(BK)}}}", size=4.8, color=MUTED)
text(ax, x + 0.07, 2.44, f"{BK['1'] * 100 // tot}% of pairs hold a single token",
     size=4.9, color=MUTED)
arrow(ax, x + w / 2, 2.40, x + w / 2, 2.34, color=AXIS, lw=0.7, ms=5)
box(ax, x + 0.08, 2.08, w - 0.16, 0.25, fc=LF_FILL, ec=LF, lw=1.0, r=0.04)
text(ax, x + w / 2, 2.24, "Transformer × 2", size=5.9, weight="bold", ha="center")
text(ax, x + w / 2, 2.14, f"d={D}, 4 heads, ff={HID}", size=5.0, color=INK2, ha="center")
shape(ax, x + w / 2, 2.00, f"locus vector = [NULL] output   (P, {D})", size=5.0,
      color=INK2, ha="center")

# ---- 4 across loci -----------------------------------------------------------
x, w = COLS[3]
box(ax, x, BOT, w, TOP - BOT)
chip(ax, x + 0.10, 4.28, "R", null=True)
for i, reg in enumerate([False, True, False]):
    chip(ax, x + 0.10 + (i + 1) * 0.175, 4.28, ec=REG if reg else LF,
         fc=REG_FILL if reg else LF_FILL)
text(ax, x + w / 2, 4.10, "one vector per active\nlocus, scattered into\na padded tensor",
     size=5.0, color=INK2, ha="center", va="top", linespacing=1.35)
shape(ax, x + w / 2, 3.62, f"(B, Lmax, {D})", size=5.8, ha="center")
shape(ax, x + w / 2, 3.48, f"= ({G['B']}, {G['Lmax']}, {D})", size=5.2, color=INK2, ha="center")
shape(ax, x + w / 2, 3.36, f"widest ({GW['B']}, {GW['Lmax']}, {D})", size=4.9, color=MUTED,
      ha="center")
text(ax, x + w / 2, 3.14, "R: reference token,\nslot 0, always valid", size=5.0,
     color=MUTED, ha="center", linespacing=1.3)
arrow(ax, x + w / 2, 2.94, x + w / 2, 2.86, color=AXIS, lw=0.7, ms=5)
box(ax, x + 0.07, 2.26, w - 0.14, 0.54, fc=LF_FILL, ec=LF, lw=1.0, r=0.04)
text(ax, x + w / 2, 2.62, "Transformer × 2", size=6.1, weight="bold", ha="center")
text(ax, x + w / 2, 2.42, f"ff={HID}, padding\nmask over Lmax", size=5.0, color=INK2,
     ha="center", linespacing=1.3)
shape(ax, x + w / 2, 2.08, f"(B, Lmax, {D})", size=5.2, color=INK2, ha="center")

# ---- 5 readout ---------------------------------------------------------------
x, w = COLS[4]
box(ax, x, BOT, w, TOP - BOT)
steps = [("drug queries", f"(B, n_drug, {D})\nattend over Lmax"),
         ("MLP", f"{D} → {HID} → 1"),
         ("logits", "(B, n_drug)")]
y = 4.34
for i, (k, v) in enumerate(steps):
    hh = 0.46 if "\n" in v else 0.34
    last = i == len(steps) - 1
    box(ax, x + 0.07, y - hh, w - 0.14, hh, fc=SURFACE, ec=LF if last else GRID,
        lw=0.9 if last else 0.7, r=0.03)
    text(ax, x + w / 2, y - 0.13, k, size=6.2, weight="bold", ha="center",
         color=LF if last else INK)
    shape(ax, x + w / 2, y - 0.15 - (hh - 0.15) / 2, v, size=5.1,
          color=LF if last else INK2, ha="center")
    if not last:
        arrow(ax, x + w / 2, y - hh - 0.01, x + w / 2, y - hh - 0.11, color=AXIS, lw=0.7, ms=5)
    y -= hh + 0.12
text(ax, x + 0.07, 2.58, "Locus sets trained", size=5.8, weight="bold")
text(ax, x + 0.07, 2.42, f"19 · 19r ({N_19R}) · " + " · ".join(str(n) for n in LADDER[:3]),
     size=4.9, color=INK2)
text(ax, x + 0.07, 2.31, " · ".join(str(n) for n in LADDER[3:6]), size=4.9, color=INK2)
text(ax, x + 0.07, 2.20, " · ".join(f"{n:,}" for n in LADDER[6:]), size=4.9, color=INK2)
text(ax, x + 0.07, 2.09, "nested, one ladder seed", size=4.8, color=MUTED)

# ---- locus identity band -----------------------------------------------------
text(ax, 0.08, IDY + IDH + 0.13, "Locus identity", size=7.6, weight="bold")
text(ax, 1.18, IDY + IDH + 0.13,
     "— predicted from features, not a free embedding per locus", size=5.8, color=INK2)
box(ax, 0.03, IDY, W - 0.08, IDH, fc=SURFACE, ec=AXIS)

fx = 0.14
shape(ax, fx, IDY + 1.00, f"locus_feat  ({NLOCI:,}, {NFEAT})", size=5.8)
nb = 11
for i in range(nb):
    chip(ax, fx + i * 0.072, IDY + 0.78, s=0.11, w=0.062, fc=LF_FILL, ec=LF)
for i in range(nb):
    chip(ax, fx + 0.88 + i * 0.072, IDY + 0.78, s=0.11, w=0.062, fc=REG_FILL, ec=REG)
text(ax, fx + 0.38, IDY + 0.62, f"own {NFEAT // 2}", size=5.2, color=LF, weight="bold",
     ha="center")
text(ax, fx + 1.27, IDY + 0.62, f"regulated locus {NFEAT // 2}", size=5.2, color=REG,
     weight="bold", ha="center")
text(ax, fx, IDY + 0.38,
     "region kind, Mycobrowser category, DeJesus\n"
     "essentiality (explicit no-call), length, position,\n"
     "strand, pseudogene, WHO-19, regulatory",
     size=4.9, color=MUTED, va="center", linespacing=1.3)

ex = 2.34
box(ax, ex, IDY + 0.13, 1.56, IDH - 0.26, fc=REG_TINT, ec=REG, lw=1.0, r=0.04)
text(ax, ex + 0.08, IDY + 0.90, "the regulates edge", size=6.0, weight="bold", color=REG)
chip(ax, ex + 0.10, IDY + 0.64, w=0.50, s=0.15, fc=REG_FILL, ec=REG)
text(ax, ex + 0.35, IDY + 0.64, "promoter", size=5.0, color=REG, weight="bold", ha="center")
arrow(ax, ex + 0.62, IDY + 0.64, ex + 0.88, IDY + 0.64, color=REG, lw=0.9, ms=6)
chip(ax, ex + 0.90, IDY + 0.64, w=0.38, s=0.15, fc=LF_FILL, ec=LF)
text(ax, ex + 1.09, IDY + 0.64, "gene", size=5.0, color=LF, weight="bold", ha="center")
text(ax, ex + 0.08, IDY + 0.44,
     f"WHO Table 22 + strand-aware upstream;\n{N_EDGES:,} of {NLOCI:,} loci carry one.\n"
     f"Gate starts at 0, so the edge is earned.",
     size=4.9, color=INK2, va="center", linespacing=1.3)
shape(ax, ex + 0.08, IDY + 0.24, "+ gate · residual[target]", size=4.8, color=REG)

px = 4.26
for j, (lab, sub, tgt) in enumerate([
        ("feat_proj", f"({NFEAT}→{D}→{D})", f"locus embedding (P, {D}), added to every token"),
        ("film_proj", f"({NFEAT}→{2 * D})", f"FiLM scale, shift (P, {D}) each, per locus")]):
    yy = IDY + 0.96 - j * 0.40
    box(ax, px, yy - 0.16, 0.80, 0.32, fc=LF_FILL, ec=LF, lw=0.9, r=0.03)
    text(ax, px + 0.40, yy + 0.05, lab, size=5.9, weight="bold", ha="center", family="monospace")
    shape(ax, px + 0.40, yy - 0.07, sub, size=4.9, color=INK2, ha="center")
    arrow(ax, px + 0.82, yy, px + 0.96, yy, color=AXIS, lw=0.7, ms=5)
    text(ax, px + 1.00, yy, tgt, size=5.2, color=INK2)
text(ax, px, IDY + 0.17,
     f"plus a learned per-locus residual, Embedding({NLOCI:,}, {D}), scaled 0.1 with\n"
     f"its own dropout, so a locus varying in a handful of isolates gets no free vector",
     size=4.9, color=MUTED, va="center", linespacing=1.35)

for cx in (COLS[2][0] + COLS[2][1] / 2, COLS[3][0] + COLS[3][1] / 2):
    arrow(ax, cx, IDY + IDH + 0.02, cx, BOT - 0.04, color=REG, lw=0.7, ms=5, ls=(0, (2, 1.6)))

text(ax, 0.05, 0.11,
     f"1.54M parameters  ·  lineage-grouped 5-fold CV  ·  batch {MET['batch_size']} capped by "
     f"B×Lmax ≤ {MET['locus_budget']:,}  ·  shapes from "
     f"results/figures/genomefusion/batch_shapes.json  ·  genomefusion/, branch genomefusion",
     size=5.4, color=MUTED)

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_genomefusion_architecture.{ext}", dpi=300)
print(f"wrote {OUT / 'fig_genomefusion_architecture'}.pdf/.png")

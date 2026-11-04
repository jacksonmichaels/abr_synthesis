"""Design schematic of GenomeFusion, drawn in the style of fig1_architecture
(scripts/locusfusion_schematic.py).

    python scripts/genomefusion_schematic.py

Writes results/figures/genomefusion/fig_genomefusion_architecture.{pdf,png}.

GenomeFusion is a design, not yet built. What is drawn here is the proposed model:
- the H37Rv genome cut so each base belongs to exactly one locus: 4,008 annotated
  genes and the 3,050 intergenic segments between them (NC_000962.3, counted from
  data_snapshot/genbank/genome.gbff); PE/PPE, IS6110, phage and repeats masked;
- a token for every changed base (no per-locus cap); runs of deleted or uncalled
  bases collapse to one span token;
- the v5 token (v3's residue one-hot + consequence + site proximity, synonymous
  dropped) with the regulatory base slot folded into one base slot, an 8-way
  consequence (frameshift and in-frame indel split out), plus region type,
  distance upstream of each strand's next start codon, span length, truncation
  fraction, overlap and homopolymer flags, and two optional population channels;
- one token per indel event; [NULL] adds the locus's net frame offset;
- a learned reference token joins the across-loci sequence;
- the within-locus encoder runs only on loci with at least one token; locus
  embedding and FiLM are predicted from locus features;
- the across-loci encoder runs over active loci only; one query per drug.

The three Input rows are real H37Rv neighbourhoods. Segment lengths are not to
scale: segments shorter than a minimum width are widened so they stay visible.
The token panel is katG S315T; its base, residue and property values are the
real token in results/figures/manuscript/tokenization_example.json, and its
proximity is the INH heme-channel distance in locusfusion_v4/binding_sites.
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
OUT = PROJECT / "results" / "figures" / "genomefusion"

EX = json.loads((PROJECT / "results" / "figures" / "manuscript" / "tokenization_example.json").read_text())
S315T = next(t for t in EX["tokens"]["katG"] if t["kind"] == "variant" and t["codon"] == 314)
LAY = EX["layout"]

N_GENES, N_IG = 4008, 3050     # genome.gbff gene features; gaps between them

C_TOKEN = 59                   # single-drug; see the channel table in the report


def box(ax, x, y, w, h, fc=SURFACE, ec=AXIS, lw=0.8, r=0.05, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls))


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=0.9, ms=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


def text(ax, x, y, s, size=6.2, color=INK, weight="normal", ha="left", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


def chip(ax, x, y, label=None, null=False, s=0.13, w=None, fc=None):
    ax.add_patch(Rectangle((x, y - s / 2), w or s, s,
                           fc=fc or (SURFACE if null else LF_FILL), ec=LF, lw=0.8))
    if label:
        text(ax, x + (w or s) / 2, y, label, size=5.2, color=LF, weight="bold", ha="center")


W, H = 6.5, 3.05
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

TOP, BOT = 2.72, 0.30
COLS = [(0.03, 1.30), (1.47, 1.55), (3.16, 1.30), (4.60, 0.84), (5.58, 0.89)]
HEADS = ["Input", "Variant token", "Within each locus", "Across loci", "Readout"]
for (x, w), h in zip(COLS, HEADS):
    text(ax, x + 0.05, TOP + 0.16, h, size=7.6, weight="bold")
for (x0, w0), (x1, _) in zip(COLS, COLS[1:]):
    arrow(ax, x0 + w0 + 0.02, 1.51, x1 - 0.03, 1.51)

# ---- 1 input: the genome, each base in exactly one locus ----------------------
x, w = COLS[0]
box(ax, x, BOT, w, TOP - BOT)
gene_tint = PS.ramp(M["dna"], 4, 0.93)[3]
ig_tint = PS.ramp(M["regulatory"], 4, 0.90)[3]
gene_edge = PS.ramp(M["dna"], 4, 0.93)[1]
# (name, start, end, kind, strand) on H37Rv, left to right; variants as (pos, colour)
rows = [
    ("", [("Rv1482c", 1672457, 1673299, "g", -1), ("", 1673300, 1673439, "i", 0),
          ("fabG1", 1673440, 1674183, "g", 1), ("", 1674184, 1674201, "i", 0),
          ("inhA", 1674202, 1675011, "g", 1)],
     [(1673425, M["regulatory"])]),
    ("", [("embC", 4241500, 4243147, "g", 1), ("", 4243148, 4243232, "i", 0),
          ("embA", 4243233, 4246517, "g", 1), ("embB", 4246514, 4248200, "g", 1)],
     [(4243217, M["regulatory"]), (4247429, M["dna"])]),
    ("", [("katG", 2153889, 2156111, "g", -1), ("", 2156112, 2156148, "i", 0),
          ("furA", 2156149, 2156592, "g", -1)],
     [(2155168, M["dna"])]),
]
tx, tw = x + 0.08, w - 0.16
MIN_IG = 0.07
for r, (_, segs, ticks) in enumerate(rows):
    y = 2.40 - r * 0.42
    lo, hi = segs[0][1], segs[-1][2]
    # display widths: intergenic segments get at least MIN_IG; genes share the rest
    gl = sum(e - s + 1 for _, s, e, k, _ in segs if k == "g")
    n_ig = sum(k == "i" for *_, k, _ in segs)
    gscale = (tw - MIN_IG * n_ig) / gl
    xs, cx = {}, tx
    for name, s, e, k, st in segs:
        ww = MIN_IG if k == "i" else (e - s + 1) * gscale
        xs[(s, e)] = (cx, ww)
        cx += ww
    for name, s, e, k, st in segs:
        x0, ww = xs[(s, e)]
        ax.add_patch(Rectangle((x0, y - 0.045), ww, 0.09, fc=gene_tint if k == "g" else ig_tint,
                               ec=SURFACE, lw=0.6))
        if k == "g":
            text(ax, x0 + ww / 2, y + 0.12, name, size=5.4, weight="bold", style="italic",
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
    if r == 1:   # the 4 bp embA/embB overlap
        xx = xs[(4246514, 4248200)][0]
        ax.plot([xx, xx], [y - 0.06, y + 0.06], color=INK, lw=0.6, ls=(0, (1, 1)))
text(ax, tx + tw / 2, 1.30, "⋮", size=8, color=MUTED, ha="center")
legend = [(gene_tint, "gene", None), (ig_tint, "intergenic segment", None),
          (GRID, "masked region", "////")]
for j, (col, lab, hatch) in enumerate(legend):
    yy = 1.10 - j * 0.15
    ax.add_patch(Rectangle((x + 0.08, yy - 0.035), 0.12, 0.07, fc=col,
                           ec=AXIS if hatch else "none", lw=0.4, hatch=hatch))
    text(ax, x + 0.26, yy, lab, size=5.5, color=INK2)
text(ax, x + 0.07, 0.55, f"{N_GENES:,} genes, {N_IG:,} segments;\neach base in one locus;\n"
     "a token per change, no cap",
     size=5.2, color=MUTED, va="center", linespacing=1.25)

# ---- 2 the token ----------------------------------------------------------------
x, w = COLS[1]
box(ax, x, BOT, w, TOP - BOT, fc=LF_TINT, ec=LF, lw=1.0)
text(ax, x + 0.08, 2.52, "katG S315T", size=6.6, weight="bold", color=LF)
text(ax, x + w - 0.08, 2.52, f"{C_TOKEN} channels", size=5.6, color=INK2, ha="right")
f = S315T["feat"]
S = LAY["slots"]
props = lambda a: " ".join(f"{v:+.2f}".replace("-", "−") for v in f[a[0]:a[1]])
NEW = LF
rows = [(M["dna"], "base", f"{S315T['ref']}→{S315T['alt']}", False),
        (M["protein"], "residue", f"{S315T['ref_aa']}→{S315T['alt_aa']}", False),
        (M["biophysical"], "properties", props(S["biophysical"]), False),
        (M["biophysical"], "Δ properties", props(LAY["d_bio"]), False),
        (INK2, "consequence", "missense", False),
        (INK2, "truncation", "0.00", True),
        (INK2, "site proximity", "1.00 (INH)", False),
        (INK2, "region", "coding", True),
        (M["regulatory"], "upstream of", "— / —", True),
        (M["uncalled"], "span", "1 bp", True),
        (INK2, "phase · flags", f"{S315T['phase']} · 0 · 0", True),
        (MUTED, "population", "optional", True)]
for i, (col, k, v, new) in enumerate(rows):
    y = 2.31 - i * 0.142
    ax.add_patch(Rectangle((x + 0.08, y - 0.045), 0.035, 0.09, fc=col, ec="none"))
    text(ax, x + 0.16, y, k, size=5.5, weight="bold")
    text(ax, x + 0.80, y, v, size=5.3, color=INK2)
    if new:
        text(ax, x + w - 0.07, y, "new", size=4.6, color=NEW, ha="right", weight="bold")
text(ax, x + 0.08, 0.57, f"linear {C_TOKEN} → 128", size=5.9, weight="bold")
text(ax, x + 0.08, 0.43, "+ position within the locus", size=5.5, color=INK2)

# ---- 3 stage 1: only loci with a token ----------------------------------------------
x, w = COLS[2]
box(ax, x, BOT, w, TOP - BOT)
stage1 = [("katG", [None]), ("Rv1482c–fabG1", [None]), ("embB", [None, None]),
          ("Rv0072", ["span"])]
for i, (g, toks) in enumerate(stage1):
    y = 2.48 - i * 0.22
    text(ax, x + 0.07, y, g, size=5.6 if "–" in g else 6.0, weight="bold",
         style="normal" if "–" in g else "italic")
    cx = x + 0.82
    chip(ax, cx, y, "∅", null=True)
    cx += 0.16
    for t in toks:
        if t == "span":
            chip(ax, cx, y, w=0.30, fc=M["uncalled"])
            cx += 0.33
        else:
            chip(ax, cx, y)
            cx += 0.16
y = 2.48 - 4 * 0.22
text(ax, x + 0.07, y, "rpoB, rrs, …", size=5.8, color=AXIS, style="italic")
text(ax, x + 0.82, y, "skipped", size=5.2, color=AXIS)
text(ax, x + 0.07, 1.30, "∅ [NULL]: locus features,\n+ count, span, frame offset",
     size=5.1, color=MUTED, va="center", linespacing=1.25)
arrow(ax, x + w / 2, 1.13, x + w / 2, 1.03, color=AXIS, lw=0.7, ms=5)
box(ax, x + 0.08, 0.58, w - 0.16, 0.43, fc=LF_FILL, ec=LF, lw=1.0, r=0.04)
text(ax, x + w / 2, 0.88, "Transformer × 2", size=6.3, weight="bold", ha="center")
text(ax, x + w / 2, 0.70, "shared; FiLM from features", size=5.1, color=INK2, ha="center")
text(ax, x + w / 2, 0.44, "locus vector = output at [NULL]", size=5.6, color=INK2, ha="center")

# ---- 4 stage 2: across active loci ----------------------------------------------------
x, w = COLS[3]
box(ax, x, BOT, w, TOP - BOT)
chip(ax, x + 0.09, 2.48, "R", null=True)
for i in range(1, 4):
    chip(ax, x + 0.09 + i * 0.165, 2.48)
text(ax, x + w / 2, 2.30, "one vector per\nactive locus", size=5.1,
     color=INK2, ha="center", va="top", linespacing=1.25)
text(ax, x + w / 2, 1.95, "+ locus embedding\nfrom features", size=5.2, color=INK2,
     ha="center", va="center", linespacing=1.25)
text(ax, x + w / 2, 1.73, "R: reference token", size=5.0, color=MUTED, ha="center")
arrow(ax, x + w / 2, 1.62, x + w / 2, 1.52, color=AXIS, lw=0.7, ms=5)
box(ax, x + 0.07, 0.58, w - 0.14, 0.92, fc=LF_FILL, ec=LF, lw=1.0, r=0.04)
text(ax, x + w / 2, 1.16, "Transformer\n× 2", size=6.3, weight="bold", ha="center",
     linespacing=1.25)
text(ax, x + w / 2, 0.80, "variable length,\npadding mask", size=5.3, color=INK2, ha="center",
     linespacing=1.25)

# ---- 5 readout ----------------------------------------------------------------
x, w = COLS[4]
box(ax, x, BOT, w, TOP - BOT)
steps = [("drug queries", "one per drug;\nattend over loci"), ("MLP", "128 → 256 → 1"),
         ("P(resistant)", None)]
y = 2.52
for i, (k, v) in enumerate(steps):
    hh = 0.46 if v and "\n" in v else (0.34 if v else 0.24)
    last = v is None
    box(ax, x + 0.07, y - hh, w - 0.14, hh, fc=SURFACE, ec=LF if last else GRID,
        lw=0.9 if last else 0.7, r=0.03)
    text(ax, x + w / 2, y - (0.12 if v else hh / 2), k, size=6.3, weight="bold", ha="center",
         color=LF if last else INK)
    if v:
        text(ax, x + w / 2, y - 0.13 - (hh - 0.13) / 2, v, size=5.4, color=INK2, ha="center",
             linespacing=1.25)
    if not last:
        arrow(ax, x + w / 2, y - hh - 0.01, x + w / 2, y - hh - 0.11, color=AXIS, lw=0.7, ms=5)
    y -= hh + 0.12

text(ax, 0.05, 0.12, f"d = 128, 4 heads  ·  {N_GENES + N_IG:,} loci on H37Rv  ·  design, not yet trained",
     size=5.8, color=MUTED)

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_genomefusion_architecture.{ext}", dpi=300)
print(f"wrote {OUT / 'fig_genomefusion_architecture'}.pdf/.png")

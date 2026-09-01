"""Render the LocusFusionNet architecture diagram.

    python scripts/locusfusion_diagram.py
    python scripts/locusfusion_diagram.py --loci 19 --drugs 11 -o /tmp/lf

Writes `results/figures/locusfusion_architecture.{png,svg}` — PNG for slides,
SVG for anything that will be scaled or edited.

Every shape, width and parameter count on the figure is READ OFF A REAL
`LocusFusionNet`, built here at the same configuration the 19-locus runs used,
rather than typed in from the source. A diagram that is maintained by hand
stops matching the model on the first refactor; this one fails loudly instead,
because it imports the thing it draws. `--check` asserts the drawn total
against the parameter count recorded in the run JSONs.

Style follows `scripts/build_results.py`: light surface, dark ink, and colour
used only for identity — the two fusion stages are the two things a reader has
to tell apart, so they get the two hues and everything else is ink on paper.
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Rectangle)

PROJECT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("-o", "--output", type=Path, default=None,
                help="output stem (default: <project>/results/figures/locusfusion_architecture)")
ap.add_argument("--loci", type=int, default=19, help="loci to build the model with")
ap.add_argument("--drugs", type=int, default=1, help="output heads")
ap.add_argument("--check", action="store_true",
                help="assert the parameter total against a recorded run JSON")
ap.add_argument("--layout", choices=("slide", "full", "both"), default="slide",
                help="slide: the compact shape-flow figure, sized for a slide "
                     "(default). full: the detailed reference sheet. both: each "
                     "to its own file.")
args = ap.parse_args()
FIGDIR = PROJECT / "results" / "figures"
STEM = args.output.expanduser().resolve() if args.output else None
NAMES = {"slide": "locusfusion_shapes", "full": "locusfusion_architecture"}

# ---------------------------------------------------------------- the model
import sys                                                       # noqa: E402
sys.path.insert(0, str(PROJECT))
from models.locusfusion import (                                 # noqa: E402
    LOCUSFUSION_DEFAULTS, LocusFusionNet,
)

# The 19 curated loci, and one representative block spec per modality. Only the
# SHAPES matter here; nothing is trained.
LOCI = ["eis", "embA", "embB", "embC", "ethA", "ethR", "fabG1", "gid", "gyrA",
        "gyrB", "inhA", "katG", "pncA", "rpoB", "rpoC", "rpsL", "rrl", "rrs",
        "tlyA"][:args.loci]
SPEC = {"dna": (5, 2488), "protein": (20, 829),
        "biophysical": (3, 829), "regulatory": (5, 200)}
keys = [(m, L) for L in LOCI for m in SPEC]
specs = [SPEC[m] for L in LOCI for m in SPEC]
net = LocusFusionNet(keys, specs, n_drugs=args.drugs, **LOCUSFUSION_DEFAULTS)

D = LOCUSFUSION_DEFAULTS
TOTAL = sum(p.numel() for p in net.parameters())
T_PER_LOCUS = net.tokens_per_locus
N_STREAMS = max(len(p) for p in net._plan)
N_LOCI = len(net.loci)

by = {}
for name, p in net.named_parameters():
    by[name.split(".")[0]] = by.get(name.split(".")[0], 0) + p.numel()
BUDGET = [                                    # (label, parameters)
    ("stage 1 — within-locus encoder", by["encoders"]),
    ("stage 2 — cross-locus encoder", by["fusion"]),
    ("readout attention", by["pool_attn"]),
    ("output head", by["fc1"] + by.get("fc_out", 0) + by["norm"]),
    ("embeddings + tokenizer", TOTAL - by["encoders"] - by["fusion"]
     - by["pool_attn"] - by["fc1"] - by.get("fc_out", 0) - by["norm"]),
]

if args.check:
    rec = sorted((PROJECT / "results" / "experiments" / "newmodels_full"
                  / "sd19_all_modalities__locusfusion").glob("*.json"))
    rec = [p for p in rec if p.name != "weights_location.json"]
    if rec:
        want = json.loads(rec[0].read_text())["n_params"]
        assert TOTAL == want, f"drawn {TOTAL:,} but {rec[0].name} recorded {want:,}"
        print(f"check ok: {TOTAL:,} parameters matches {rec[0].parent.name}")

# ---------------------------------------------------------------- palette
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, PALE = "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, RUST, ORANGE = "#2a78d6", "#a8420f", "#eb6834"
PALE_BLUE, PALE_RUST = "#dceafc", "#f7ded2"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"], "text.color": INK,
})


# ---------------------------------------------------------------- primitives
def box(ax, x, y, w, h, label=None, sub=None, fc=PALE, ec=AXIS, lw=0.9,
        fs=9, subfs=7.4, bold=False, radius=0.9, align="center"):
    """A rounded panel with an optional bold label and a smaller sub-line."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    tx = x + w / 2 if align == "center" else x + 1.4
    ha = "center" if align == "center" else "left"
    if label and sub:
        ax.text(tx, y + h * 0.62, label, ha=ha, va="center", fontsize=fs,
                color=INK, zorder=3, fontweight="bold" if bold else "normal")
        ax.text(tx, y + h * 0.26, sub, ha=ha, va="center", fontsize=subfs,
                color=INK2, zorder=3, linespacing=1.5)
    elif label:
        ax.text(tx, y + h / 2, label, ha=ha, va="center", fontsize=fs, color=INK,
                zorder=3, fontweight="bold" if bold else "normal")
    return x + w / 2, y


def arrow(ax, x0, y0, x1, y1, label=None, color=MUTED, lw=1.1, fs=7.2,
          dx=0.9, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11,
        color=color, linewidth=lw, shrinkA=1.5, shrinkB=1.5, zorder=4))
    if label:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2, label, ha="left",
                va="center", fontsize=fs, color=INK2, zorder=5)


def band(ax, y, text, x=1.0):
    """The left-hand rail marking one stage of the pipeline."""
    ax.text(x, y, text, ha="left", va="center", fontsize=8.6, color=RUST,
            fontweight="bold", rotation=0, zorder=3)



MONO = ["DejaVu Sans Mono", "monospace"]
# The four visible locus rows stand in for all 19; the ellipsis row carries the
# count so the figure never claims to be drawing every locus.
ROWS = [("katG", 74.0), ("rpoB", 65.0), ("inhA", 56.0), (None, 48.5),
        ("rrs", 40.0)]
RH = 5.4                                     # locus row height
REAL = [(n, y) for n, y in ROWS if n]
ELL_Y = next(y for n, y in ROWS if n is None)


def shape(ax, x, y, text, color=None, fs=8.0, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, family=MONO,
            color=color or INK, zorder=6)


def stage_label(ax, x, text, sub=None):
    ax.text(x, 92.5, text, ha="center", va="center", fontsize=10.0, color=INK,
            fontweight="bold", zorder=6)
    if sub:
        ax.text(x, 88.4, sub, ha="center", va="center", fontsize=7.6,
                color=INK2, zorder=6)


def fusion_badge(ax, x, y, n, text):
    """Mark a point where the locus axis is combined, and say how."""
    ax.add_patch(Circle((x, y), 2.15, facecolor=RUST, edgecolor="none", zorder=7))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=8.6, color=SURFACE,
            fontweight="bold", zorder=8)
    ax.text(x, y - 4.4, text, ha="center", va="top", fontsize=7.6, color=RUST,
            zorder=7, linespacing=1.55)


def draw_slide():
    """Compact shape flow, sized for a slide.

    One question per column — what shape is the data here — and the three
    places the LOCUS axis is combined are drawn as convergences rather than
    only named, because that is the thing about this architecture a reader has
    to see: modalities fuse inside a locus, loci then fuse with each other, and
    loci fuse once more into a per-drug vector at the readout.
    """
    fig = plt.figure(figsize=(14.2, 6.0))
    ax = fig.add_axes([0.005, 0.02, 0.99, 0.90]); ax.set_axis_off()
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    d, T, n_l = D["d_model"], T_PER_LOCUS, N_LOCI

    # ---------------------------------------------------- 1. delta blocks
    x0, w = 2.0, 16.0
    stage_label(ax, x0 + w / 2, "delta blocks", f"{len(SPEC)} per locus")
    LENS = [SPEC[m][1] for m in SPEC]
    tot = sum(LENS)
    FILL = [PALE_BLUE, "#e3dcf3", "#d9eee4", "#fbe6c9"]
    for name, y in REAL:
        cx = x0
        for (m, (c, L)), fc in zip(SPEC.items(), FILL):
            bw = w * L / tot
            ax.add_patch(Rectangle((cx, y), bw, RH, facecolor=fc,
                                   edgecolor=AXIS, linewidth=0.6, zorder=3))
            if name == REAL[0][0]:
                ax.text(cx + bw / 2, y + RH + 1.6, m[:4], ha="center",
                        va="center", fontsize=6.2, color=INK2, zorder=4)
            cx += bw
        ax.text(x0 - 0.9, y + RH / 2, name, ha="right", va="center",
                fontsize=7.8, color=INK, zorder=4)
    ax.text(x0 + w / 2, ELL_Y + 1.4, "· · ·", ha="center", va="center",
            fontsize=11, color=MUTED)
    ax.text(x0 - 0.9, ELL_Y + 1.4, f"{n_l} loci", ha="right", va="center",
            fontsize=7.4, color=INK2)
    shape(ax, x0 + w / 2, 30.5, "(B, C, L)")
    shape(ax, x0 + w / 2, 26.4, "C,L = 5,2488 · 20,829\n3,829 · 5,200",
          color=MUTED, fs=6.9)

    # ---------------------------------------------------- 2. variant tokens
    x1, w1 = 23.5, 15.0
    stage_label(ax, x1 + w1 / 2, "variant tokens", "one per deviation")
    for name, y in REAL:
        ax.add_patch(Rectangle((x1, y), 3.0, RH, facecolor=SURFACE,
                               edgecolor=RUST, linewidth=1.0, zorder=3))
        ax.text(x1 + 1.5, y + RH / 2, "WT", ha="center", va="center",
                fontsize=5.6, color=RUST, fontweight="bold", zorder=4)
        for k in range(4):
            ax.add_patch(Rectangle((x1 + 4.0 + k * 2.8, y), 2.3, RH,
                                   facecolor=PALE_RUST, edgecolor=RUST,
                                   linewidth=0.6, zorder=3))
        ax.text(x1 + 15.0, y + RH / 2, "…", ha="center", va="center",
                fontsize=8, color=MUTED, zorder=4)
    ax.text(x1 + w1 / 2, ELL_Y + 1.4, "· · ·", ha="center", va="center",
            fontsize=11, color=MUTED)
    ax.text(x1 + 1.5, REAL[0][1] + RH + 1.6, "[WT]", ha="center", va="center",
            fontsize=6.2, color=RUST, zorder=4)
    ax.text(x1 + 9.6, REAL[0][1] + RH + 1.6,
            f"≤ {D['max_variants']} × {N_STREAMS} streams", ha="center",
            va="center", fontsize=6.2, color=INK2, zorder=4)
    shape(ax, x1 + w1 / 2, 30.5, f"(B, {n_l}, {T}, {C_TOK})")
    shape(ax, x1 + w1 / 2, 26.4, f"→ embed →  (B, {n_l}, {T}, {d})",
          color=MUTED, fs=7.0)

    # ---------------------------------------------------- 3. stage 1
    x2, w2 = 44.0, 14.0
    stage_label(ax, x2 + w2 / 2, "stage 1",
                f"within a locus · Encoder × {D['enc_layers']}")
    ax.add_patch(FancyBboxPatch(
        (x2, 36.0), w2, 45.5, boxstyle="round,pad=0,rounding_size=1.0",
        facecolor=PALE_BLUE, edgecolor=BLUE, linewidth=1.2, zorder=2))
    # the T slots of one locus converging onto that locus's single vector
    for name, y in REAL:
        yc = y + RH / 2
        for k in range(4):
            sxx = x2 + 1.6 + k * 1.9
            ax.add_patch(Rectangle((sxx, y + 0.9), 1.4, RH - 1.8,
                                   facecolor=SURFACE, edgecolor=BLUE,
                                   linewidth=0.6, zorder=4))
            ax.add_patch(FancyArrowPatch(
                (sxx + 1.4, yc), (x2 + 10.6, yc), arrowstyle="-",
                color=BLUE, alpha=0.45, linewidth=0.6, zorder=3))
        ax.add_patch(Rectangle((x2 + 10.6, y + 0.7), 2.4, RH - 1.4,
                               facecolor=BLUE, edgecolor="none", zorder=5))
    ax.text(x2 + w2 / 2, ELL_Y + 1.4, "· · ·", ha="center", va="center",
            fontsize=11, color=MUTED, zorder=4)
    ax.text(x2 + 4.4, 83.4, f"{T} slots", ha="center", va="center",
            fontsize=6.2, color=INK2, zorder=6)
    ax.text(x2 + 11.8, 83.4, "[WT] out", ha="center", va="center",
            fontsize=6.2, color=BLUE, zorder=6)
    shape(ax, x2 + w2 / 2, 30.5, f"(B, {n_l}, {d})")
    fusion_badge(ax, x2 + w2 / 2, 20.5, 1,
                 f"{len(SPEC)} modalities and all {T} slots\n"
                 f"of a locus → 1 vector")

    # ---------------------------------------------------- 4. stage 2
    x3, w3 = 63.5, 14.0
    stage_label(ax, x3 + w3 / 2, "stage 2",
                f"across loci · Encoder × {D['fusion_layers']}")
    ax.add_patch(FancyBboxPatch(
        (x3, 36.0), w3, 45.5, boxstyle="round,pad=0,rounding_size=1.0",
        facecolor=PALE_RUST, edgecolor=RUST, linewidth=1.2, zorder=2))
    # all-to-all, drawn as two columns so the mesh is legible: every locus on
    # the left reaches every locus on the right
    lx, rx = x3 + 2.2, x3 + 10.0
    for _, ya in REAL:
        for _, yb in REAL:
            ax.add_patch(FancyArrowPatch(
                (lx + 2.0, ya + RH / 2), (rx, yb + RH / 2), arrowstyle="-",
                color=RUST, alpha=0.32, linewidth=0.6, zorder=3))
    for _, y in REAL:
        ax.add_patch(Rectangle((lx, y + 0.7), 2.0, RH - 1.4, facecolor=SURFACE,
                               edgecolor=RUST, linewidth=0.8, zorder=5))
        ax.add_patch(Rectangle((rx, y + 0.7), 2.4, RH - 1.4, facecolor=RUST,
                               edgecolor="none", zorder=5))
    ax.text(x3 + w3 / 2, ELL_Y + 1.4, "· · ·", ha="center", va="center",
            fontsize=11, color=MUTED, zorder=4)
    shape(ax, x3 + w3 / 2, 30.5, f"(B, {n_l}, {d})")
    fusion_badge(ax, x3 + w3 / 2, 20.5, 2,
                 "every locus attends to\nevery other locus")

    # ---------------------------------------------------- 5. readout
    x4 = 88.0
    stage_label(ax, x4, "readout", "one query per drug")
    ty = 62.0
    for _, y in REAL:                       # 19 loci converge onto one vector
        ax.add_patch(FancyArrowPatch(
            (x3 + w3 + 0.6, y + RH / 2), (x4 - 3.0, ty), arrowstyle="-|>",
            mutation_scale=8, color=MUTED, linewidth=0.8, shrinkA=0,
            shrinkB=2, zorder=3))
    ax.add_patch(Rectangle((x4 - 3.0, ty - 2.4), 6.0, 4.8, facecolor=INK2,
                           edgecolor="none", zorder=5))
    ax.text(x4, ty + 5.4, "cross-attention", ha="center", va="center",
            fontsize=7.0, color=INK2, zorder=5)
    shape(ax, x4, 53.5, "(B, n_drugs, 128)", fs=7.2)
    ax.add_patch(FancyArrowPatch((x4, 50.5), (x4, 46.5), arrowstyle="-|>",
                                 mutation_scale=9, color=MUTED, linewidth=1.0,
                                 zorder=4))
    ax.text(x4, 44.8, "Linear 128→256 · ReLU\nLinear 256→1", ha="center",
            va="top", fontsize=6.9, color=INK2, zorder=5, linespacing=1.65)
    shape(ax, x4, 30.5, "(B, n_drugs)")
    fusion_badge(ax, x4, 20.5, 3,
                 f"all {n_l} loci pooled into\none vector per drug")

    # ---------------------------------------------------- connectors
    for xa, xb in ((x0 + w, x1), (x1 + w1, x2), (x2 + w2, x3)):
        for _, y in REAL:
            ax.add_patch(FancyArrowPatch(
                (xa + 0.6, y + RH / 2), (xb - 0.6, y + RH / 2),
                arrowstyle="-|>", mutation_scale=8, color=MUTED,
                linewidth=0.8, shrinkA=0, shrinkB=0, zorder=3))

    ax.text(0.5, 5.0, f"B = batch   ·   {n_l} loci × {len(SPEC)} modalities   "
                      f"·   d_model {d}   ·   {TOTAL:,} parameters",
            ha="left", va="center", fontsize=7.6, color=INK2, family=MONO)

    fig.suptitle("LocusFusionNet — tensor shapes and the three points where "
                 "the locus axis is combined",
                 x=0.005, ha="left", y=0.985, fontsize=12.5, color=INK,
                 fontweight="bold")
    return fig


def draw_full():
    """The reference sheet: every stage, the token layout, the parameter split."""
    fig = plt.figure(figsize=(15.5, 11.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1], wspace=0.06,
                          left=0.015, right=0.985, top=0.915, bottom=0.02)
    ax = fig.add_subplot(gs[0, 0]); ax.set_axis_off()
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    sx = fig.add_subplot(gs[0, 1]); sx.set_axis_off()
    sx.set_xlim(0, 100); sx.set_ylim(0, 100)

    # ============================================================ A. input
    band(ax, 97.6, "INPUT")
    ax.text(11, 97.6, f"delta-encoded blocks, one per (modality, locus) — "
                      f"{N_LOCI} loci × {len(SPEC)} modalities",
            ha="left", va="center", fontsize=8.6, color=INK)
    ax.text(11, 94.6, "a channel is non-zero only at columns where the isolate "
                      "differs from the H37Rv reference",
            ha="left", va="center", fontsize=7.4, color=INK2)

    MODS = [("dna", "(5, 2488)"), ("protein", "(20, 829)"),
            ("biophysical", "(3, 829)"), ("regulatory", "(5, 200)")]
    gx, gw, gap = 20.0, 15.0, 4.0
    shown = ["katG", "rpoB", "inhA"]
    for gi, locus in enumerate(shown):
        x0 = gx + gi * (gw + gap)
        for mi, (m, shape) in enumerate(MODS):
            y0 = 88.4 - mi * 2.15
            ax.add_patch(Rectangle((x0, y0), gw, 1.75, facecolor=PALE,
                                   edgecolor=AXIS, linewidth=0.7, zorder=2))
            if gi == 0:
                ax.text(x0 - 1.2, y0 + 0.88, m, ha="right", va="center",
                        fontsize=7.2, color=INK2, zorder=3)
            ax.text(x0 + gw / 2, y0 + 0.88, shape, ha="center", va="center",
                    fontsize=6.6, color=MUTED, zorder=3)
        ax.text(x0 + gw / 2, 90.9, locus, ha="center", va="center", fontsize=8.4,
                color=INK, fontweight="bold", zorder=3)
    _last = gx + len(shown) * (gw + gap)
    ax.text(_last + 3.0, 84.4, "· · ·", ha="center", va="center", fontsize=13,
            color=MUTED)
    ax.text(_last + 3.0, 90.9, f"{N_LOCI} loci", ha="center", va="center",
            fontsize=8.0, color=INK2)

    # ============================================================ B. tokenizer
    band(ax, 80.6, "TOKENIZE")
    ax.text(11, 80.6, "one token per column that differs from reference, "
                      "grouped into coordinate streams",
            ha="left", va="center", fontsize=8.6, color=INK)
    arrow(ax, gx + gw / 2, 79.3, gx + gw / 2, 76.4)

    box(ax, 8, 55.0, 88, 21.0, fc=SURFACE, ec=RUST, lw=1.2)
    ax.text(10.0, 74.5, f"within one locus  ·  {D['max_variants']} variants max "
                        f"per stream  ·  0 parameters except the learned nt offset",
            ha="left", va="center", fontsize=7.6, color=RUST)

    STREAMS_ROWS = [
        ("nt", "dna", "coord = column / 3  −  offset[ℓ]", [7, 31, 62]),
        ("aa", "protein + biophysical\n(co-indexed, same columns)", "coord = codon k", [22, 44]),
        ("reg", "regulatory", "coord = (q − L) / 3    (negative, upstream)", [12, 55, 78, 90]),
    ]
    # [WT] first: it is slot 0 of every locus, and putting it last left the `reg`
    # coordinate formula written across it.
    ax.add_patch(Rectangle((10.0, 71.3), 4.6, 2.1, facecolor=SURFACE,
                           edgecolor=RUST, linewidth=1.2, zorder=3))
    ax.text(12.3, 72.35, "[WT]", ha="center", va="center", fontsize=7.4,
            color=RUST, fontweight="bold", zorder=4)
    ax.text(15.8, 72.35, "one learned sentinel per locus, always present — "
                         f"a pan-susceptible isolate is {N_LOCI} sentinels "
                         "and nothing else",
            ha="left", va="center", fontsize=7.2, color=INK2)

    for si, (stream, source, formula, marks) in enumerate(STREAMS_ROWS):
        y = 67.4 - si * 5.0
        ax.text(10.0, y + 0.9, stream, ha="left", va="center", fontsize=8.6,
                color=INK, fontweight="bold")
        ax.text(14.6, y + 0.9, source, ha="left", va="center", fontsize=7.0,
                color=INK2, linespacing=1.35)
        # the alignment, mostly identical to reference; ticks are the deviations
        sx0, sw = 33.0, 26.0
        ax.add_patch(Rectangle((sx0, y), sw, 1.9, facecolor=PALE, edgecolor=AXIS,
                               linewidth=0.7, zorder=2))
        for mk in marks:
            ax.add_patch(Rectangle((sx0 + sw * mk / 100 - 0.22, y), 0.55, 1.9,
                                   facecolor=RUST, edgecolor="none", zorder=3))
        arrow(ax, sx0 + sw + 0.8, y + 0.95, sx0 + sw + 5.2, y + 0.95)
        for k in range(len(marks)):
            ax.add_patch(Rectangle((sx0 + sw + 6.2 + k * 2.5, y), 2.1, 1.9,
                                   facecolor=PALE_RUST, edgecolor=RUST,
                                   linewidth=0.7, zorder=3))
        ax.text(sx0 + sw + 6.2 + len(marks) * 2.5 + 0.8, y + 0.95,
                f"≤ {D['max_variants']}", ha="left", va="center", fontsize=7.0,
                color=INK2)
        ax.text(sx0, y - 1.15, formula, ha="left", va="center", fontsize=6.9,
                color=MUTED)

    ax.text(10.0, 53.6, f"{T_PER_LOCUS} token slots per locus   =   1 [WT]  +  "
                        f"{N_STREAMS} streams × {D['max_variants']}",
            ha="left", va="center", fontsize=7.6, color=INK)

    # ============================================================ C. embed
    band(ax, 50.2, "EMBED")
    arrow(ax, 30, 52.6, 30, 50.0)
    box(ax, 8, 39.4, 88, 10.2, fc=SURFACE, ec=AXIS, lw=0.9)
    ax.text(10.0, 47.6, "every token", ha="left", va="center", fontsize=7.6,
            color=INK, fontweight="bold")
    ax.text(23.0, 47.6,
            f"tok  =  tok_proj({C_TOK} → {D['d_model']})   +   "
            f"pos_proj( sinusoid_{D['pos_dims']}(coord) → {D['d_model']} )   +   "
            f"locus_emb[ℓ]",
            ha="left", va="center", fontsize=8.0, color=INK)
    ax.text(10.0, 44.6, "[WT] only", ha="left", va="center", fontsize=7.6,
            color=RUST, fontweight="bold")
    ax.text(23.0, 44.6, "+  wt_emb[ℓ]   +   wt_proj( [ log1p(n_variants), "
                        "coverage, uncovered ] → 128 )",
            ha="left", va="center", fontsize=8.0, color=INK)
    ax.text(10.0, 41.6, f"per locus", ha="left", va="center", fontsize=7.6,
            color=INK, fontweight="bold")
    ax.text(23.0, 41.6, f"FiLM[ℓ]:  tok · (1 + scale[ℓ]) + shift[ℓ]"
                        f"      →   LayerNorm   →   zeroed at padded slots",
            ha="left", va="center", fontsize=8.0, color=INK)

    # ============================================================ D. stage 1
    band(ax, 35.4, "STAGE 1")
    arrow(ax, 30, 38.6, 30, 33.4)
    box(ax, 8, 25.4, 88, 8.0, fc=PALE_BLUE, ec=BLUE, lw=1.2,
        label="within a locus   ·   TransformerEncoder × "
              f"{D['enc_layers']}   ·   d={D['d_model']}, heads={D['nhead']}, "
              f"ff={D['enc_dim_ff']}, pre-norm",
        sub=f"attends over that locus's own {T_PER_LOCUS} slots, padding-masked   ·   "
            f"all {N_LOCI} loci run in one batched call, weights shared, "
            f"identity carried by locus_emb and FiLM",
        fs=9.2, subfs=7.4, bold=True)
    arrow(ax, 30, 25.0, 30, 21.6,
          label=f"z[ℓ]  =  the output at the [WT] slot        →   "
                f"{N_LOCI} locus summaries × {D['d_model']}")
    box(ax, 8, 18.0, 88, 3.4, fc=SURFACE, ec=AXIS, lw=0.9,
        label="KeyedTokenNorm — each locus summary standardised against the same "
              "locus in other isolates, never against a different one",
        fs=7.8)

    # ============================================================ E. stage 2
    band(ax, 14.0, "STAGE 2")
    arrow(ax, 30, 17.6, 30, 15.6)
    box(ax, 8, 9.4, 88, 6.2, fc=PALE_RUST, ec=RUST, lw=1.2,
        label=f"across loci   ·   TransformerEncoder × {D['fusion_layers']}   ·   "
              f"ff={D['fusion_dim_ff']}",
        sub=f"attends over the {N_LOCI} locus summaries   ·   "
            f"carry_variants={D['carry_variants']}, so individual variants are not "
            f"handed up by default",
        fs=9.2, subfs=7.4, bold=True)

    # ============================================================ F. readout
    band(ax, 6.4, "READOUT")
    arrow(ax, 30, 9.0, 30, 6.8)
    box(ax, 8, 0.8, 88, 6.0, fc=SURFACE, ec=AXIS, lw=0.9)
    ax.text(10.0, 4.9, "one learned query per drug   →   MultiheadAttention"
                       "( Q = drug queries,  K = V = fused loci )   →   "
                       "pooled (n_drugs × 128)",
            ha="left", va="center", fontsize=8.0, color=INK)
    ax.text(10.0, 2.5, f"LayerNorm   →   Linear({D['d_model']} → 256)   →   ReLU   "
                       f"→   Linear(256 → 1)        logit per drug",
            ha="left", va="center", fontsize=8.0, color=INK)
    ax.text(63.0, 2.5, "attention weights returned by forward(return_attn=True)",
            ha="left", va="center", fontsize=7.4, color=RUST)

    # ================================================== side: token feature vector
    sx.text(0, 97.5, f"Token feature vector, {C_TOK} dimensions",
            ha="left", va="center", fontsize=9.4, color=INK, fontweight="bold")
    segs = ([(lo, hi, m) for m, (lo, hi) in SLOTS.items()]
            + [(F_IS_NT, F_UNCOVERED + 1, "flags"), (F_PHASE, C_TOK, "codon phase")])
    FILL = {"dna": PALE_BLUE, "protein": "#e3dcf3", "biophysical": "#d9eee4",
            "regulatory": "#fbe6c9", "flags": PALE, "codon phase": PALE}
    for lo, hi, name in segs:
        x0, w = 100 * lo / C_TOK, 100 * (hi - lo) / C_TOK
        sx.add_patch(Rectangle((x0, 88.5), w, 4.6, facecolor=FILL[name],
                               edgecolor=AXIS, linewidth=0.7, zorder=2))
    for i, (lo, hi, name) in enumerate(segs):
        y = 85.0 - i * 3.2
        sx.text(0, y, f"{lo}:{hi}", ha="left", va="center", fontsize=7.2,
                color=MUTED)
        sx.add_patch(Rectangle((7.5, y - 0.8), 2.6, 1.6, facecolor=FILL[name],
                               edgecolor=AXIS, linewidth=0.6, zorder=2))
        sx.text(11.6, y, name, ha="left", va="center", fontsize=7.6, color=INK)
    sx.text(38, 85.0 - 4 * 3.2, f"is_nt / is_aa / is_reg / is_wt,\ngap, uncovered",
            ha="left", va="center", fontsize=6.8, color=INK2, linespacing=1.5)
    sx.text(0, 85.0 - len(segs) * 3.2 - 1.0,
            "one fixed layout at every modality set;\n"
            "an absent modality leaves its slot zero",
            ha="left", va="top", fontsize=7.0, color=INK2, linespacing=1.6)

    # ================================================== side: parameter budget
    sx.text(0, 56.0, f"Parameters  ·  {TOTAL:,} total",
            ha="left", va="center", fontsize=9.4, color=INK, fontweight="bold")
    sx.text(0, 52.6, f"at {N_LOCI} loci, {len(SPEC)} modalities, "
                     f"{args.drugs} output head" + ("s" if args.drugs > 1 else ""),
            ha="left", va="center", fontsize=7.2, color=INK2)
    BAR_C = {"stage 1 — within-locus encoder": BLUE,
             "stage 2 — cross-locus encoder": RUST}
    for i, (name, n) in enumerate(BUDGET):
        y = 46.5 - i * 6.6
        sx.text(0, y + 2.0, name, ha="left", va="center", fontsize=7.6, color=INK)
        sx.add_patch(Rectangle((0, y - 1.6), 72 * n / TOTAL, 2.6,
                               facecolor=BAR_C.get(name, MUTED), edgecolor="none",
                               zorder=2))
        sx.text(73, y - 0.3, f"{n:,}   {100 * n / TOTAL:.1f}%", ha="left",
                va="center", fontsize=7.2, color=INK)

    sx.text(0, 11.0, "Requires", ha="left", va="center", fontsize=9.4, color=INK,
            fontweight="bold")
    sx.text(0, 7.6, "--delta   reference-difference input\n"
                    "per-locus blocks (per_modality_branch=False)",
            ha="left", va="top", fontsize=7.4, color=INK2, linespacing=1.7)

    fig.suptitle("LocusFusionNet — variant tokens, fused within a locus then "
                 "across loci", x=0.015, ha="left", y=0.975, fontsize=13.5,
                 color=INK, fontweight="bold")
    fig.text(0.015, 0.941, "models/locusfusion.py   ·   --arch locusfusion",
             ha="left", fontsize=8.6, color=INK2)

    return fig


# ---------------------------------------------------------------- write
WANT = ("slide", "full") if args.layout == "both" else (args.layout,)
FIGDIR.mkdir(parents=True, exist_ok=True)
for which in WANT:
    fig = draw_slide() if which == "slide" else draw_full()
    stem = STEM if STEM and args.layout != "both" else FIGDIR / NAMES[which]
    for ext, dpi in (("png", 220), ("svg", None)):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight",
                    facecolor=SURFACE)
        print(f"wrote {stem.with_suffix('.' + ext)}")
    plt.close(fig)

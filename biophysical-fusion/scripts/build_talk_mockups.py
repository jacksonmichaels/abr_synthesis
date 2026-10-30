"""Schematic mock-ups for the lab talk, in the manuscript palette.

Writes to results/figures/talk/mockups/:
  s01_resistance_cartoon   rpoB S450L: codon change -> residue change -> rifampicin no longer seated
  s03_literature_table     what each published model changes (loci, representation, model)
  s05_axes                 the four axes varied in the study
  s06_models               block diagrams: LASSO, stacked CNN, late-fusion, cis-fusion
  s07_onehot_vs_tokens     one-hot alignment vs variant tokens for one locus
  s23_site_distance_method how per-position distances to the drug / binding site were computed
  s28_next_distance_channel next step: one token, a distance channel that depends on the queried drug
  s15_lasso_simple         LASSO for a general audience: yes/no features, summed weights, most weights zero

Drawing only; no data is read.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, Polygon, Rectangle, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paper_style as ps
from paper_style import MODEL, MODALITY, NEUTRAL as N

ps.apply()
plt.rcParams.update({"font.size": 15})
OUT = ROOT / "results" / "figures" / "talk" / "mockups"
DRUG = "#c0392b"          # rifampicin (cartoon only)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", fc="white", ec=N["axis"], lw=1.5, tc=None, fs=14, bold=False, r=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc or N["ink"], fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color=None, lw=1.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                 color=color or N["muted"], lw=lw))


# ---- slide 1 -------------------------------------------------------------------
def resistance_cartoon():
    fig, ax = canvas(14, 6.4, (0, 28), (-0.4, 12.4))
    for x0, codon, aa, resistant in [(0.5, "TCG", "Ser", False), (15.0, "TTG", "Leu", True)]:
        ax.text(x0 + 6, 11.7, "resistant" if resistant else "susceptible", ha="center", fontsize=19,
                fontweight="bold", color=N["ink"])
        # DNA strip with codon 450
        for i in range(12):
            ax.add_patch(Rectangle((x0 + i, 9.6), 0.92, 0.9, fc=N["grid"], ec="none"))
        for j, b in enumerate(codon):
            mut = resistant and j == 1
            ax.add_patch(Rectangle((x0 + 4.5 + j, 9.6), 0.92, 0.9,
                                   fc=MODALITY["dna"] if mut else "white", ec=MODALITY["dna"], lw=1.5))
            ax.text(x0 + 4.96 + j, 10.05, b, ha="center", va="center", fontsize=15,
                    color="white" if mut else N["ink"], fontweight="bold")
        arrow(ax, x0 + 6, 8.75, x0 + 6, 8.05)
        ax.text(x0 + 6, 9.0, f"rpoB codon 450: {codon} → {aa}", ha="center", fontsize=14, color=N["ink2"])
        # protein with a pocket
        ax.add_patch(Ellipse((x0 + 6, 3.9), 11, 6.4, fc="#e9e3f0", ec=N["axis"], lw=1.5))
        ax.add_patch(Ellipse((x0 + 6, 6.9), 3.2, 2.4, fc="white", ec="none"))       # pocket
        ax.text(x0 + 6, 2.6, "RNA polymerase β (RpoB)", ha="center", fontsize=14, color=N["ink2"])
        # residue 450 on the pocket floor
        rc = MODALITY["protein"]
        if resistant:
            ax.add_patch(Ellipse((x0 + 6, 5.95), 1.9, 1.5, fc=rc, ec=N["ink"], lw=1))
            ax.text(x0 + 6, 5.95, "L450", ha="center", va="center", fontsize=13, fontweight="bold", color="white")
            drug_y, drug_x = 7.6, x0 + 10.2
        else:
            ax.add_patch(Ellipse((x0 + 6, 5.75), 1.1, 0.7, fc=rc, ec=N["ink"], lw=1))
            ax.text(x0 + 6, 4.9, "S450", ha="center", va="center", fontsize=13, fontweight="bold", color=N["ink"])
            drug_y, drug_x = 6.8, x0 + 6
        hexagon = [(drug_x + 0.95 * np.cos(t), drug_y + 0.95 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 7)[:-1]]
        ax.add_patch(Polygon(hexagon, fc=DRUG, ec=N["ink"], lw=1))
        ax.text(drug_x + (0 if resistant else 1.9), drug_y + (-1.4 if resistant else 0.9), "rifampicin",
                fontsize=14, color=DRUG, ha="center" if resistant else "left", va="center")
        ax.text(x0 + 6, -0.1, "drug binds, transcription blocked" if not resistant else
                "drug cannot seat, transcription continues", ha="center", fontsize=14, color=N["ink"])
    save(fig, "s01_resistance_cartoon")


# ---- slide 3 -------------------------------------------------------------------
def literature_table():
    cols = ["", "loci / region", "input representation", "model"]
    rows = [
        ["Green et al. 2022", "18 loci", "one-hot DNA", "CNN, multi- and single-drug"],
        ["Big-TB 2026", "own loci (SD-CNN) /\nall loci (MD-CNN)", "one-hot DNA; DNABERT,\nESM embeddings",
         "logistic regression, CNN,\ntransformer"],
        ["Kulkarni et al. 2026", "resistance loci", "DNA + residue properties\n+ lineage", "CNN (MIC)"],
        ["Serajian et al. 2024", "whole genome", "31-mer counts", "LASSO, random forest"],
        ["DNA language model 2025", "annotated genes\n+ flanks", "DNA tokens", "fine-tuned language model"],
        ["this work", "own loci / 19 loci", "one-hot / variant tokens;\n4 modality sets",
         "LASSO, 3 CNNs, LocusFusion"],
    ]
    widths = [5.6, 4.2, 4.9, 5.6]
    rh, hh = 1.25, 0.8
    W, H = sum(widths), hh + rh * len(rows)
    fig, ax = canvas(15, 15 * H / W, (0, W), (0, H))
    x = 0
    for c, w in zip(cols, widths):
        ax.text(x + 0.15, H - hh / 2, c, va="center", fontsize=15, fontweight="bold", color=N["ink2"])
        x += w
    ax.plot([0, W], [H - hh, H - hh], color=N["axis"], lw=1.5)
    for i, r in enumerate(rows):
        y = H - hh - (i + 1) * rh
        ours = r[0] == "this work"
        if ours:
            ax.add_patch(Rectangle((0, y), W, rh, fc="#fbeee6", ec=MODEL["locusfusion"], lw=2))
        elif i % 2:
            ax.add_patch(Rectangle((0, y), W, rh, fc="#f7f7f7", ec="none"))
        x = 0
        for j, (t, w) in enumerate(zip(r, widths)):
            ax.text(x + 0.15, y + rh / 2, t, va="center", fontsize=14,
                    color=MODEL["locusfusion"] if ours else (N["ink"] if j == 0 else N["ink2"]),
                    fontweight="bold" if j == 0 or ours else "normal", linespacing=1.15)
            x += w
    save(fig, "s03_literature_table")


# ---- slide 5 -------------------------------------------------------------------
def chip(ax, x, y, text, fc, w=None, tc="white", fs=14):
    w = w or 0.24 * len(text) + 0.6
    box(ax, x, y, w, 0.75, text, fc=fc, ec=fc, tc=tc, fs=fs, r=0.3)
    return x + w + 0.25


def axes_figure():
    fig, ax = canvas(14, 7.6, (0, 28), (0, 15.2))
    panels = [(0.2, 7.9, "locus set"), (14.2, 7.9, "modalities"), (0.2, 0.2, "model size"),
              (14.2, 0.2, "input representation")]
    for x, y, t in panels:
        box(ax, x, y, 13.6, 7.0, ec=N["axis"], lw=1.5, r=0.3)
        ax.text(x + 0.5, y + 6.2, t, fontsize=19, fontweight="bold", color=N["ink"])
    # locus set
    x, y = 0.7, 11.6
    ax.text(x, y + 1.15, "each drug's own loci", fontsize=15, color=N["ink2"])
    xx = x
    for g in ["katG", "inhA"]:
        xx = chip(ax, xx, y, g, MODALITY["dna"], fs=14)
    ax.text(xx + 0.1, y + 0.35, "(isoniazid)", fontsize=13, color=N["muted"], va="center")
    y = 9.0
    ax.text(x, y + 1.15, "all 19 loci, for every drug", fontsize=15, color=N["ink2"])
    xx = x
    for g in ["katG", "inhA", "rpoB", "embB", "pncA", "rrs", "…"]:
        xx = chip(ax, xx, y, g, MODALITY["dna"], fs=13)
    # modalities
    x = 14.7
    rows = [("DNA", MODALITY["dna"], "the aligned sequence"),
            ("protein", MODALITY["protein"], "translated residues"),
            ("biophysical", MODALITY["biophysical"], "weight, pI, hydrophobicity"),
            ("regulatory", MODALITY["regulatory"], "WHO promoter windows")]
    for k, (t, c, d) in enumerate(rows):
        yy = 12.3 - k * 1.05
        chip(ax, x, yy, t, c, w=3.4, fs=14)
        ax.text(x + 3.8, yy + 0.37, d, fontsize=14, color=N["ink2"], va="center")
    ax.text(x, 8.25, "DNA alone, DNA + one, or all four", fontsize=14, color=N["ink"])
    # model size: log axis of parameter counts, drawn in data coordinates
    lo, hi, ax0, ax1, yax = 2, 8, 1.2, 13.0, 3.0
    pos = lambda v: ax0 + (np.log10(v) - lo) / (hi - lo) * (ax1 - ax0)
    ax.plot([ax0, ax1], [yax, yax], color=N["axis"], lw=1.5)
    for e in range(lo, hi + 1):
        ax.plot([pos(10 ** e)] * 2, [yax, yax - 0.2], color=N["axis"], lw=1.5)
        ax.text(pos(10 ** e), yax - 0.65, f"$10^{e}$", ha="center", fontsize=13, color=N["ink2"])
    ax.text((ax0 + ax1) / 2, 1.0, "parameters (LASSO: nonzero weights)", ha="center", fontsize=14, color=N["ink2"])
    pts = [("LASSO", 228, MODEL["lasso"], 1, "center"), ("LocusFusion", 0.65e6, MODEL["locusfusion"], -1, "right"),
           ("stacked CNN", 5.1e6, MODEL["cnn"], 1, "center"), (None, 45.9e6, MODEL["cnn"], 0, None),
           ("late-fusion,\ncis-fusion", 47.9e6, MODEL["cnn"], -1, "right")]
    for t, v, c, side, ha in pts:
        ax.scatter(pos(v), yax + 1.8, s=150, color=c, zorder=3)
        if t:
            two = "\n" in t
            ax.text(pos(v) + (0.5 if ha == "right" else 0),
                    yax + 1.8 + (0.75 if side > 0 else -0.85) + (0.3 * side if two else 0),
                    t, ha=ha, va="center", fontsize=14, color=c, linespacing=1.0)
    # representation
    x = 14.7
    ax.text(x, 5.3, "one-hot alignment", fontsize=15, color=N["ink"], fontweight="bold")
    ax.text(x, 4.6, "every position, A/C/G/T/gap/N", fontsize=14, color=N["ink2"])
    for i in range(24):
        hit = i in (6, 15)
        ax.add_patch(Rectangle((x + i * 0.5, 3.6), 0.45, 0.6, fc=MODALITY["dna"] if hit else N["grid"], ec="none"))
    ax.text(x, 2.5, "variant tokens", fontsize=15, color=MODEL["locusfusion"], fontweight="bold")
    ax.text(x, 1.8, "only positions that differ", fontsize=14, color=N["ink2"])
    xx = x
    for t, c in [("S", N["ink2"]), ("v1", MODEL["locusfusion"]), ("v2", MODEL["locusfusion"]),
                 ("N", MODALITY["uncalled"])]:
        xx = chip(ax, xx, 0.75, t, c, w=1.0, fs=13)
    ax.text(xx + 0.1, 1.1, "± uncalled-base channel", fontsize=14, color=N["ink2"], va="center")
    save(fig, "s05_axes")


# ---- slide 6 -------------------------------------------------------------------
def models_figure():
    fig, ax = canvas(16, 7.4, (0, 40), (0, 18.5))
    C = {"d": MODALITY["dna"], "p": MODALITY["protein"], "b": MODALITY["biophysical"], "r": MODALITY["regulatory"]}

    def head(x0, w, y_top, color, params):
        box(ax, x0 + w / 2 - 2.0, 3.6, 4.0, 1.3, "MLP", fc="white", ec=color, lw=2)
        arrow(ax, x0 + w / 2, 3.6, x0 + w / 2, 2.6, color)
        box(ax, x0 + w / 2 - 2.6, 1.3, 5.2, 1.3, "P(resistant)", fc="white", ec=color, lw=2, tc=color, bold=True)
        ax.text(x0 + w / 2, 0.3, params, ha="center", fontsize=15, color=N["ink2"])

    def stack(x, y, w, h, colors, label=None):
        """Input block: horizontal stripes, one per locus or modality."""
        n = len(colors)
        for i, c in enumerate(colors):
            ax.add_patch(Rectangle((x, y + i * h / n), w, h / n * 0.85, fc=c, ec="none"))
        if label:
            ax.text(x + w / 2, y + h + 0.3, label, ha="center", fontsize=12, color=N["ink2"])

    # LASSO
    x0, w = 0.5, 8.5
    ax.text(x0 + w / 2, 17.6, "LASSO", ha="center", fontsize=19, fontweight="bold", color=MODEL["lasso"])
    stack(x0 + 1.5, 12.5, 5.5, 2.4, [C["d"]] * 6, "19 loci, one-hot, end to end")
    arrow(ax, x0 + w / 2, 12.4, x0 + w / 2, 10.7, MODEL["lasso"])
    box(ax, x0 + 0.8, 7.6, 6.9, 3.0, "L1 logistic\nregression", fc="#e4f4f1", ec=MODEL["lasso"], lw=2, fs=15)
    arrow(ax, x0 + w / 2, 7.5, x0 + w / 2, 5.0, MODEL["lasso"])
    box(ax, x0 + w / 2 - 2.0, 3.6, 4.0, 1.3, "sigmoid", fc="white", ec=MODEL["lasso"], lw=2)
    arrow(ax, x0 + w / 2, 3.6, x0 + w / 2, 2.6, MODEL["lasso"])
    box(ax, x0 + w / 2 - 2.6, 1.3, 5.2, 1.3, "P(resistant)", fc="white", ec=MODEL["lasso"], lw=2,
        tc=MODEL["lasso"], bold=True)
    ax.text(x0 + w / 2, 0.3, "228 of 22,384 weights", ha="center", fontsize=15, color=N["ink2"])

    cnn = MODEL["cnn"]
    # stacked CNN: loci as channels, DNA + regulatory share a trunk
    x0, w = 10.0, 9.5
    ax.text(x0 + w / 2, 17.6, "stacked CNN", ha="center", fontsize=19, fontweight="bold", color=cnn)
    groups = [([C["d"]] * 3 + [C["r"]] * 2, "DNA + reg"), ([C["p"]] * 3, "protein"), ([C["b"]] * 3, "biophys.")]
    for k, (cols, lab) in enumerate(groups):
        gx = x0 + 0.3 + k * 3.1
        stack(gx, 12.5, 2.7, 2.4, cols, lab)
        arrow(ax, gx + 1.35, 12.4, gx + 1.35, 11.2, cnn)
        box(ax, gx, 8.8, 2.7, 2.3, "CNN", fc="#e3ecf5", ec=cnn, lw=2)
        arrow(ax, gx + 1.35, 8.7, x0 + w / 2, 7.1, cnn)
    box(ax, x0 + w / 2 - 3.1, 5.6, 6.2, 1.4, "flatten + concat", ec=cnn, fs=13)
    arrow(ax, x0 + w / 2, 5.5, x0 + w / 2, 5.0, cnn)
    head(x0, w, 0, cnn, "5.1M parameters")
    ax.text(x0 + w / 2, 16.2, "loci stacked as channels", ha="center", fontsize=12, color=N["muted"])

    # late-fusion: one encoder per modality, loci concatenated along the sequence
    x0, w = 20.0, 9.5
    ax.text(x0 + w / 2, 17.6, "late-fusion", ha="center", fontsize=19, fontweight="bold", color=cnn)
    ax.text(x0 + w / 2, 16.2, "one encoder per modality", ha="center", fontsize=12, color=N["muted"])
    for k, (c, lab) in enumerate([(C["d"], "DNA"), (C["p"], "prot."), (C["b"], "bio."), (C["r"], "reg.")]):
        gx = x0 + 0.2 + k * 2.35
        # one sequence per modality: the loci laid end to end (segment = one locus)
        sx = gx
        for frac in (0.45, 0.33, 0.22):
            sw = 2.0 * frac
            ax.add_patch(Rectangle((sx, 13.6), sw - 0.08, 1.1, fc=c, ec="none"))
            sx += sw
        ax.text(gx + 1.0, 15.0, lab, ha="center", fontsize=12, color=N["ink2"])
        arrow(ax, gx + 1.0, 13.5, gx + 1.0, 11.2, cnn)
        box(ax, gx, 8.8, 2.0, 2.3, "enc", fc="#e3ecf5", ec=cnn, lw=2, fs=13)
        arrow(ax, gx + 1.0, 8.7, x0 + w / 2, 7.1, cnn)
    ax.text(x0 + w / 2, 12.55, "loci joined end to end", ha="center", fontsize=12, color=N["ink2"],
            bbox=dict(fc="white", ec="none", pad=1), zorder=4)
    box(ax, x0 + w / 2 - 3.1, 5.6, 6.2, 1.4, "flatten + concat", ec=cnn, fs=13)
    arrow(ax, x0 + w / 2, 5.5, x0 + w / 2, 5.0, cnn)
    head(x0, w, 0, cnn, "45.9M parameters")

    # cis-fusion: one encoder per locus and modality, regulatory window appended to its locus
    x0, w = 30.0, 9.5
    ax.text(x0 + w / 2, 17.6, "cis-fusion", ha="center", fontsize=19, fontweight="bold", color=cnn)
    ax.text(x0 + w / 2, 16.2, "one encoder per locus and modality", ha="center", fontsize=12, color=N["muted"])
    for k, lab in enumerate(["locus 1", "locus 2", "locus N"]):
        gx = x0 + 0.2 + k * 3.15
        for m, c in enumerate([C["d"], C["p"], C["b"]]):
            sx = gx + m * 0.95
            if m == 0:
                ax.add_patch(Rectangle((sx, 12.5), 0.8, 1.7, fc=C["d"], ec="none"))
                ax.add_patch(Rectangle((sx, 14.3), 0.8, 0.6, fc=C["r"], ec="none"))
            else:
                ax.add_patch(Rectangle((sx, 12.5), 0.8, 2.4, fc=c, ec="none"))
            arrow(ax, sx + 0.4, 12.4, sx + 0.4, 11.2, cnn, lw=1.2)
            box(ax, sx, 8.8, 0.8, 2.3, "", fc="#e3ecf5", ec=cnn, lw=1.5, r=0.08)
        ax.text(gx + 1.35, 15.2, lab, ha="center", fontsize=12, color=N["ink2"])
        arrow(ax, gx + 1.35, 8.7, x0 + w / 2, 7.1, cnn)
    box(ax, x0 + w / 2 - 3.1, 5.6, 6.2, 1.4, "flatten + concat", ec=cnn, fs=13)
    arrow(ax, x0 + w / 2, 5.5, x0 + w / 2, 5.0, cnn)
    head(x0, w, 0, cnn, "47.9M parameters")

    # modality key
    kx = 9.5
    for t, c in [("DNA", C["d"]), ("protein", C["p"]), ("biophysical", C["b"]), ("regulatory", C["r"])]:
        ax.add_patch(Rectangle((kx, -1.4), 0.6, 0.6, fc=c, ec="none"))
        ax.text(kx + 0.85, -1.1, t, fontsize=14, va="center", color=N["ink2"])
        kx += 5.6
    ax.set_ylim(-1.8, 18.5)
    save(fig, "s06_models")


# ---- slide 7 -------------------------------------------------------------------
def onehot_vs_tokens():
    fig, ax = canvas(15, 5.2, (0, 37.5), (1.6, 14))
    ncol, cw = 52, 0.42
    ref = np.random.default_rng(3).integers(0, 4, ncol)
    variants = {9: 2, 23: 0, 38: None}          # column -> alt base row (None = uncalled)
    letters = "ACGT"
    ax.text(0, 13.2, "one-hot alignment", fontsize=18, fontweight="bold", color=N["ink"])
    ax.text(0, 12.2, "one column per aligned position; mostly identical to H37Rv", fontsize=14, color=N["ink2"])
    for r, L in enumerate(letters + "–N"):
        y = 10.1 - r * 1.05
        ax.text(-0.4, y + 0.45, L, ha="right", va="center", fontsize=13, color=N["ink2"])
        for c in range(ncol):
            if c in variants:
                alt = variants[c]
                on = (alt is None and L == "N") or (alt is not None and r == alt)
                fc = (MODALITY["uncalled"] if alt is None else MODALITY["dna"]) if on else "white"
                ax.add_patch(Rectangle((c * cw, y), cw * 0.9, 0.9, fc=fc,
                                       ec=MODALITY["uncalled"] if alt is None else MODALITY["dna"], lw=0.8))
            else:
                ax.add_patch(Rectangle((c * cw, y), cw * 0.9, 0.9, fc=N["axis"] if r == ref[c] else N["grid"],
                                       ec="none"))
    for c, lab in [(9, "v1"), (23, "v2"), (38, "N")]:
        ax.text(c * cw + cw * 0.45, 3.9, lab, ha="center", fontsize=13, fontweight="bold",
                color=MODALITY["uncalled"] if lab == "N" else MODALITY["dna"])
    ax.text(ncol * cw / 2, 2.6, "thousands of columns per locus, a few that differ", ha="center", fontsize=14,
            color=N["ink2"])
    arrow(ax, ncol * cw + 0.8, 7.5, ncol * cw + 3.2, 7.5, N["ink2"], lw=2.2)
    x0 = ncol * cw + 4.0
    ax.text(x0, 13.2, "variant tokens", fontsize=18, fontweight="bold", color=MODEL["locusfusion"])
    ax.text(x0, 12.2, "one token per difference, ≤ 16 per locus", fontsize=14, color=N["ink2"])
    toks = [("[S]", "summary token", N["ink2"]),
            ("v1", "base + residue + properties + Δ + phase", MODEL["locusfusion"]),
            ("v2", "base + residue + properties + Δ + phase", MODEL["locusfusion"]),
            ("N", "uncalled base", MODALITY["uncalled"])]
    for k, (t, d, c) in enumerate(toks):
        y = 9.6 - k * 1.6
        box(ax, x0, y, 1.6, 1.15, t, fc="white", ec=c, lw=2, tc=c, bold=True, fs=14)
        ax.text(x0 + 2.0, y + 0.57, d, va="center", fontsize=14, color=N["ink2"])
    save(fig, "s07_onehot_vs_tokens")


# ---- slide 23 ------------------------------------------------------------------
def site_distance_method():
    fig, ax = canvas(19, 8.2, (0, 47.5), (-1.2, 19.3))
    w, h, gap = 8.4, 4.4, 0.8

    def lane(y, title, steps, color):
        ax.text(0.4, y + h + 0.9, title, fontsize=15, fontweight="bold", color=color)
        for i, (head, body) in enumerate(steps):
            x = 0.4 + i * (w + gap)
            box(ax, x, y, w, h, fc="white", ec=color, lw=2, r=0.25)
            ax.text(x + 0.35, y + h - 0.65, head, fontsize=13, fontweight="bold", color=N["ink"], va="top")
            ax.text(x + 0.35, y + h - 1.65, body, fontsize=12, color=N["ink2"], va="top", linespacing=1.35)
            if i < len(steps) - 1:
                arrow(ax, x + w + 0.05, y + h / 2, x + w + gap - 0.05, y + h / 2, color)
        return 0.4 + len(steps) * (w + gap) - gap

    ribo, prot = N["ink"], MODEL["locusfusion"]
    end_a = lane(11.6, "ribosome loci rrs, rrl, rpsL  (streptomycin, kanamycin, amikacin, capreomycin)", [
        ("1  drug-bound structure", "RCSB PDB, mmCIF\n4DR3 STR   8EV7 KAN\n8EV6 AMK   5V93 CAP (Mtb)"),
        ("2  chains + drug atoms", "16S / 23S / S12 chains\ndrug copy bound to them;\nKAN / AMK: copy at rrs 1401"),
        ("3  map to H37Rv", "global alignment of each\nchain to rrs / rrl / rpsL\n(T. thermophilus \u2192 Mtb)"),
        ("4  distance to drug", "every H37Rv position:\nclosest heavy atom of its\nresidue to any drug atom"),
    ], ribo)
    end_b = lane(2.6, "protein loci rpoB, katG, inhA, ethR, embB, pncA, gyrA, gyrB, gid, eis", [
        ("1  binding-site residues", "literature review, mapped\nto H37Rv codon numbers\ne.g. rpoB D435 H445 S450"),
        ("2  AlphaFold model", "AlphaFold DB, H37Rv protein\nheavy atoms per codon\n(no drug in the model)"),
        ("3  site atoms", "all heavy atoms of the\nsite residues = the pocket"),
        ("4  distance to pocket", "every codon: closest heavy\natom to any site atom\n(site residue itself = 0 \u00c5)"),
    ], prot)

    # merge: variants read the distance at their position
    x = max(end_a, end_b) + 1.1
    xm = x + 0.2
    for y0 in (11.6 + h / 2, 2.6 + h / 2):
        arrow(ax, max(end_a, end_b) + 0.05, y0, xm, 9.6 if y0 > 9 else 8.6, N["ink2"])
    bx, bw = xm, 47.5 - xm - 0.2
    box(ax, bx, 5.4, bw, 7.4, fc="#f7f7f7", ec=N["axis"], lw=1.5, r=0.25)
    ax.text(bx + 0.35, 12.1, "5  per variant", fontsize=13.5, fontweight="bold", color=N["ink"], va="top")
    ax.text(bx + 0.35, 11.0, "each observed variant\nreads the distance at\nits position, then binned:\n\u22644, 4\u20138, 8\u201312,\n12\u201320, >20 \u00c5",
            fontsize=12, color=N["ink2"], va="top", linespacing=1.35)
    ax.text(23.7, 0.6, r"$d(i) = \min_{a \,\in\, \mathrm{residue}\ i,\ b \,\in\, \mathrm{target}} | a - b |$"
            "      target = drug atoms (ribosome) or site-residue atoms (AlphaFold)",
            ha="center", fontsize=14, color=N["ink"])
    save(fig, "s23_site_distance_method")


# ---- slide 28 ------------------------------------------------------------------
def next_distance_channel():
    """rrs 1401G as a token in a multi-drug model: the distance channel changes with the drug query.
    Distances (A) and % resistant carriers from site_distance_by_drug (4DR3, 8EV7, 8EV6, 5V93; test isolates)."""
    fig, ax = canvas(15, 5.2, (0, 37.5), (0.8, 14))
    dist = [("streptomycin", 13.3, 69, "#2a78d6"), ("kanamycin", 3.3, 92, "#eb6834"),
            ("amikacin", 3.4, 94, "#1baf7a"), ("capreomycin", 3.4, 81, "#eda100")]
    ax.text(0.3, 13.2, "one variant token", fontsize=16, fontweight="bold", color=N["ink"])
    ax.text(0.3, 12.2, "rrs 1401 A\u2192G", fontsize=14, color=N["ink2"])
    for c in range(20):
        on = c in (3, 9, 14)
        ax.add_patch(Rectangle((0.3 + c * 0.42, 9.9), 0.36, 1.4, fc=N["ink2"] if on else N["grid"], ec="none"))
    ax.add_patch(Rectangle((0.3 + 20 * 0.42 + 0.2, 9.9), 0.6, 1.4, fc="white", ec=MODEL["locusfusion"], lw=2))
    ax.text(0.3 + 20 * 0.42 + 0.5, 9.3, "d", ha="center", fontsize=14, color=MODEL["locusfusion"],
            fontweight="bold", style="italic")
    ax.text(0.3, 7.9, "existing 45 channels\n+ d: distance to the\n   queried drug", fontsize=13, color=N["ink2"],
            va="top", linespacing=1.3)
    arrow(ax, 10.6, 10.6, 12.8, 10.6, N["ink2"], lw=2.2)
    ax.text(13.3, 13.2, "drug query", fontsize=16, fontweight="bold", color=N["ink"])
    ax.text(22.6, 13.2, "d (\u00c5)", fontsize=16, fontweight="bold", color=MODEL["locusfusion"])
    ax.text(33.2, 13.2, "% R", fontsize=16, fontweight="bold", color=N["ink"])
    scale = 8.0 / 14
    for k, (drug, d, pr, c) in enumerate(dist):
        y = 10.9 - k * 2.4
        box(ax, 13.3, y - 0.6, 7.2, 1.5, drug, fc="white", ec=c, lw=2.2, tc=c, bold=True, fs=14, r=0.3)
        arrow(ax, 20.7, y + 0.15, 22.4, y + 0.15, N["muted"])
        ax.add_patch(Rectangle((22.6, y - 0.45), d * scale, 1.2, fc=c, ec="none", alpha=0.9))
        ax.text(22.6 + d * scale + 0.3, y + 0.15, f"{d:.1f}", va="center", fontsize=14, color=N["ink"])
        ax.text(33.2, y + 0.15, f"{pr}%", va="center", fontsize=14, color=N["ink"])
    save(fig, "s28_next_distance_channel")


# ---- slide 15 (general audience) -----------------------------------------------
def lasso_simple():
    """Three steps, schematic. The only numbers are real: ~22,384 candidate weights, ~228 kept (11-drug means)."""
    teal = MODEL["lasso"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1, 1, 1.25], "wspace": 0.35})

    # 1. yes / no features
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("1  each mutation becomes\n    a yes / no feature", loc="left", fontsize=15, fontweight="bold")
    feats = [("rpoB codon 450 changed?", 1), ("katG codon 315 changed?", 0), ("embB codon 306 changed?", 0),
             ("gyrA codon 94 changed?", 1), ("\u2026 22,384 of these", None)]
    for k, (f, v) in enumerate(feats):
        y = 8.3 - k * 1.7
        ax.text(0.0, y, f, fontsize=13, va="center", color=N["ink"] if v is not None else N["muted"])
        if v is not None:
            box(ax, 8.5, y - 0.55, 1.3, 1.1, "1" if v else "0", fc=teal if v else "white",
                ec=teal, lw=2, tc="white" if v else teal, bold=True, fs=15, r=0.2)

    # 2. add up weights -> probability
    ax = axes[1]
    ax.set_title("2  add up the weights of\n    the \u201cyes\u201d features", loc="left", fontsize=15, fontweight="bold")
    z = np.linspace(-6, 6, 200)
    ax.plot(z, 1 / (1 + np.exp(-z)), color=teal, lw=3)
    for zz, lab in ((-3.0, "low score"), (3.5, "high score")):
        pz = 1 / (1 + np.exp(-zz))
        ax.scatter([zz], [pz], s=90, color=teal, zorder=3, edgecolor="white", lw=1.5)
        ax.annotate(f"{lab}\nP = {pz:.2f}", (zz, pz), xytext=(16, 0) if zz < 0 else (-16, 0),
                    textcoords="offset points", fontsize=13, ha="left" if zz < 0 else "right", va="center",
                    color=N["ink"])
    ax.set_xlabel("score = sum of weights", fontsize=14)
    ax.set_ylabel("P(resistant)", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([0, 0.5, 1])
    ax.grid(True, axis="y")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # 3. LASSO: most weights exactly zero
    ax = axes[2]
    ax.set_title("3  LASSO switches most\n    weights off", loc="left", fontsize=15, fontweight="bold")
    rng = np.random.default_rng(4)
    n = 70
    plain = rng.normal(0, 0.35, n)
    lasso = np.zeros(n)
    keep = rng.choice(n, 5, replace=False)
    lasso[keep] = rng.choice([-1, 1], 5) * rng.uniform(0.9, 1.6, 5)
    for y0, w, lab in ((6.6, plain, "ordinary regression: every weight a little on"),
                       (1.6, lasso, "LASSO: a few weights on, the rest exactly 0")):
        ax.axhline(y0, color=N["axis"], lw=1)
        ax.bar(np.arange(n), w * 1.5, bottom=y0, color=teal if w is lasso else N["muted"], width=0.75)
        ax.text(-1, y0 + 2.7, lab, fontsize=13.5, color=N["ink"])
    ax.text(-1, -1.4, "here: ~22,384 possible weights \u2192 ~228 kept", fontsize=14, color=teal, fontweight="bold")
    ax.set_xlim(-2, n + 1)
    ax.set_ylim(-2.2, 10.2)
    ax.axis("off")
    save(fig, "s15_lasso_simple")


if __name__ == "__main__":
    lasso_simple()
    next_distance_channel()
    site_distance_method()
    resistance_cartoon()
    literature_table()
    axes_figure()
    models_figure()
    onehot_vs_tokens()

"""Paper figure: one real isolate through the LocusFusion v3 tokenizer.

    python scripts/tokenization_example.py

Reads results/figures/manuscript/tokenization_example.json (written by
scripts/tokenization_example_extract.py from the model's own feature tensor) and
writes results/figures/manuscript/fig_tokenization.{pdf,png} at the printed
width (6.5 in). Light: no data loading, safe anywhere.

Three bands: the isolate's loci with its variant positions, the tokens those
positions become (all 45 channels), and where the tokens go in the model.
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
OUTDIR = PROJECT / "results" / "figures" / "manuscript"
D = json.loads((OUTDIR / "tokenization_example.json").read_text())
LAY = D["layout"]
S = LAY["slots"]

# H37Rv context for the labels. inhA's promoter window runs through fabG1, which
# is upstream of inhA in the same operon: fabG1 starts 762 bp before inhA, so
# inhA c.-777 is fabG1 c.-15 (H37Rv fabG1 1673440, inhA 1674202).
FABG1_TO_INHA = 762


def box(ax, x, y, w, h, fc=SURFACE, ec=AXIS, lw=0.8, r=0.05):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=0.9, ms=7, **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, **kw))


def text(ax, x, y, s, size=6.4, color=INK, weight="normal", ha="left", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


def badge(ax, x, y, k, color, size=5.6, r=0.055):
    ax.add_patch(plt.Circle((x, y), r, fc=color, ec="none", zorder=5))
    text(ax, x, y - 0.003, str(k), size=size, color=SURFACE, weight="bold", ha="center", zorder=6)


# ---- what each variant token is, in words ------------------------------------
def describe(locus, t):
    nt = round(t["coord_codons"] * 3)               # 0-based nt from the start codon
    if t["alt"] == "N":
        where = "3′ flank" if locus in D["frames"] and t["codon"] < 0 and nt > 0 else "promoter"
        return dict(color=M["uncalled"], short=f"{locus} {where}",
                    long=f"{locus} {where}, uncalled base (N)")
    if t["region"] == "promoter":
        name = f"c.{nt}{t['ref']}>{t['alt']}"
        if locus == "inhA" and -FABG1_TO_INHA - 60 < nt < -FABG1_TO_INHA:
            name = f"fabG1 c.{nt + FABG1_TO_INHA}{t['ref']}>{t['alt']}"
        return dict(color=M["regulatory"], short=name.replace("-", "−"),
                    long=f"{name.replace('-', '−')}, promoter")
    aa = f"{t['ref_aa']}{t['codon'] + 1}{t['alt_aa']}"
    kind = "synonymous" if t["ref_aa"] == t["alt_aa"] else "missense"
    return dict(color=M["dna"], short=f"{locus} {aa}",
                long=f"{locus} c.{nt + 1}{t['ref']}>{t['alt']}, {aa} ({kind})")


rows = []                                            # (label, feat, colour, badge no.)
k = 0
for locus in D["loci"]:
    for t in D["tokens"][locus]:
        if t["kind"] == "null":
            rows.append(dict(label=f"[S]  {locus}", feat=t["feat"], color=INK2, n=None,
                             locus=locus, t=t))
        else:
            k += 1
            rows.append(dict(**describe(locus, t), feat=t["feat"], n=k, locus=locus, t=t,
                             label=None))

W, H = 6.5, 4.05
Y0 = 0.30                                            # empty strip below the model band
fig = plt.figure(figsize=(W, H - Y0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(Y0, H); ax.axis("off")

# =============================================================================
# 1  the isolate: its two loci to scale, variant positions marked
# =============================================================================
text(ax, 0.05, 3.92, "Isolate", size=8, weight="bold")
text(ax, 0.62, 3.92, f"{D['isolate']}, {D['drug'].lower()}-resistant; per-drug loci "
     f"{' and '.join(D['loci'])}", size=6.4, color=INK2)
x0, xw = 0.62, 5.70                                   # track origin and width (in)
lens = D["locus_lengths"]
total = max(v["promoter"] + 25 + v["cds"] for v in lens.values())
sc = xw / total
for i, locus in enumerate(D["loci"]):
    y = 3.58 - i * 0.36
    text(ax, 0.05, y, locus, size=6.8, weight="bold", style="italic")
    pl, cl = lens[locus]["promoter"], lens[locus]["cds"]
    ax.add_patch(Rectangle((x0, y - 0.045), pl * sc, 0.09, fc=PS.ramp(M["regulatory"], 4, 0.93)[3],
                           ec="none"))
    rx = x0 + (pl + 25) * sc
    ax.add_patch(Rectangle((rx, y - 0.045), cl * sc, 0.09, fc=PS.ramp(M["dna"], 4, 0.93)[3],
                           ec="none"))
    if i == 0:
        text(ax, x0 + pl * sc / 2, y + 0.115, "promoter window", size=5.6, color=MUTED, ha="center")
        text(ax, rx + cl * sc / 2, y + 0.115, "coding record", size=5.6, color=MUTED, ha="center")
    for r in rows:
        if r["locus"] != locus or r["n"] is None:
            continue
        t = r["t"]
        xx = x0 + (t["column"] if t["region"] == "promoter" else pl + 25 + t["column"]) * sc
        ax.plot([xx, xx], [y - 0.075, y + 0.075], color=r["color"], lw=1.4, solid_capstyle="butt")
        badge(ax, xx, y - 0.15, r["n"], r["color"])
ax.plot([x0 + xw - 500 * sc, x0 + xw], [2.98, 2.98], color=MUTED, lw=0.9, solid_capstyle="butt")
text(ax, x0 + xw - 250 * sc, 2.92, "500 columns", size=5.4, color=MUTED, ha="center", va="top")

# =============================================================================
# 2  the tokens: every row is what the model receives, all 45 channels
# =============================================================================
text(ax, 0.05, 2.80, "Variant tokens", size=8, weight="bold")
c_tok = D["c_tok"]
mx0, cw, rh = 1.02, 0.080, 0.165                     # matrix origin, cell width, row height
my_top = 2.42
groups = [("dna", "base", M["dna"]), ("protein", "residue", M["protein"]),
          ("biophysical", "props", M["biophysical"]), ("regulatory", "promoter", M["regulatory"])]
spans = [(S[g][0], S[g][1], lab, col) for g, lab, col in groups]
spans += [(LAY["d_bio"][0], LAY["d_bio"][1], "Δ props", M["biophysical"]),
          (LAY["f_is_null"], LAY["f_is_null"] + 1, None, INK2),
          (LAY["f_phase"], LAY["f_phase"] + 3, "phase", INK2)]
uncalled = {S["dna"][1] - 1, S["regulatory"][1] - 1, S["protein"][1] - 2, S["protein"][1] - 1}
continuous = set(range(*S["biophysical"])) | set(range(*LAY["d_bio"]))
chan_color = {}
for a, b, _, col in spans:
    for c in range(a, b):
        chan_color[c] = M["uncalled"] if c in uncalled else col

# group headers, one row; the 3-channel groups use short labels (key below the grid)
for j, (a, b, lab, col) in enumerate(spans):
    xa, xb = mx0 + a * cw, mx0 + b * cw
    ax.plot([xa + 0.008, xb - 0.008], [my_top + 0.105, my_top + 0.105], color=col, lw=1.6,
            solid_capstyle="butt")
    if lab:
        text(ax, (xa + xb) / 2, my_top + 0.17, lab, size=5.5, color=INK2, ha="center")

# channel letters for the short one-hot groups
nt_letters = D["alphabets"]["nt"]
for g in ("dna", "regulatory"):
    for j, c in enumerate(range(*S[g])):
        text(ax, mx0 + (c + 0.5) * cw, my_top + 0.035, nt_letters[j].replace("-", "–"), size=4.6,
             color=MUTED, ha="center")
text(ax, mx0 + (LAY["f_is_null"] + 0.5) * cw, my_top + 0.035, "S", size=4.6, color=MUTED,
     ha="center")
for j, c in enumerate(range(LAY["f_phase"], LAY["f_phase"] + 3)):
    text(ax, mx0 + (c + 0.5) * cw, my_top + 0.035, str(j), size=4.6, color=MUTED, ha="center")
for j, c in enumerate(range(*S["protein"])):
    lab = D["alphabets"]["protein"][j]
    text(ax, mx0 + (c + 0.5) * cw, my_top + 0.035, {"*": "∗", "-": "–"}.get(lab, lab), size=4.0,
         color=MUTED, ha="center")
for j, c in enumerate(list(range(*S["biophysical"])) + list(range(*LAY["d_bio"]))):
    text(ax, mx0 + (c + 0.5) * cw, my_top + 0.035, "MpH"[j % 3], size=4.4, color=MUTED, ha="center")

vmax = max(abs(r["feat"][c]) for r in rows for c in continuous) or 1.0
for i, r in enumerate(rows):
    y = my_top - (i + 1) * rh
    if i and r["label"] is not None:                 # a new locus starts at its summary token
        ax.plot([0.05, W - 0.05], [y + rh + 0.012, y + rh + 0.012], color=GRID, lw=0.6)
    for c in range(c_tok):
        v = r["feat"][c]
        xc = mx0 + c * cw
        ax.add_patch(Rectangle((xc + 0.004, y + 0.022), cw - 0.008, rh - 0.044,
                               fc=SURFACE, ec=GRID, lw=0.4))
        if v == 0:
            continue
        if c in continuous:
            # signed bar in the cell: up = above the residue mean, down = below
            hh = (rh - 0.05) / 2 * min(abs(v) / vmax, 1.0)
            mid = y + rh / 2
            ax.add_patch(Rectangle((xc + 0.012, mid if v > 0 else mid - hh), cw - 0.024, hh,
                                   fc=chan_color[c], ec="none"))
            ax.plot([xc + 0.006, xc + cw - 0.006], [mid, mid], color=AXIS, lw=0.3)
        else:
            ax.add_patch(Rectangle((xc + 0.004, y + 0.022), cw - 0.008, rh - 0.044,
                                   fc=chan_color[c], ec="none"))
    # row label (left) and read-out (right)
    if r["n"] is None:
        text(ax, 0.05, y + rh / 2, r["label"], size=6.0, color=INK2)
    else:
        badge(ax, 0.11, y + rh / 2, r["n"], r["color"])
        text(ax, 0.21, y + rh / 2, r["short"], size=6.0)
    t = r["t"]
    xr = mx0 + c_tok * cw + 0.10
    if r["n"] is None:
        s = "locus summary token"
    else:
        bits = [f"{t['ref']}→{t['alt']}"]
        if t["alt_aa"]:
            bits.append(f"{t['ref_aa']}→{t['alt_aa']}")
        if t["phase"] is not None:
            bits.append(f"phase {t['phase']}")
        db = [r["feat"][c] for c in range(*LAY["d_bio"])]
        if t["alt_aa"] and any(db):
            bits.append("Δ " + ", ".join(f"{n} {v:+.2f}" for n, v in zip("MpH", db) if v))
        elif t["alt_aa"]:
            bits.append("Δ 0")
        if t["alt"] == "N":
            bits.append("no codon")
        elif not t["alt_aa"] and t["region"] == "promoter":
            bits.append("no codon")
        s = " · ".join(bits)
    text(ax, xr, y + rh / 2, s, size=5.6, color=INK2)

ny = my_top - (len(rows) + 1) * rh + 0.03
text(ax, mx0, ny, "props  biophysical properties: M molecular weight, p isoelectric point, H hydrophobicity (z-scored)",
     size=5.2, color=MUTED)
lx = mx0 + 4.25
for j, (col, lab) in enumerate([(M["uncalled"], "uncalled-base channels")]):
    ax.add_patch(Rectangle((lx, ny - 0.035), 0.07, 0.07, fc=col, ec="none"))
    text(ax, lx + 0.10, ny, lab, size=5.2, color=MUTED)

# =============================================================================
# 3  where the tokens go
# =============================================================================
yb = 0.66
text(ax, 0.05, 1.10, "Model", size=8, weight="bold")
# per-locus token sets
bx = 0.10
for locus in D["loci"]:
    toks = [r for r in rows if r["locus"] == locus]
    text(ax, bx, yb + 0.25, locus, size=6.0, weight="bold", style="italic")
    for j, r in enumerate(toks):
        xx = bx + j * 0.17
        if r["n"] is None:
            ax.add_patch(Rectangle((xx, yb - 0.07), 0.14, 0.14, fc=SURFACE, ec=LF, lw=0.9))
            text(ax, xx + 0.07, yb, "S", size=5.6, color=LF, weight="bold", ha="center")
        else:
            ax.add_patch(Rectangle((xx, yb - 0.07), 0.14, 0.14, fc=LF_FILL, ec=LF, lw=0.9))
            text(ax, xx + 0.07, yb, str(r["n"]), size=5.4, color=LF, weight="bold", ha="center")
    bx += len(toks) * 0.17 + 0.22
text(ax, 0.10, yb - 0.20, "S  summary token", size=5.2, color=MUTED)

stages = [("Within each locus", "attention over the summary\n+ its variant tokens"),
          ("Across loci", "attention over the\nlocus summaries"),
          ("Readout", "MLP → P(resistant)")]
sx = bx + 0.10
sw = (W - 0.05 - sx - 2 * 0.22) / 3
arrow(ax, bx - 0.12, yb, sx - 0.03, yb)
for j, (h, s) in enumerate(stages):
    xx = sx + j * (sw + 0.22)
    box(ax, xx, yb - 0.30, sw, 0.60, fc=LF_TINT if j < 2 else SURFACE, ec=LF, lw=0.9, r=0.04)
    text(ax, xx + sw / 2, yb + 0.14, h, size=6.4, weight="bold", ha="center", color=LF if j < 2 else INK)
    text(ax, xx + sw / 2, yb - 0.09, s, size=5.6, color=INK2, ha="center", linespacing=1.25)
    if j < 2:
        arrow(ax, xx + sw + 0.03, yb, xx + sw + 0.19, yb)

OUTDIR.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUTDIR / f"fig_tokenization.{ext}", dpi=300)
print(f"wrote {OUTDIR / 'fig_tokenization.pdf'} and .png")

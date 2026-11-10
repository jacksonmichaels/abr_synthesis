"""Mechanism schematics and probe panels for the alternative GenomeFusion designs.
Used by notebooks/genomefusion/genomefusion_architectures.ipynb; trains nothing.

    python scripts/genomefusion_design_figures.py     # render all, for checking

Every colour comes from paper_style: neutrals plus copper for model parts. R and S
are told apart by filled vs open marks, so identity is never colour alone.
Schematic bar lengths, heat values and trees are illustrative, not fitted; the
notebook says so, and only `evidence` plots measured numbers.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from genomefusion.layout import RESIDUE_SYMBOLS  # noqa: E402
from paper_style import MODALITY, MODEL, NEUTRAL, diverging_cmap, ramp  # noqa: E402

N = NEUTRAL
ACCENT = MODEL["locusfusion"]
FIGDIR = PROJECT / "results" / "figures" / "genomefusion"
PROBES = PROJECT / "results" / "analysis" / "genomefusion_design_probes"
SHORT = {"AMIKACIN": "AMK", "CAPREOMYCIN": "CAP", "ETHAMBUTOL": "EMB", "ETHIONAMIDE": "ETO",
         "ISONIAZID": "INH", "KANAMYCIN": "KAN", "LEVOFLOXACIN": "LEV", "MOXIFLOXACIN": "MOX",
         "PYRAZINAMIDE": "PZA", "RIFAMPICIN": "RIF", "STREPTOMYCIN": "STR"}
HEAT = LinearSegmentedColormap.from_list("effect", [N["surface"], ACCENT])
BURDEN = LinearSegmentedColormap.from_list("burden", [N["surface"], N["ink2"]])


# ---- primitives: axes coordinates are inches, so geometry reads directly ------
def canvas(w, h, title):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.text(0.15, h - 0.12, title, fontsize=11.5, color=N["ink"], va="top")
    return fig, ax


def box(ax, x, y, w, h, text="", edge=None, fs=8.6, lw=1.2, face=None, color=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.06",
                                linewidth=lw, edgecolor=edge or N["axis"],
                                facecolor=face or N["surface"], zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=color or N["ink"], linespacing=1.3, zorder=3)


def arrow(ax, a, b, color=None, lw=1.0, style="-|>", zorder=1):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=9, linewidth=lw,
                                 color=color or N["ink2"], shrinkA=0, shrinkB=0, zorder=zorder))


def label(ax, x, y, text, fs=8.2, color=None, ha="center", va="center", **kw):
    ax.text(x, y, text, fontsize=fs, color=color or N["ink2"], ha=ha, va=va,
            linespacing=1.3, zorder=4, **kw)


def line(ax, xs, ys, color=None, lw=1.0, **kw):
    kw.setdefault("zorder", 1)
    ax.plot(xs, ys, color=color or N["axis"], lw=lw, solid_capstyle="round", **kw)


def save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=200, facecolor="white")
    return fig


# ---- the model in use, drawn large -----------------------------------------------
REG = MODALITY["regulatory"]
ACCENT_FILL, ACCENT_TINT = ramp(ACCENT, 4, light=0.95)[2:]
REG_FILL, REG_TINT = ramp(REG, 4, light=0.95)[2:]


def _token_example():
    return json.loads((FIGDIR / "token_example.json").read_text())


def _chip(ax, x, y, w=0.26, h=0.26, face=None, edge=None, text="", fs=11, color=None):
    ax.add_patch(Rectangle((x, y - h / 2), w, h, facecolor=face or ACCENT_FILL,
                           edgecolor=edge or ACCENT, linewidth=1.1, zorder=3))
    if text:
        label(ax, x + w / 2, y, text, fs=fs, color=color or edge or ACCENT, fontweight="bold")


def _panel(ax, x, y, w, h, head):
    box(ax, x, y, w, h, face="none")
    label(ax, x + 0.03, y + h + 0.14, head, fs=14, ha="left", va="baseline", color=N["ink"],
          fontweight="bold")


def current_architecture():
    ex = _token_example()
    by = {t["kind"]: t for t in ex["tokens"]}
    rows = [("katG", by["substitution"]["locus_tokens"], False),
            ("fabG1 promoter", by["promoter"]["locus_tokens"], True),
            (by["deletion"]["locus"], by["deletion"]["locus_tokens"], False)]
    W, H = 15.0, 9.0
    fig, ax = canvas(W, H, "")
    label(ax, 0.2, H - 0.25, "GenomeFusion architecture as built, whole-genome arm", fs=16, ha="left",
          color=N["ink"])
    top, bot = 3.45, 7.75                 # the pipeline band
    h = bot - top
    P1, P2, P3, P4, P5 = (0.2, 2.75), (3.25, 2.55), (6.1, 3.2), (9.6, 2.65), (12.55, 2.3)
    for (x0, w0), (x1, _) in zip((P1, P2, P3, P4), (P2, P3, P4, P5)):
        arrow(ax, (x0 + w0 + 0.03, top + h / 2), (x1 - 0.03, top + h / 2), lw=1.6)

    # 1 genome to loci
    x, w = P1
    _panel(ax, x, top, w, h, "1. Genome to loci")
    ys = (7.05, 6.3, 5.55)
    bx, bw = x + 0.18, w - 0.36
    for y, (name, frac, reg) in zip(ys, (("katG", 0.45, False), ("fabG1 promoter", 0.5, True),
                                         (rows[2][0], 0.3, False))):
        label(ax, bx, y + 0.27, name, fs=12.5, ha="left", color=REG if reg else N["ink"],
              fontstyle="normal" if reg or name.startswith("Rv") else "italic")
        ax.add_patch(Rectangle((bx, y - 0.08), bw, 0.16, zorder=2,
                               facecolor=REG_FILL if reg else N["grid"],
                               edgecolor=REG if reg else N["axis"], linewidth=0.9))
        line(ax, [bx + bw * frac] * 2, [y - 0.15, y + 0.15], color=N["ink"], lw=2.4, zorder=4)
    label(ax, bx, 4.3, "6,776 loci:\n3,749 genes +\n3,027 intergenic\none locus per base", fs=12.5,
          ha="left", color=N["ink2"])

    # 2 changes to tokens
    x, w = P2
    _panel(ax, x, top, w, h, "2. Changes to tokens")
    for y, name, reg in zip(ys, ("S315T", "c-15t", f"{by['deletion']['span']} bp deletion"),
                            (False, True, False)):
        label(ax, x + 0.18, y, name, fs=12.5, ha="left", color=REG if reg else N["ink"])
        _chip(ax, x + w - 0.5, y, face=REG_FILL if reg else ACCENT_FILL, edge=REG if reg else ACCENT)
    label(ax, x + 0.18, 4.3, "one token per changed\nbase or indel event\n"
          "57 channels each\n881 per isolate (mean)", fs=12.5, ha="left", color=N["ink2"])

    # 3 within each locus
    x, w = P3
    _panel(ax, x, top, w, h, "3. Within each locus")
    ys3 = (7.3, 6.85, 6.4)
    for y, (name, n, reg) in zip(ys3, rows):
        edge = REG if reg else ACCENT
        label(ax, x + 0.15, y, name, fs=11.5, ha="left", color=edge if reg else N["ink"])
        cx = x + 1.62
        _chip(ax, cx, y, face=N["surface"], edge=edge, text="∅")
        for k in range(min(n, 4)):
            _chip(ax, cx + 0.32 * (k + 1), y, face=REG_FILL if reg else ACCENT_FILL, edge=edge)
    arrow(ax, (x + w / 2, 6.14), (x + w / 2, 5.72), lw=1.3)
    box(ax, x + 0.35, 5.02, w - 0.7, 0.66, "shared Transformer × 2\nwith FiLM per locus", edge=ACCENT,
        face=ACCENT_FILL, fs=12.5)
    arrow(ax, (x + w / 2, 4.98), (x + w / 2, 4.62), lw=1.3)
    for k, reg in enumerate((False, True, False)):
        ax.scatter(x + w / 2 + (k - 1) * 0.42, 4.42, s=170, zorder=3, linewidths=1.1,
                   facecolor=REG_FILL if reg else ACCENT_FILL, edgecolor=REG if reg else ACCENT)
    label(ax, x + w / 2, 3.87, "∅ output: one vector per active locus\n707 per isolate (mean)", fs=11.5)

    # 4 across loci
    x, w = P4
    _panel(ax, x, top, w, h, "4. Across loci")
    _chip(ax, x + 0.42, 7.1, face=N["surface"], text="R", w=0.34, h=0.34, fs=12)
    for k, reg in enumerate((False, True, False)):
        ax.scatter(x + 1.08 + k * 0.42, 7.1, s=170, zorder=3, linewidths=1.1,
                   facecolor=REG_FILL if reg else ACCENT_FILL, edgecolor=REG if reg else ACCENT)
    arrow(ax, (x + w / 2, 6.8), (x + w / 2, 6.3), lw=1.3)
    box(ax, x + 0.3, 5.5, w - 0.6, 0.76, "Transformer × 2", edge=ACCENT, face=ACCENT_FILL, fs=12.5)
    label(ax, x + w / 2, 4.5, "R: learned reference\ntoken, always present", fs=12.5, color=N["ink2"])

    # 5 readout
    x, w = P5
    _panel(ax, x, top, w, h, "5. Readout")
    steps = [(6.72, "drug query\nattends over\nloci and R", N["surface"], N["axis"]),
             (5.42, "MLP", N["surface"], N["axis"]),
             (4.0, "P(resistant)", ACCENT_FILL, ACCENT)]
    for (yy, txt, face, edge), hh in zip(steps, (0.86, 0.6, 0.6)):
        box(ax, x + 0.22, yy, w - 0.44, hh, txt, edge=edge, face=face, fs=12.5)
    arrow(ax, (x + w / 2, 6.68), (x + w / 2, 6.06), lw=1.3)
    arrow(ax, (x + w / 2, 5.38), (x + w / 2, 4.64), lw=1.3)

    # locus identity
    bx0, by0, bw0, bh0 = 0.2, 0.72, W - 0.35, 2.1
    label(ax, bx0 + 0.03, by0 + bh0 + 0.2, "Locus identity, predicted from locus features", fs=14,
          ha="left", color=N["ink"], fontweight="bold")
    box(ax, bx0, by0, bw0, bh0, face="none")
    arrow(ax, (P3[0] + P3[1] / 2, by0 + bh0 + 0.04), (P3[0] + P3[1] / 2, top - 0.05), color=REG, lw=1.5)
    for k in range(10):
        _chip(ax, 0.42 + k * 0.17, 2.1, w=0.15, h=0.26)
        _chip(ax, 2.3 + k * 0.17, 2.1, w=0.15, h=0.26, face=REG_FILL, edge=REG)
    label(ax, 0.42, 2.46, "own: 29 features", fs=12, ha="left", color=N["ink"])
    label(ax, 2.3, 2.46, "regulated locus: 29", fs=12, ha="left", color=REG)
    label(ax, 0.42, 1.28, "region kind, category, essentiality,\nlength, position, strand, "
          "WHO-19, regulatory", fs=11.5, ha="left")

    ex0 = 4.5
    box(ax, ex0, 0.92, 4.1, 1.7, edge=REG, face=REG_TINT)
    _chip(ax, ex0 + 0.25, 2.2, w=1.6, h=0.38, face=REG_FILL, edge=REG, text="fabG1 promoter", fs=11)
    arrow(ax, (ex0 + 1.88, 2.2), (ex0 + 2.42, 2.2), color=REG, lw=1.6, zorder=4)
    _chip(ax, ex0 + 2.45, 2.2, w=0.85, h=0.38, text="fabG1", fs=11)
    label(ax, ex0 + 0.25, 1.42, "gated edge: + gate × target's residual\nWHO regions + next start codon:\n"
          "4,419 loci, one shared gate", fs=11.5, ha="left")

    px = 9.05
    for yy, name, target in ((2.15, "feat_proj", "locus embedding,\nadded to every token"),
                             (1.27, "film_proj", "FiLM scale and shift,\nper locus")):
        box(ax, px, yy - 0.27, 1.45, 0.54, name, edge=ACCENT, face=ACCENT_FILL, fs=12)
        arrow(ax, (px + 1.5, yy), (px + 1.95, yy), lw=1.3)
        label(ax, px + 2.03, yy, target, fs=12, ha="left")

    ax.legend(handles=[
        Patch(facecolor=ACCENT_FILL, edgecolor=ACCENT, label="token, locus or model part"),
        Patch(facecolor=REG_FILL, edgecolor=REG, label="regulatory locus or link"),
        Patch(facecolor=N["surface"], edgecolor=ACCENT, label="∅ [NULL] per-locus token; R reference token"),
        Line2D([], [], color=N["ink"], lw=2.4, label="a changed base"),
        Patch(facecolor=N["grid"], edgecolor=N["axis"], label="locus")],
        loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=11)
    return save(fig, "fig_genomefusion_current")


def _decode(t):
    f, S = t["feat"], t["slots"]
    hot = lambda slot, names: next((names[i] for i, v in enumerate(f[S[slot][0]:S[slot][1]]) if v == 1), None)
    bits = [f"base {'deleted' if t['base'] == '-' else t['base']}"]
    res = hot("residue", RESIDUE_SYMBOLS) if t["codon"] >= 0 else None
    if res:
        bits.append("residue deleted" if res == "-" else f"residue {res}")
    second = [c for c in (t["consequence"], hot("region", ("coding", "RNA gene", "intergenic"))) if c]
    third = []
    if t["truncation"] > 0:
        third.append(f"{t['truncation']:.0%} of CDS lost")
    if t["span"] > 1:
        third.append(f"span {t['span']} bp")
    if 0 <= t["up_plus"] <= 200:
        third.append(f"{t['up_plus']} bp upstream (+)")
    return "\n".join(s for s in (", ".join(bits), ", ".join(c.replace("_", " ") for c in second),
                                 ", ".join(third)) if s)


def tokenization():
    ex = _token_example()
    S = ex["slots"]
    W, H = 15.0, 10.0
    fig, ax = canvas(W, H, "")
    label(ax, 0.2, H - 0.25, f"GenomeFusion tokenization: isolate {ex['isolate']}, "
          f"{ex['n_tokens']} tokens in {ex['n_active_loci']} of {ex['n_loci']:,} loci", fs=16,
          ha="left", color=N["ink"])
    x0, cell, rh, ytop = 3.75, 0.148, 0.62, 7.72
    names = {"substitution": "katG S315T\nsubstitution in a gene",
             "promoter": "fabG1 c-15t\nsubstitution in a promoter"}
    t_ = {t["kind"]: t for t in ex["tokens"]}
    names["frameshift"] = f"{t_['frameshift']['locus']} codon {t_['frameshift']['codon'] + 1}\nframeshift indel"
    names["deletion"] = f"{t_['deletion']['locus']}\n{t_['deletion']['span']} bp in-frame deletion"
    M = np.array([t["feat"] for t in ex["tokens"]], float)
    cmap = diverging_cmap()
    nr, nc = M.shape
    im = ax.imshow(M, extent=(x0, x0 + nc * cell, ytop - nr * rh, ytop), cmap=cmap, vmin=-1.5, vmax=1.5,
                   aspect="auto", interpolation="nearest", zorder=2)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    for j in range(nc + 1):
        line(ax, [x0 + j * cell] * 2, [ytop - nr * rh, ytop], color=N["grid"], lw=0.5, zorder=3)
    for i in range(nr + 1):
        line(ax, [x0, x0 + nc * cell], [ytop - i * rh] * 2, color=N["axis"], lw=0.9, zorder=3)
    heads = [("base", "base"), ("residue", "residue"), ("bio", "properties"), ("d_bio", "Δ properties"),
             ("consequence", "consequence"), ("truncation", "truncation"), ("region", "region"),
             ("upstream", "upstream"), ("span", "span"), ("phase", "codon phase"), ("flags", "flags"),
             ("is_null", "[NULL]")]
    for k, (slot, lab) in enumerate(heads):
        a, b = S[slot]
        c = N["ink2"] if k % 2 else N["muted"]
        ax.add_patch(Rectangle((x0 + a * cell, ytop + 0.03), (b - a) * cell, 0.09, facecolor=c,
                               edgecolor=N["surface"], linewidth=0.8, zorder=3))
        line(ax, [x0 + a * cell] * 2, [ytop - nr * rh, ytop], color=N["ink2"], lw=1.1, zorder=4)
        label(ax, x0 + (a + b) / 2 * cell, ytop + 0.18, f"{lab} ({b - a})", fs=11, rotation=62, ha="left",
              va="bottom", rotation_mode="anchor", color=N["ink"])
    line(ax, [x0 + nc * cell] * 2, [ytop - nr * rh, ytop], color=N["ink2"], lw=1.1, zorder=4)
    label(ax, 0.2, ytop + 0.3, "What changed", fs=14, ha="left", color=N["ink"], fontweight="bold")
    label(ax, x0 + nc * cell + 0.36, ytop + 0.3, "What the token says", fs=13.5, ha="left", color=N["ink"],
          fontweight="bold")
    for i, t in enumerate(ex["tokens"]):
        y = ytop - (i + 0.5) * rh
        head, sub = names[t["kind"]].split("\n")
        label(ax, 0.2, y + 0.14, head, fs=12.5, ha="left", color=N["ink"], fontweight="bold")
        label(ax, 0.2, y - 0.14, sub, fs=11, ha="left")
        arrow(ax, (x0 - 0.42, y), (x0 - 0.05, y), lw=1.3)
        label(ax, x0 + nc * cell + 0.12, y, _decode({**t, "slots": S}), fs=10.5, ha="left")
    y = ytop - nr * rh - 0.32
    n_skip = ex["n_loci"] - ex["n_active_loci"]
    label(ax, 0.2, y, f"{n_skip:,} unchanged loci", fs=12.5, ha="left", color=N["ink"], fontweight="bold")
    arrow(ax, (x0 - 0.42, y), (x0 - 0.05, y), lw=1.3)
    label(ax, x0 + 0.05, y, "no tokens", fs=12.5, ha="left")
    cax = fig.add_axes([(x0 + nc * cell - 3.2) / W, (y - 0.09) / H, 3.2 / W, 0.14 / H])
    fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cax.tick_params(labelsize=10, length=2)
    cax.set_xlabel("channel value (one-hot channels are 1; white is 0)", fontsize=10.5, labelpad=2)

    # the old token layout beside the new one
    band = 0.3
    label(ax, 0.2, 3.72, "LocusFusion v3: fixed grid", fs=14, ha="left", color=N["ink"], fontweight="bold")
    label(ax, 7.75, 3.72, "GenomeFusion: only what changed", fs=14, ha="left", color=N["ink"], fontweight="bold")
    box(ax, 0.2, band, 7.25, 3.15)
    box(ax, 7.75, band, 7.05, 3.15)
    rng = np.random.default_rng(3)
    c, gx, gy = 0.105, 0.5, 3.05
    for i in range(19):
        n = min(int(rng.geometric(0.55)), 16)
        for j in range(16):
            ax.add_patch(Rectangle((gx + j * c, gy - (i + 1) * c), c, c, linewidth=0.4, zorder=2,
                                   edgecolor=N["grid"] if j >= n else ACCENT,
                                   facecolor=N["surface"] if j >= n else ACCENT_FILL))
    label(ax, gx + 8 * c, gy + 0.11, "16 token slots", fs=10.5)
    label(ax, gx - 0.13, gy - 9.5 * c, "19 loci", fs=10.5, rotation=90)
    label(ax, 2.5, 1.85, "19 loci, each always present\nat most 16 tokens per locus\n"
          "read from per-gene alignments\nsynonymous changes kept\n"
          "regulatory bases in their own slot", fs=12, ha="left", color=N["ink"])
    ragged = [("katG", t_["substitution"]["locus_tokens"], False),
              ("Rv1482c-Rv1483", t_["promoter"]["locus_tokens"], True),
              (t_["deletion"]["locus"], t_["deletion"]["locus_tokens"], False),
              (ex["busiest"]["locus"], ex["busiest"]["n"], False)]
    for i, (name, n, reg) in enumerate(ragged):
        y = 2.92 - i * 0.42
        label(ax, 7.95, y, name, fs=11, ha="left", color=REG if reg else N["ink"])
        for k in range(min(n, 3)):
            _chip(ax, 9.4 + k * 0.27, y, w=0.23, h=0.23, face=REG_FILL if reg else ACCENT_FILL,
                  edge=REG if reg else ACCENT)
        if n > 3:
            label(ax, 9.4 + 3 * 0.27 + 0.05, y, f"{n} tokens", fs=10.5, ha="left")
    label(ax, 7.95, 1.2, f"⋮  {ex['n_active_loci'] - 4} more of this isolate's active loci", fs=11, ha="left")
    label(ax, 11.05, 2.2, f"6,776 loci, only changed ones enter\nno limit on tokens per locus\n"
          "read from the VCFs\nsynonymous changes dropped\none base slot; region says where",
          fs=11.5, ha="left", color=N["ink"])
    label(ax, 7.95, 0.68, "new channels: consequence, truncation, region, upstream, span, flags",
          fs=11.5, ha="left", color=N["ink"])
    return save(fig, "fig_genomefusion_tokenization")


# ---- overview ------------------------------------------------------------------
def overview():
    cols = ["locus\ngrouping", "within-locus\nencoder", "across-loci\nattention",
            "drug-query\nreadout", "57-channel\ntoken", "isolate\nBCE", "single-drug\nlabels"]
    rows = [("GenomeFusion (built)", "1111111"), ("1. Effect Field", "1000111"),
            ("2. Clade Delta", "1000101"), ("3. Retriever-Reader", "0001111"),
            ("4. Genome Pyramid", "0001111"), ("5. Resistome Factorisation", "1000110"),
            ("6. Pattern Forest", "1000011")]
    fig, ax = canvas(9.4, 3.9, "Components of GenomeFusion retained or replaced, by design")
    x0, dx, y0, dy = 2.7, 0.94, 2.72, 0.36
    for j, c in enumerate(cols):
        label(ax, x0 + j * dx, 3.2, c, fs=8.2)
    for i, (name, flags) in enumerate(rows):
        y = y0 - i * dy
        label(ax, 0.2, y, name, fs=9, ha="left", color=N["ink"])
        line(ax, [0.2, x0 + 6 * dx + 0.35], [y - dy / 2] * 2, color=N["grid"], lw=0.8)
        for j, f in enumerate(flags):
            ax.scatter(x0 + j * dx, y, s=62, marker="o", zorder=3, linewidths=1.2,
                       facecolor=N["ink2"] if f == "1" else N["surface"], edgecolor=N["ink2"])
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=7, mfc=N["ink2"], mec=N["ink2"], label="retained"),
        Line2D([], [], marker="o", ls="", ms=7, mfc=N["surface"], mec=N["ink2"], label="replaced")],
        loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=2, fontsize=8.4, handletextpad=0.2)
    return save(fig, "fig_design_overview")


# ---- 1. effect field -----------------------------------------------------------
def effect_field():
    fig, ax = canvas(10.4, 5.1, "1. Effect Field: per-token resistance effects combined by noisy-OR")
    toks = [("rpoB S450L", .86), ("rpoC V483G", .20), ("katG R463L", .03), ("gyrA S95T", .02),
            ("~870 more", .05)]
    ys = [4.0, 3.62, 3.24, 2.86, 2.48]
    label(ax, 1.05, 4.42, "tokens in one isolate")
    label(ax, 5.95, 4.42, "$p_v$ per token")
    for (t, p), y in zip(toks, ys):
        box(ax, 0.3, y - 0.14, 1.5, 0.28, t, fs=8.4, color=N["ink"] if "more" not in t else N["ink2"])
        arrow(ax, (1.87, y), (2.42, 3.24 + (y - 3.24) * .55), color=N["axis"])
        arrow(ax, (4.53, 3.24 + (y - 3.24) * .55), (5.05, y), color=N["axis"])
        ax.add_patch(Rectangle((5.1, y - 0.1), 1.65 * p, 0.2, facecolor=N["ink2"], zorder=2))
        line(ax, [5.1, 5.1], [y - 0.15, y + 0.15], color=N["axis"])
        arrow(ax, (5.1 + 1.65 * p + 0.05, y), (7.0, 3.24 + (y - 3.24) * .4), color=N["grid"])
    box(ax, 2.45, 2.42, 2.05, 1.64,
        "effect network\n\n57 token channels\n+ locus features\n+ hierarchy terms\n$\\rightarrow$ $e_d(v)$",
        edge=ACCENT, lw=1.6)
    box(ax, 7.03, 2.64, 1.95, 1.2,
        "noisy-OR\n\n$P(R) = 1 - (1-\\lambda)\\prod_v (1-p_v)$", edge=ACCENT, lw=1.6, fs=7.9)
    arrow(ax, (9.03, 3.24), (9.42, 3.24))
    box(ax, 9.45, 2.96, 0.72, 0.56, "$P(R)$", fs=9.5)

    # the hierarchy
    label(ax, 0.3, 1.98, "hierarchy of effect terms", fs=9, ha="left", color=N["ink"])
    nodes = {"genome": (0.8, 1.05), "rpoB": (2.25, 1.45), "pncA": (2.25, 0.62),
             "codon 450": (3.85, 1.45), "codon 445": (3.85, 0.95),
             "S450L": (5.4, 1.45), "S450W": (5.4, 0.95)}
    edges = [("genome", "rpoB"), ("genome", "pncA"), ("rpoB", "codon 450"), ("rpoB", "codon 445"),
             ("codon 450", "S450L"), ("codon 450", "S450W")]
    path = {("genome", "rpoB"), ("rpoB", "codon 450"), ("codon 450", "S450L")}
    for a, b in edges:
        on = (a, b) in path
        line(ax, [nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]],
             color=ACCENT if on else N["axis"], lw=1.8 if on else 1.0)
    on_path = {"genome", "rpoB", "codon 450", "S450L"}
    for k, (x, y) in nodes.items():
        box(ax, x - 0.46, y - 0.15, 0.92, 0.3, k, fs=8.2,
            edge=ACCENT if k in on_path else N["axis"], lw=1.5 if k in on_path else 1.0)
    label(ax, 6.2, 1.2, "$\\beta_{S450L} \\; = \\; \\beta_{genome} + \\beta_{rpoB} "
          "+ \\beta_{codon\\,450} + \\beta_{allele}$", fs=9.6, ha="left", color=N["ink"])
    return save(fig, "fig_design_effect_field")


# ---- 2. clade delta -------------------------------------------------------------
def clade_delta():
    fig, ax = canvas(10.4, 4.5, "2. Clade Delta: training on discordant pairs within a sublineage")
    # tree
    label(ax, 1.4, 3.92, "one level-2 sublineage")
    leaves = [("S", 3.5), ("S", 3.15), ("R", 2.8), ("S", 2.45), ("R", 2.1), ("S", 1.75)]
    X_LEAF = 2.3
    joins = [(0, 1, 1.7), (2, 3, 1.95), (4, 5, 1.55)]
    mids = []
    for a, b, x in joins:
        ya, yb = leaves[a][1], leaves[b][1]
        pair = (a, b) == (2, 3)
        col = ACCENT if pair else N["axis"]
        lw = 1.8 if pair else 1.0
        line(ax, [x, X_LEAF], [ya, ya], color=col, lw=lw)
        line(ax, [x, X_LEAF], [yb, yb], color=col, lw=lw)
        line(ax, [x, x], [ya, yb], color=col, lw=lw)
        mids.append((x, (ya + yb) / 2))
    line(ax, [1.0, mids[0][0]], [mids[0][1]] * 2)
    line(ax, [1.0, mids[1][0]], [mids[1][1]] * 2)
    line(ax, [1.0, 1.0], [mids[0][1], mids[1][1]])
    ym = (mids[0][1] + mids[1][1]) / 2
    line(ax, [0.5, 1.0], [ym, ym])
    line(ax, [0.5, mids[2][0]], [mids[2][1]] * 2)
    line(ax, [0.5, 0.5], [ym, mids[2][1]])
    line(ax, [0.25, 0.5], [(ym + mids[2][1]) / 2] * 2)
    for t, y in leaves:
        ax.scatter(X_LEAF + 0.07, y, s=44, zorder=3, linewidths=1.2, edgecolor=N["ink"],
                   facecolor=N["ink"] if t == "R" else N["surface"])
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=6.5, mfc=N["ink"], mec=N["ink"], label="resistant"),
        Line2D([], [], marker="o", ls="", ms=6.5, mfc=N["surface"], mec=N["ink"], label="susceptible"),
        Line2D([], [], color=ACCENT, lw=1.8, label="nearest discordant pair")],
        loc="lower left", bbox_to_anchor=(0.012, 0.025), fontsize=8, ncol=1, labelspacing=0.3)

    # tracks
    rng = np.random.default_rng(3)
    x0, x1 = 3.05, 6.9
    yR, yS = 3.15, 2.45
    shared = rng.uniform(x0 + .05, x1 - .05, 46)
    Ronly = rng.uniform(x0 + .1, x1 - .1, 7)
    Sonly = rng.uniform(x0 + .1, x1 - .1, 4)
    label(ax, (x0 + x1) / 2, 3.92, "tokens along the genome")
    label(ax, x0, yR + 0.33, "resistant isolate", ha="left")
    label(ax, x0, yS + 0.33, "susceptible isolate", ha="left")
    for y in (yR, yS):
        line(ax, [x0, x1], [y, y], color=N["axis"], lw=1)
        ax.vlines(shared, y - 0.1, y + 0.1, color=N["axis"], lw=1, zorder=2)
    ax.vlines(Ronly, yR - 0.17, yR + 0.17, color=N["ink"], lw=1.8, zorder=3)
    ax.scatter(Sonly, [yS] * len(Sonly), s=30, facecolor=N["surface"], edgecolor=N["ink"],
               linewidths=1.2, zorder=3)
    leg_y = 1.78
    ax.vlines([x0 + 0.05], leg_y - 0.08, leg_y + 0.08, color=N["axis"], lw=1)
    label(ax, x0 + 0.13, leg_y, "shared", ha="left")
    ax.vlines([x0 + 0.85], leg_y - 0.12, leg_y + 0.12, color=N["ink"], lw=1.8)
    label(ax, x0 + 0.93, leg_y, "R-only", ha="left")
    ax.scatter([x0 + 1.7], [leg_y], s=30, facecolor=N["surface"], edgecolor=N["ink"], linewidths=1.2)
    label(ax, x0 + 1.8, leg_y, "S-only", ha="left")

    # loss
    box(ax, 7.3, 3.02, 0.98, 0.42, "$f$(R-only)", edge=ACCENT, lw=1.5)
    box(ax, 7.3, 2.28, 0.98, 0.42, "$f$(S-only)", edge=ACCENT, lw=1.5)
    arrow(ax, (6.95, yR), (7.27, 3.23))
    arrow(ax, (6.95, yS), (7.27, 2.49))
    box(ax, 8.62, 2.6, 1.55, 0.52, "pair loss\n$-\\log\\sigma(f_R - f_S)$", fs=8.2)
    arrow(ax, (8.31, 3.23), (8.6, 2.98))
    arrow(ax, (8.31, 2.49), (8.6, 2.74))
    # scoring
    label(ax, 3.05, 1.06, "scoring", ha="left", fs=9, color=N["ink"])
    box(ax, 3.05, 0.42, 2.25, 0.42, "any isolate, all tokens")
    box(ax, 5.75, 0.42, 1.25, 0.42, "$f$ (same weights)", edge=ACCENT, lw=1.5, fs=8.2)
    box(ax, 7.45, 0.42, 1.0, 0.42, "score")
    arrow(ax, (5.34, 0.63), (5.72, 0.63))
    arrow(ax, (7.04, 0.63), (7.42, 0.63))
    return save(fig, "fig_design_clade_delta")


# ---- 3. retriever-reader ---------------------------------------------------------
def retriever_reader():
    fig, ax = canvas(10.4, 4.6, "3. Retriever-Reader: selecting a fixed number of tokens per isolate")
    rng = np.random.default_rng(1)
    ytop = 3.1
    # all tokens
    label(ax, 0.95, 3.92, "all tokens, ~875")
    xs = rng.uniform(0.25, 1.65, 120)
    ys = rng.uniform(ytop - 0.42, ytop + 0.42, 120)
    ax.vlines(xs, ys - 0.05, ys + 0.05, color=N["axis"], lw=0.9)
    arrow(ax, (1.72, ytop), (2.03, ytop))
    box(ax, 2.05, ytop - 0.55, 1.6, 1.1,
        "retriever\n\nper-drug scorer\nfit on the training fold", edge=ACCENT, lw=1.6, fs=8)
    arrow(ax, (3.7, ytop), (3.98, ytop))
    # ranked scores
    label(ax, 5.0, 3.92, "tokens ranked by score")
    n = 110
    h = 0.95 * np.exp(-np.arange(n) / 16) * (1 + 0.1 * rng.normal(size=n))
    bx = np.linspace(4.05, 5.95, n)
    keep = 8
    ax.vlines(bx[keep:], ytop - 0.5, ytop - 0.5 + h[keep:], color=N["axis"], lw=1.1)
    ax.vlines(bx[:keep], ytop - 0.5, ytop - 0.5 + h[:keep], color=N["ink"], lw=1.5)
    label(ax, bx[keep // 2], ytop - 0.68, "top 64", color=N["ink"])
    line(ax, [bx[0], bx[keep - 1]], [ytop - 0.55] * 2, color=N["ink"], lw=1)
    arrow(ax, (6.02, ytop), (6.3, ytop))
    # reader: all-to-all among kept tokens, plus a drug query
    label(ax, 7.0, 3.92, "reader")
    k = 11
    th = np.linspace(0, 2 * np.pi, k, endpoint=False) + .2
    cx, cy, r = 7.0, ytop, 0.5
    px, py = cx + r * np.cos(th), cy + r * np.sin(th)
    for i in range(k):
        for j in range(i + 1, k):
            line(ax, [px[i], px[j]], [py[i], py[j]], color=N["grid"], lw=0.7)
    ax.scatter(px, py, s=26, color=N["ink"], zorder=3)
    arrow(ax, (7.62, ytop), (7.92, ytop))
    box(ax, 7.95, ytop - 0.3, 1.1, 0.6, "drug query", edge=ACCENT, lw=1.6)
    arrow(ax, (9.09, ytop), (9.4, ytop))
    box(ax, 9.43, ytop - 0.24, 0.72, 0.48, "logit")
    # attention sizes, area to scale
    label(ax, 0.25, 1.92, "attention pairs per isolate", fs=9, ha="left", color=N["ink"])
    s = 1.55
    ax.add_patch(Rectangle((0.3, 0.18), s, s, facecolor=N["grid"], edgecolor=N["axis"], lw=1))
    s2 = s * 64 / 707
    ax.add_patch(Rectangle((3.55, 0.18), s2, s2, facecolor=N["ink2"], edgecolor=N["ink2"], lw=1))
    label(ax, 1.95, 0.42, "across-loci stage\n707 x 707", ha="left")
    label(ax, 3.55 + s2 + 0.1, 0.42, "reader\n64 x 64", ha="left")
    return save(fig, "fig_design_retriever_reader")


# ---- 4. genome pyramid ------------------------------------------------------------
def genome_pyramid():
    fig, ax = canvas(10.4, 5.2, "4. Genome Pyramid: sparse multi-scale bins over genome coordinates")
    rng = np.random.default_rng(5)
    x0, x1 = 1.05, 8.05
    pos = np.concatenate([rng.uniform(0, 1, 26), rng.normal(0.27, 0.0035, 5),
                          rng.normal(0.72, 0.006, 4)]).clip(0, 0.9999)
    hot = np.arange(26, 31)
    widths = ["3 bp", "48 bp", "768 bp", "12 kb", "197 kb", "genome"]
    counts = [1024, 256, 64, 16, 4, 1]
    y0, dy, bh = 1.0, 0.56, 0.2
    px = lambda v: x0 + (x1 - x0) * v  # noqa: E731
    label(ax, 0.2, 0.52, "tokens", ha="left")
    line(ax, [x0, x1], [0.52, 0.52], color=N["axis"])
    ax.vlines(px(pos), 0.42, 0.62, color=N["axis"], lw=1.2, zorder=2)
    ax.vlines(px(pos[hot]), 0.38, 0.66, color=N["ink"], lw=1.6, zorder=3)
    prev = {i: px(p) for i, p in enumerate(pos)}
    prev_y = 0.64
    centres = []
    hot_bins = set(np.unique((pos[hot] * counts[0]).astype(int)))
    for lv, (c, w) in enumerate(zip(counts, widths)):
        y = y0 + lv * dy
        label(ax, 0.2, y, w, ha="left")
        bins = np.unique((pos * c).astype(int))
        hb = set(np.unique((pos[hot] * c).astype(int)))
        for i, p in enumerate(pos):
            b = int(p * c)
            xb = px((b + .5) / c)
            is_hot = i in hot
            line(ax, [prev[i], xb], [prev_y, y - bh / 2],
                 color=ACCENT if is_hot else N["grid"], lw=1.3 if is_hot else 0.7)
            prev[i] = xb
        for b in bins:
            bw = max((x1 - x0) / c, 0.055)
            xb = px((b + .5) / c) - bw / 2
            on = b in hb
            ax.add_patch(Rectangle((xb, y - bh / 2), bw, bh, zorder=3, linewidth=1,
                                   facecolor=N["surface"] if not on else ACCENT,
                                   edgecolor=ACCENT if on else N["axis"]))
            centres.append((xb + bw / 2, y, lv))
        prev_y = y + bh / 2
    label(ax, 0.2, y0 + 5 * dy + 0.42, "bin width", ha="left", fs=8.2)
    # drug latents
    lat_x = 9.0
    lats = [3.7, 3.0, 2.3]
    pick = [centres[i] for i in rng.choice(len(centres), 16, replace=False)]
    for (cx, cy, _), ly in zip(pick, np.tile(lats, 6)):
        line(ax, [cx, lat_x], [cy, ly], color=N["grid"], lw=0.6)
    for y_ in lats:
        ax.scatter(lat_x, y_, s=120, facecolor=N["surface"], edgecolor=ACCENT, linewidths=1.6, zorder=4)
    label(ax, lat_x, 4.12, "drug latents")
    arrow(ax, (lat_x + 0.17, 3.0), (9.47, 3.0))
    box(ax, 9.5, 2.76, 0.7, 0.48, "logit")
    return save(fig, "fig_design_genome_pyramid")


# ---- 5. resistome factorisation ------------------------------------------------------
def resistome():
    drugs = ["AMK", "CAP", "EMB", "ETO", "INH", "KAN", "LEV", "MOX", "PZA", "RIF", "STR"]
    variants = ["gyrA D94G", "gyrA A90V", "rrs a1401g", "fabG1 c-15t", "eis c-14t", "katG S315T",
                "rpoB S450L", "pncA H57D", "embB M306V", "rpsL K43R"]
    d = {k: i for i, k in enumerate(drugs)}
    W = np.zeros((3, 11))
    for k, v in {"LEV": 1, "MOX": .9}.items():
        W[0, d[k]] = v
    for k, v in {"AMK": 1, "KAN": 1, "CAP": .8}.items():
        W[1, d[k]] = v
    for k, v in {"ETO": 1, "INH": .7}.items():
        W[2, d[k]] = v
    U = np.zeros((10, 3))
    U[0, 0], U[1, 0], U[2, 1], U[3, 2] = .95, .75, .95, .85
    S = np.zeros((10, 11))
    for (i, k), v in {(4, "KAN"): .7, (5, "INH"): .95, (6, "RIF"): .95, (7, "PZA"): .85,
                      (8, "EMB"): .7, (9, "STR"): .9}.items():
        S[i, d[k]] = v
    B = U @ W + S

    fig, ax = canvas(10.4, 6.2, "5. Resistome Factorisation: variant-by-drug effects, shared plus drug-specific")

    def heat(x, y, w, h, M, title, xt=None, yt=None):
        a = fig.add_axes([x / 10.4, y / 6.2, w / 10.4, h / 6.2])
        a.imshow(M, cmap=HEAT, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        a.set_xticks(np.arange(M.shape[1] + 1) - .5, minor=True)
        a.set_yticks(np.arange(M.shape[0] + 1) - .5, minor=True)
        a.grid(which="minor", color=N["grid"], lw=0.6)
        a.tick_params(which="both", length=0, labelsize=6.8)
        a.set_xticks(range(M.shape[1]) if xt else [])
        a.set_yticks(range(M.shape[0]) if yt else [])
        if xt:
            a.set_xticklabels(xt, rotation=90)
            a.xaxis.tick_top()
        if yt:
            a.set_yticklabels(yt)
        for sp in a.spines.values():
            sp.set_color(N["axis"])
        a.set_xlabel(title, fontsize=8.6, color=N["ink2"], labelpad=4)
        return a

    top, ht = 3.0, 2.3
    heat(1.05, top, 2.75, ht, B, "effect $B$  (variants x drugs)", drugs, variants)
    label(ax, 4.03, top + ht / 2, "=", fs=15, color=N["ink"])
    heat(4.3, top, 0.75, ht, U, "$U$  (variants x r)", ["r1", "r2", "r3"])
    label(ax, 5.25, top + ht / 2, "x", fs=11, color=N["ink"])
    heat(5.72, top + 0.72, 1.72, 0.86, W, "$W$  (r x drugs)", drugs, ["r1", "r2", "r3"])
    label(ax, 7.62, top + ht / 2, "+", fs=15, color=N["ink"])
    heat(7.85, top, 2.4, ht, S, "sparse $S$  (variants x drugs)", drugs)

    # the label mask
    label(ax, 0.2, 2.42, "labels when LEV is the target drug", fs=9, ha="left", color=N["ink"])
    cols = ["LEV", "MOX", "KAN"]
    roles = ["train"] * 5 + ["val"] * 2 + ["test"] * 2
    x_start, y_start, cw, ch = 1.1, 0.28, 0.5, 0.2
    rng = np.random.default_rng(2)
    for j, c in enumerate(cols):
        label(ax, x_start + (j + .5) * cw, y_start + 9 * ch + 0.11, c, fs=7.6)
    for i, r in enumerate(roles[::-1]):
        y = y_start + i * ch
        for j in range(3):
            target = j == 0
            held = r != "train"
            have = target and not held or (not target and rng.random() < .8)
            if target and held:
                f, hatch, e = N["surface"], None, N["ink"]
            elif not target and held and have:
                f, hatch, e = N["surface"], "////", N["muted"]
            elif have:
                f, hatch, e = (ACCENT if target else N["axis"]), None, N["surface"]
            else:
                f, hatch, e = N["surface"], None, N["grid"]
            ax.add_patch(Rectangle((x_start + j * cw, y), cw - .02, ch - .02, facecolor=f,
                                   hatch=hatch, edgecolor=e, lw=0.8, zorder=2))
    for r, yc in (("test", 0), ("val", 2), ("train", 4)):
        n_ = {"test": 2, "val": 2, "train": 5}[r]
        label(ax, x_start - 0.1, y_start + (yc + n_ / 2) * ch, r, ha="right")
    ax.legend(handles=[
        Rectangle((0, 0), 1, 1, fc=ACCENT, ec=ACCENT, label="target label, trained on"),
        Rectangle((0, 0), 1, 1, fc=N["axis"], ec=N["axis"], label="auxiliary label, trained on"),
        Rectangle((0, 0), 1, 1, fc=N["surface"], ec=N["ink"], label="target label, held out"),
        Rectangle((0, 0), 1, 1, fc=N["surface"], ec=N["muted"], hatch="////", label="auxiliary label, masked"),
        Rectangle((0, 0), 1, 1, fc=N["surface"], ec=N["grid"], label="no phenotype")],
        loc="lower left", bbox_to_anchor=(0.31, 0.04), fontsize=8.2, ncol=1, labelspacing=0.35)
    return save(fig, "fig_design_resistome")


# ---- 6. pattern forest --------------------------------------------------------------
def _tree(ax, x, y, w, h, leaf_on):
    root = (x + w / 2, y + h)
    mids = [(x + w * .25, y + h * .5), (x + w * .75, y + h * .5)]
    leaves = [(x + w * t, y) for t in (.1, .4, .6, .9)]
    for m in mids:
        line(ax, [root[0], m[0]], [root[1], m[1]], color=N["axis"])
    for i, lf in enumerate(leaves):
        m = mids[i // 2]
        line(ax, [m[0], lf[0]], [m[1], lf[1]], color=N["axis"])
    for p in [root, *mids]:
        ax.scatter(*p, s=34, facecolor=N["surface"], edgecolor=ACCENT, linewidths=1.4, zorder=3)
    for i, lf in enumerate(leaves):
        ax.scatter(*lf, s=30, zorder=3, facecolor=N["ink2"] if i == leaf_on else N["surface"],
                   edgecolor=N["ink2"], linewidths=1.2, marker="s")


def pattern_forest():
    fig, ax = canvas(10.4, 3.7, "6. Pattern Forest: boosted trees over deduplicated variant patterns")
    rng = np.random.default_rng(7)
    n_iso = 9
    pats = [rng.random(n_iso) < p for p in (.5, .35, .2, .7, .15)]
    order = [0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4]
    cw, ch = 0.2, 0.2

    def grid(x, y, cols, fill):
        for j, c in enumerate(cols):
            for i in range(n_iso):
                ax.add_patch(Rectangle((x + j * cw, y + i * ch), cw - .02, ch - .02,
                                       facecolor=fill[j] if c[i] else N["surface"],
                                       edgecolor=N["grid"], lw=0.6, zorder=2))

    y = 0.75
    x_all = 0.55
    grid(x_all, y, [pats[k] for k in order], [N["ink2"]] * len(order))
    label(ax, x_all + 6 * cw, y + 9 * ch + 0.17, "variants, 636,869")
    label(ax, x_all - 0.12, y + 4.5 * ch, "isolates", rotation=90)
    # brackets over identical carrier sets
    k = 0
    for run in (3, 1, 2, 4, 2):
        xa, xb = x_all + k * cw, x_all + (k + run) * cw - .02
        line(ax, [xa, xb], [y - 0.07, y - 0.07], color=N["ink2"], lw=1.2)
        k += run
    label(ax, x_all + 6 * cw, y - 0.26, "identical carrier sets")
    arrow(ax, (x_all + 12 * cw + 0.08, y + 4.5 * ch), (3.42, y + 4.5 * ch))

    # patterns + burden
    x_p = 3.5
    grid(x_p, y, pats, [N["ink2"]] * 5)
    label(ax, x_p + 2.5 * cw, y + 9 * ch + 0.17, "patterns\n112,266", fs=8)
    x_b = x_p + 5 * cw + 0.15
    burden = [rng.random(n_iso) for _ in range(3)]
    for j, c in enumerate(burden):
        for i in range(n_iso):
            ax.add_patch(Rectangle((x_b + j * cw, y + i * ch), cw - .02, ch - .02, zorder=2,
                                   facecolor=BURDEN(float(c[i])), edgecolor=N["grid"], lw=0.6))
    label(ax, x_b + 1.5 * cw, y + 9 * ch + 0.17, "burden\ncounts", fs=8)
    label(ax, x_p + 4.25 * cw, y - 0.26, "by locus, codon, consequence", fs=8)
    arrow(ax, (x_b + 3 * cw + 0.08, y + 4.5 * ch), (5.65, y + 4.5 * ch))

    # boosted trees
    tx, tw = 5.75, 0.82
    for i, on in enumerate((0, 2, 3)):
        _tree(ax, tx + i * (tw + 0.28), y + 0.4, tw, 1.0, on)
        if i < 2:
            label(ax, tx + i * (tw + 0.28) + tw + 0.14, y + 0.9, "+", fs=12, color=N["ink"])
    label(ax, tx + 1.5 * tw + 0.28, y + 9 * ch + 0.17, "boosted trees")
    arrow(ax, (tx + 3 * tw + 0.65, y + 0.9), (9.3, y + 0.9))
    box(ax, 9.33, y + 0.65, 0.82, 0.5, "logit")
    return save(fig, "fig_design_pattern_forest")


# ---- probe evidence ------------------------------------------------------------------
def _axstyle(a):
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.tick_params(labelsize=8.2, length=0)
    a.grid(axis="x", color=N["grid"], lw=0.7)
    a.set_axisbelow(True)


def _dumbbell(a, names, left, right, lab_l, lab_r, mfc_l, mfc_r):
    y = np.arange(len(names))[::-1]
    a.hlines(y, left, right, color=N["axis"], lw=1.4, zorder=1)
    a.scatter(left, y, s=48, facecolor=mfc_l, edgecolor=N["ink2"], linewidths=1.2, zorder=3, label=lab_l)
    a.scatter(right, y, s=48, facecolor=mfc_r, edgecolor=N["ink2"], linewidths=1.2, zorder=3, label=lab_r)
    a.set_yticks(y, names)
    a.set_ylim(-0.7, len(names) - 0.3)


def evidence(baselines=None):
    voc = pd.read_csv(PROBES / "vocabulary.csv")
    pairs = pd.read_csv(PROBES / "pairs.csv")
    tr = pd.read_csv(PROBES / "transfer.csv").groupby("target")[["own", "pooled"]].mean()
    cf = pd.read_csv(PROBES / "closed_form.csv")
    base = pd.read_csv(PROBES / "baselines.csv", index_col=0).mean()

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))
    (a, b), (c, d) = axes

    y = np.arange(len(voc))[::-1]
    a.barh(y + .19, voc.pct_variants, height=.34, color=N["ink2"], label="% of variants")
    a.barh(y - .19, voc.pct_tokens, height=.34, color=N["axis"], label="% of tokens")
    for yi, v, t in zip(y, voc.pct_variants, voc.pct_tokens):
        a.text(v + 1, yi + .19, f"{v:.1f}", va="center", fontsize=7.8, color=N["ink2"])
        a.text(t + 1, yi - .19, f"{t:.1f}", va="center", fontsize=7.8, color=N["ink2"])
    a.set_yticks(y, voc.allele_count)
    a.set_ylabel("allele count across 17,943 isolates", fontsize=8.6)
    a.set_xlabel("percent", fontsize=8.6)
    a.set_xlim(0, 85)
    a.set_title("Variants and tokens by allele count", fontsize=10, loc="left")
    a.legend(fontsize=8.2, loc="center right")
    _axstyle(a)

    p = pairs.assign(s=pairs.drug.map(SHORT)).sort_values("own_locus_R_only", ascending=False)
    _dumbbell(b, p.s, 100 * p.own_locus_S_only, 100 * p.own_locus_R_only, "S-only side", "R-only side",
              N["surface"], N["ink2"])
    b.set_xlabel("% of pairs with a variant in the drug's own loci", fontsize=8.6)
    b.set_xlim(0, 100)
    b.set_title("Nearest discordant pairs, same level-2 sublineage", fontsize=10, loc="left")
    b.legend(fontsize=8.2, loc="lower right")
    _axstyle(b)

    t = tr.assign(s=[SHORT[i] for i in tr.index], gain=tr.pooled - tr.own).sort_values("gain", ascending=False)
    _dumbbell(c, t.s, t.own, t.pooled, "own labels", "pooled with related drug", N["surface"], N["ink2"])
    c.set_xlabel("CV AUC, top-30 stratified score, genome-wide", fontsize=8.6)
    c.set_title("Own labels vs labels pooled across related drugs", fontsize=10, loc="left")
    c.set_ylim(-0.7, len(t) + 0.25)
    c.legend(fontsize=8.2, loc="upper right")
    _axstyle(c)

    m = cf[cf.stat == "cmh"].groupby(["set", "top_k", "drug"]).auc.mean().groupby(["set", "top_k"]).mean()
    ks = [30, 300, 3000]
    xs = np.arange(3)
    for name, mark, mfc in (("19p", "o", N["ink2"]), ("genome", "o", N["surface"])):
        d.plot(xs, [m[(name, k)] for k in ks], color=N["ink2"], lw=1.6, marker=mark, ms=7.5, mfc=mfc,
               mec=N["ink2"], mew=1.2, label=f"closed-form score, {name}")
    for name, ls in (("LF v3, 19 loci", "-"), ("LF v3, per-drug", "--")):
        d.axhline(base[name], color=ACCENT, lw=1.3, ls=ls, zorder=1)
        d.text(2.32, base[name] + .0018, f"{name} {base[name]:.4f}", ha="right", va="bottom",
               fontsize=8, color=N["ink2"])
    d.set_xticks(xs, [f"top {k:,}" for k in ks])
    d.set_xlim(-.25, 2.35)
    d.set_ylim(0.77, 0.93)
    d.set_ylabel("macro CV AUC, 11 drugs", fontsize=8.6)
    d.set_title("Closed-form score vs LocusFusion v3", fontsize=10, loc="left")
    d.legend(fontsize=8.2, loc="lower left")
    _axstyle(d)
    d.grid(axis="x", visible=False)
    d.grid(axis="y", color=N["grid"], lw=0.7)
    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    return save(fig, "fig_design_probes")


ALL = (current_architecture, tokenization, overview, evidence, effect_field, clade_delta, retriever_reader, genome_pyramid,
       resistome, pattern_forest)

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import paper_style
    paper_style.apply()
    for fn in ALL:
        fn()
        plt.close("all")
    print("wrote to", FIGDIR)

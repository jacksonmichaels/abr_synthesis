"""GenomeFusion method figures, drawn only from real data.

    python scripts/genomefusion_method_extract.py      # writes method_trace.json
    python scripts/genomefusion_method_figures.py      # safe on a login node

    fig_gf_genome_to_tokens   the H37Rv genome -> one isolate's tokens
    fig_gf_tokens_to_model    those tokens -> packed tensors -> the model's shapes
    fig_gf_cohort_limits      what the limits look like over the whole cohort

Every number is read from results/figures/genomefusion/method_trace.json. The only
things typed by hand are the names of the modules and channels. These functions are
also what notebooks/genomefusion/genomefusion_architectures.ipynb calls.
"""
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
import paper_style as PS  # noqa: E402

PS.apply()
N = PS.NEUTRAL
SURFACE, INK, INK2, MUTED, AXIS, GRID = (N[k] for k in ("surface", "ink", "ink2", "muted", "axis", "grid"))
MOD = PS.MODALITY
LF = PS.MODEL["locusfusion"]
LF_FILL, LF_TINT = PS.ramp(LF, 4, light=0.95)[2:]
KIND = {"cds": MOD["dna"], "rna": PS.MODEL["cnn"], "intergenic": MOD["regulatory"], "masked": AXIS}
SEQ = LinearSegmentedColormap.from_list("copper_seq", [SURFACE, LF, "#5c2f14"])
GREY = LinearSegmentedColormap.from_list("grey_seq", [SURFACE, INK2])
FIG = PROJECT / "results" / "figures" / "genomefusion"
MINUS = "−"


def load():
    return json.loads((FIG / "method_trace.json").read_text())


def text(ax, x, y, s, size=5.4, color=INK, weight="normal", ha="left", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


def panel(fig, x, y, letter, title):
    fig.text(x, y, letter, fontsize=8.5, fontweight="bold", color=INK, va="baseline")
    fig.text(x + 0.018, y, title, fontsize=7.4, fontweight="bold", color=INK, va="baseline")


def blank(fig, rect, xlim=(0, 1), ylim=(0, 1)):
    ax = fig.add_axes(rect)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return ax


def signed(v, d=2):
    return f"{v:+.{d}f}".replace("-", MINUS)


def shape(*s):
    return "(" + ", ".join(f"{x:,}" if isinstance(x, int) else str(x) for x in s) + ")"


def token_label(t):
    if t["residue"] and t["ref_residue"] and t["consequence"] in ("missense", "nonsense"):
        return f"{t['locus']} {t['ref_residue']}{t['codon'] + 1}{'*' if t['residue'] == '*' else t['residue']}"
    return t["locus"]


# ==================================================================== genome -> tokens
def genome_to_tokens(T=None, save=True):
    T = T or load()
    G, E, Sm = T["genome"], T["example"], T["funnel_sample"]
    fig = plt.figure(figsize=(7.5, 9.9))

    # ---- a: the partition, by base, along the genome --------------------------------
    y = 0.977
    panel(fig, 0.005, y, "a", f"H37Rv partition: {G['length']:,} bases into {G['n_loci']:,} loci")
    ax = blank(fig, [0.06, 0.930, 0.925, 0.030], (0, G["length"]), (0, 1))
    x = 0
    order = [("cds", "protein-coding"), ("rna", "RNA gene"), ("intergenic", "intergenic"), ("masked", "masked")]
    for k, lab in order:
        w = G["bases"][k]
        ax.add_patch(Rectangle((x, 0), w, 1, fc=KIND[k], ec=SURFACE, lw=0.6))
        x += w
    ly = 0.918
    lx = 0.06
    for k, lab in order:
        n = f"{G['bases'][k]:,} bp, {100 * G['bases'][k] / G['length']:.1f}%"
        n += f", {G['loci'][k]:,} loci" if k != "masked" else ""
        fig.patches.append(Rectangle((lx, ly - 0.004), 0.009, 0.008, transform=fig.transFigure, fc=KIND[k], ec="none"))
        fig.text(lx + 0.012, ly, f"{lab}: {n}", fontsize=5.1, color=INK2, va="center")
        lx += 0.255 if k != "rna" else 0.205
        lx = lx - 0.05 if k == "masked" else lx
    mb = (np.arange(len(G["masked_bin"])) + 0.5) * G["bin"] / 1e6
    rows = [(f"masked fraction, per {G['bin'] // 1000} kb", np.asarray(G["masked_bin"]), GREY, None),
            (f"mean tokens per isolate, per {G['bin'] // 1000} kb", np.asarray(G["tokens_mean_bin"]), SEQ,
             LogNorm(vmin=0.02, vmax=max(G["tokens_mean_bin"]))),
            (f"{E['isolate']} tokens, per {G['bin'] // 1000} kb", np.asarray(G["tokens_example_bin"]), SEQ, None)]
    yy = 0.885
    for lab, v, cm, nrm in rows:
        fig.text(0.06, yy + 0.013, lab, fontsize=5.0, color=INK2, va="bottom")
        a = fig.add_axes([0.06, yy, 0.925, 0.012])
        kw = dict(norm=nrm) if nrm else dict(vmin=0, vmax=v.max())
        a.imshow(v[None, :], aspect="auto", cmap=cm, extent=(0, mb[-1] + G["bin"] / 2e6, 0, 1),
                 interpolation="nearest", **kw)
        a.set_yticks([])
        a.set_xticks([])
        for sp in a.spines.values():
            sp.set_linewidth(0.3)
            sp.set_color(AXIS)
        text(a, 1.0, 1.28, f"max {v.max():,.1f}" if v.max() % 1 else f"max {int(v.max()):,}", size=4.6,
             color=INK2, ha="right", transform=a.transAxes)
        yy -= 0.030
    a.set_xticks(np.arange(0, 4.5, 0.5))
    a.tick_params(axis="x", labelsize=4.8, length=1.5, width=0.4, pad=1)
    a.set_xlabel("genome position, Mb", fontsize=5.0, labelpad=0.5)

    # ---- b: one isolate through every filter ------------------------------------------
    yb = 0.786
    F = E["funnel"]
    panel(fig, 0.005, yb, "b", f"From VCF to tokens: {E['isolate']}, and {T['funnel_sample_n']} sampled isolates")
    steps = [("VCF variant records", F["records"], "records", None),
             ("events after QC", F["events"], "events", f"{F['qc_left_as_reference']:,} left as reference, "
              f"{F['symbolic_alt']:,} symbolic ALT; indels and runs expanded"),
             ("placed in a locus", F["placed"], "placed", f"{F['masked']:,} events in masked bases"),
             ("tokens", F["tokens"], "tokens", f"{F['synonymous']:,} synonymous changes dropped"),
             ("active loci", F["active_loci"], "active_loci", f"tokens grouped; {G['n_loci'] - F['active_loci']:,} "
              f"of {G['n_loci']:,} loci carry none")]
    ax = fig.add_axes([0.150, yb - 0.112, 0.235, 0.088])
    ax.set_xlim(0, F["records"] * 1.02)
    ax.set_ylim(len(steps) - 0.5, -0.5)
    ax.axis("off")
    tx = blank(fig, [0.385, yb - 0.112, 0.61, 0.088], (0, 1), (len(steps) - 0.5, -0.5))
    text(tx, 0.015, -0.92, E["isolate"], size=4.8, weight="bold")
    text(tx, 0.155, -0.92, f"median [5th, 95th], n = {T['funnel_sample_n']}", size=4.8, weight="bold")
    text(tx, 0.47, -0.92, "removed at this step", size=4.8, weight="bold")
    for i, (lab, n, key, why) in enumerate(steps):
        last = key in ("tokens", "active_loci")
        ax.barh(i, n, height=0.62, color=LF if last else LF_FILL)
        text(ax, -F["records"] * 0.02, i, lab, size=5.3, ha="right", weight="bold" if last else "normal")
        text(tx, 0.015, i, f"{n:,}", size=5.3, weight="bold" if last else "normal")
        s = Sm[key]
        text(tx, 0.155, i, f"{s['median']:,.0f} [{s['p5']:,.0f}, {s['p95']:,.0f}]", size=5.1, color=INK2)
        if why:
            text(tx, 0.47, i, why, size=5.0, color=INK2)

    # ---- c: two real windows of the partition -----------------------------------------
    yc = yb - 0.140
    panel(fig, 0.005, yc, "c", "Partition around two loci, with this isolate's tokens and the regulated-locus edges")

    def track(rect, W, title):
        ax = fig.add_axes(rect)
        lo, hi = W["lo"], W["hi"]
        ax.set_xlim(lo, hi)
        ax.set_ylim(-1.15, 2.05)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks(np.arange(np.ceil(lo / 500) * 500, hi + 1, 500))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.tick_params(axis="x", labelsize=4.6, length=1.5, width=0.4, pad=1)
        ax.set_xlabel(title, fontsize=5.0, labelpad=0.5)
        by_tag = {L["name"]: L for L in W["loci"]}
        for L in W["loci"]:
            s, e = max(L["start"], lo), min(L["end"], hi)
            ax.add_patch(Rectangle((s, -0.25), e - s, 0.5, fc=KIND[L["kind"]],
                                   alpha=0.28 if L["kind"] == "cds" and not L["is_19"] else 0.85,
                                   ec=INK if L["is_regulatory"] else "none", lw=0.7))
            mid = (s + e) / 2
            if e - s > (hi - lo) * 0.045:
                arrow = {1: "→", -1: "←", 0: ""}[L["strand"]]
                ttl = f"{L['name']} {arrow}" if L["kind"] != "intergenic" else f"{e - s} bp"
                if L["kind"] != "intergenic" or e - s > (hi - lo) * 0.03:
                    text(ax, mid, -0.58, ttl, size=4.8, ha="center", style="italic" if L["kind"] == "cds" else "normal",
                         color=INK)
            if L["edge_to"] and L["edge_to"] in by_tag and by_tag[L["edge_to"]]["name"] != L["name"]:
                t = by_tag[L["edge_to"]]
                x1 = (max(t["start"], lo) + min(t["end"], hi)) / 2
                ax.add_patch(FancyArrowPatch((mid, 0.27), (x1, 0.27), connectionstyle=f"arc3,rad={-0.28 if x1 > mid else 0.28}",
                                             arrowstyle="-|>", mutation_scale=4.5, lw=0.5, color=MUTED, zorder=2))
        return ax

    W1, W2 = T["windows"]["fabG1-inhA"], T["windows"]["katG"]
    a1 = track([0.02, yc - 0.082, 0.47, 0.070], W1, "genome position, bp")
    a2 = track([0.515, yc - 0.082, 0.47, 0.070], W2, "genome position, bp")
    ex = {t["kind"]: t for t in E["tokens"]}
    for a_, t, lab in ((a1, ex["intergenic"], f"token: fabG1 c.{MINUS}15C>T, position {ex['intergenic']['pos'] + 1:,}"),
                       (a2, ex["missense"], f"token: katG S315T, position {ex['missense']['pos'] + 1:,}")):
        a_.plot([t["pos"]] * 2, [-0.40, 0.42], color=LF, lw=1.3, solid_capstyle="butt", zorder=5)
        text(a_, t["pos"], 1.82, lab, size=5.0, color=LF, ha="left" if a_ is a1 else "center", weight="bold")
        a_.plot([t["pos"]] * 2, [0.42, 1.62], color=LF, lw=0.4, zorder=1)
    fig.legend(handles=[Patch(fc=KIND["cds"], ec="none", label="gene in the curated 19"),
                        Patch(fc=KIND["cds"], alpha=0.28, ec="none", label="other gene"),
                        Patch(fc=KIND["intergenic"], ec="none", label="intergenic"),
                        Patch(fc=SURFACE, ec=INK, lw=0.7, label="WHO regulatory annotation"),
                        Line2D([], [], color=MUTED, lw=0.6, marker=">", ms=2.5, label="regulated-locus edge")],
               loc="center", bbox_to_anchor=(0.5, yc - 0.1105), ncol=5, fontsize=5.0, frameon=False, columnspacing=1.1,
               handlelength=1.3, handletextpad=0.4)

    # ---- d: the 57 channels of real tokens --------------------------------------------
    yd = yc - 0.138
    panel(fig, 0.005, yd, "d", f"Token channels: {E['n_tokens']:,} × 57 for this isolate, one token of each kind shown")
    rows = [t for t in E["tokens"] if t["kind"] not in ("insertion", "overlap")]
    S = T["slots"]
    BL = [("base", "base", list("ACTG-N"), MOD["dna"]), ("residue", "residue", list("ACDEFGHIKLMNPQRSTVWY*-X"), MOD["protein"]),
          ("bio", "props", ["M", "p", "H"], MOD["biophysical"]), ("d_bio", "Δ props", ["M", "p", "H"], MOD["biophysical"]),
          ("consequence", "consequence", ["syn", "mis", "non", "fs", "ind", "sl", "stl", "unr"], INK2),
          ("truncation", "tr", ["t"], INK2), ("region", "region", ["c", "r", "i"], MOD["regulatory"]),
          ("upstream", "up", ["+", MINUS], MOD["regulatory"]), ("span", "sp", ["s"], INK2),
          ("phase", "phase", ["0", "1", "2"], INK2), ("flags", "flags", ["o", "h", "i"], INK2), ("is_null", "", ["∅"], INK2)]
    never = {T["channels"].index(c) for c in T["never_nonzero"]}
    h = 0.0178 * (len(rows) + 2.4)
    ax = blank(fig, [0.170, yd - 0.008 - h, 0.56, h], (0, 57), (len(rows), -2.4))
    col = {}
    for key, nm, alpha, c in BL:
        a, b = S[key]
        ax.plot([a + 0.12, b - 0.12], [-1.55, -1.55], color=c, lw=1.2, solid_capstyle="butt")
        if nm:
            text(ax, (a + b) / 2, -2.05, nm, size=4.4, ha="center", color=INK2)
        for k, s in enumerate(alpha):
            col[a + k] = c
            text(ax, a + k + 0.5, -0.80, s, size=3.1 if len(s) > 1 else 3.7, ha="center", color=MUTED, rotation=90 if len(s) > 1 else 0)
    vmax = max(abs(v) for t in rows for v in t["feat"][S["bio"][0]:S["d_bio"][1]]) or 1
    for r, t in enumerate(rows):
        f = t["feat"]
        for c in range(57):
            v = f[c]
            ax.add_patch(Rectangle((c + 0.06, r + 0.08), 0.88, 0.84, fc=GRID if c in never else SURFACE, ec=GRID, lw=0.25))
            if v:
                numeric = S["bio"][0] <= c < S["d_bio"][1] or c in (S["truncation"][0], S["span"][0], S["upstream"][0], S["upstream"][1] - 1)
                alpha = min(1, max(abs(v) / (vmax if S["bio"][0] <= c < S["d_bio"][1] else 3.0), 0.2)) if numeric else 1
                fc = MOD["uncalled"] if c == S["base"][0] + 5 else col[c]
                ax.add_patch(Rectangle((c + 0.06, r + 0.08), 0.88, 0.84, fc=fc, ec="none", alpha=alpha))
        text(ax, -0.5, r + 0.5, f"{t['kind']}: {token_label(t)}", size=4.8, ha="right")
        bits = [f"pos {t['pos'] + 1:,}", f"coord {t['coord']:,.0f}"]
        if t["span"] > 1:
            bits.append(f"span {t['span']}")
        if t["truncation"]:
            bits.append(f"trunc {t['truncation']:.3f}")
        if t["up_plus"] >= 0 and t["up_plus"] <= 200:
            bits.append(f"up+ {t['up_plus']} bp")
        text(ax, 57.5, r + 0.5, "  ".join(bits), size=4.5, color=INK2)
    leg_y = yd - 0.008 - h - 0.008
    fig.patches.append(Rectangle((0.170, leg_y - 0.003), 0.009, 0.006, transform=fig.transFigure, fc=GRID, ec=AXIS, lw=0.3))
    fig.text(0.182, leg_y, "channel never nonzero in 2,000,000 sampled tokens", fontsize=4.9, color=INK2, va="center")
    fig.patches.append(Rectangle((0.47, leg_y - 0.003), 0.009, 0.006, transform=fig.transFigure, fc=MOD["uncalled"], ec="none"))
    fig.text(0.482, leg_y, "uncalled base", fontsize=4.9, color=INK2, va="center")
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"fig_gf_genome_to_tokens.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    return fig


# ==================================================================== tokens -> model
def box(ax, x, y, w, h, title, lines=(), fc=SURFACE, ec=AXIS, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008", fc=fc, ec=ec, lw=lw))
    text(ax, x + 0.012, y + h - 0.016, title, size=5.5, weight="bold", va="top")
    yy = y + h - 0.048
    for ln in lines:
        text(ax, x + 0.012, yy, ln, size=4.9, color=INK2, va="top")
        yy -= 0.027


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, lw=0.8, color=MUTED, shrinkA=0, shrinkB=0))


def tokens_to_model(T=None, save=True):
    T = T or load()
    E, P, C = T["example"], T["params"], T["cohort"]
    Pg = P["groups"]
    n_tok, n_pair, lmax = E["n_tokens"], E["n_active"], E["lmax"]
    fig = plt.figure(figsize=(7.5, 8.2))

    # ---- a: packed tensors ------------------------------------------------------------
    y = 0.972
    panel(fig, 0.005, y, "a", f"Packed batch for {E['isolate']}: {n_tok:,} tokens in {n_pair:,} pairs; a pair is one isolate's tokens in one locus")
    tx = blank(fig, [0.01, y - 0.118, 0.47, 0.105], (0, 1), (7.5, -0.5))
    rows = [("tok_feat", shape(n_tok, 57), "channels of each token"),
            ("tok_coord", shape(n_tok), "base offset from the locus's lower coordinate"),
            ("tok_pair", shape(n_tok), "which pair each token belongs to"),
            ("pair_lid", shape(n_pair), "locus id of each pair"),
            ("pair_stats", shape(n_pair, 4), "[NULL] inputs"),
            ("pair_n", shape(n_pair), "tokens in each pair"),
            ("pair_slot", shape(n_pair), "slot in stage 2; slot 0 is the reference token")]
    for i, (a, b, c) in enumerate(rows):
        text(tx, 0.0, i, a, size=5.2, weight="bold")
        text(tx, 0.19, i, b, size=5.2)
        text(tx, 0.38, i, c, size=4.9, color=INK2)
    # bucket table, this isolate and the whole cohort
    tb = blank(fig, [0.50, y - 0.118, 0.49, 0.105], (0, 1), (7.5, -0.9))
    text(tb, 0.0, -0.6, "tokens in pair", size=4.8, weight="bold")
    text(tb, 0.21, -0.6, "pairs", size=4.8, weight="bold")
    text(tb, 0.34, -0.6, "stage 1 tensor", size=4.8, weight="bold")
    text(tb, 0.66, -0.6, "cohort pairs", size=4.8, weight="bold")
    lo, r = 0, 0
    cb = C["pair_buckets"]
    for hi in T["buckets"]:
        k = E["buckets"].get(str(hi), 0)
        if k or hi <= 16:
            lab = f"{hi}" if hi - lo == 1 else f"{lo + 1}–{hi}"
            text(tb, 0.0, r + 0.4, lab, size=5.1)
            text(tb, 0.21, r + 0.4, f"{k:,}", size=5.1)
            text(tb, 0.34, r + 0.4, shape(k, hi + 1, 128) if k else "—", size=5.1, color=INK if k else AXIS)
            text(tb, 0.66, r + 0.4, f"{cb[str(hi)]:,}  ({100 * cb[str(hi)] / C['pairs_total']:.2f}%)", size=5.1, color=INK2)
            r += 1
        lo = hi
    rest = sum(v for k_, v in cb.items() if k_ == ">512" or (k_.isdigit() and int(k_) > 16))
    text(tb, 0.0, r + 0.4, f"> 16, max {C['pair_tokens_max']:,}", size=5.1, color=INK2)
    text(tb, 0.21, r + 0.4, "0", size=5.1, color=AXIS)
    text(tb, 0.66, r + 0.4, f"{rest:,}  ({100 * rest / C['pairs_total']:.2f}%)", size=5.1, color=INK2)

    # ---- b: the model, with this isolate's shapes -------------------------------------
    y2 = y - 0.150
    panel(fig, 0.005, y2, "b", "GenomeFusion, whole-genome arm: this isolate's shapes and the parameters of each module")
    ax = blank(fig, [0, 0.235, 1, y2 - 0.245])
    # coordinates are axes fractions; the column widths are fixed
    W_ = 0.315
    X = [0.01, 0.342, 0.674]
    H1 = 0.222
    top = 0.975
    box(ax, X[0], top - H1, W_, H1, f"Variant token: {shape(n_tok, 128)}", [
        f"tok_proj 57→128: {Pg['tok_proj']:,}",
        f"+ pos_proj(sinusoid(coord)) 32→128: {Pg['pos_proj']:,}",
        "+ locus identity of its pair",
        "× (1 + scale) + shift, then LayerNorm:",
        f"    {Pg['tok_norm']:,}, shared with [NULL]"], fc=LF_TINT, ec=LF)
    box(ax, X[1], top - H1, W_, H1, f"[NULL] token: {shape(n_pair, 128)}", [
        f"null_emb {shape(128)}: {Pg['null_emb']}",
        f"+ null_proj 4→128: {Pg['null_proj']:,}",
        "+ locus identity of its pair",
        "× (1 + scale) + shift, then LayerNorm",
        "4 inputs: log1p(n), n / locus length,",
        "    net frame offset / 3, out-of-frame fraction"], fc=LF_TINT, ec=LF)
    box(ax, X[2], top - H1, W_, H1, f"Locus identity, one per pair: {shape(n_pair, 128)}", [
        f"locus_feat buffer {shape(T['genome']['n_loci'], P['feat_width'])}: 29 own + 29 of target",
        f"feat_proj 58→128→128: {Pg['feat_proj']:,}",
        f"+ 0.1 × locus_residual {shape(T['genome']['n_loci'], 128)}: {Pg['locus_residual']:,}",
        f"+ reg_gate × residual of the edge target: {Pg['reg_gate']}",
        f"film_proj 58→256, scale + shift: {Pg['film_proj']:,}",
        "residual dropout 0.3"], fc=LF_TINT, ec=LF)
    arrow(ax, (X[2], top - H1 / 2), (X[1] + W_, top - H1 / 2))
    # stage 1
    H2 = 0.205
    t2 = top - H1 - 0.050
    buckets = [(int(k), v) for k, v in E["buckets"].items() if v]
    txt = ",  ".join(shape(v, k + 1, 128) for k, v in buckets)
    box(ax, X[0], t2 - H2, 2 * W_ + 0.017, H2, f"Stage 1, within each pair: shared encoder, 2 layers, 4 heads, FF 256: {Pg['encoder']:,}", [
        f"this isolate's buckets: {txt}",
        "each row is [NULL] followed by the pair's tokens, padded to the bucket width and masked",
        f"output at [NULL]: one locus vector, {shape(n_pair, 128)}"], fc=LF_FILL, ec=LF)
    ax.add_patch(FancyBboxPatch((X[2], t2 - H2), W_, H2, boxstyle="round,pad=0,rounding_size=0.008", fc=SURFACE, ec=AXIS, lw=0.8))
    text(ax, X[2] + 0.012, t2 - 0.016, "[NULL] inputs, this isolate", size=5.5, weight="bold", va="top")
    text(ax, X[2] + 0.012, t2 - 0.052, "locus", size=4.5, weight="bold", va="top")
    for hx, hd in ((0.125, "n"), (0.158, "log1p"), (0.203, "n/len"), (0.25, "frame"), (0.29, "oof")):
        text(ax, X[2] + hx, t2 - 0.052, hd, size=4.5, weight="bold", ha="right", va="top")
    for i, e in enumerate(E["null_examples"]):
        yy = t2 - 0.078 - i * 0.024
        text(ax, X[2] + 0.012, yy, e["locus"][:15], size=4.5, va="top", style="italic")
        for hx, v in ((0.125, f"{e['n']}"), (0.158, f"{e['log1p']:.2f}"), (0.203, f"{e['density']:.4f}"),
                      (0.25, signed(e["frame"])), (0.29, f"{e['oof']:.3f}")):
            text(ax, X[2] + hx, yy, v, size=4.5, ha="right", va="top")
    arrow(ax, (X[0] + W_ / 2, t2 + 0.050), (X[0] + W_ / 2, t2))
    arrow(ax, (X[1] + W_ / 2, t2 + 0.050), (X[1] + W_ / 2, t2))
    # stage 2 and readout
    H3 = 0.190
    t3 = t2 - H2 - 0.050
    box(ax, X[0], t3 - H3, W_, H3, f"Stage 2, across loci: {shape(1, lmax, 128)}", [
        f"reference token {shape(128)}: {Pg['reference']} in slot 0",
        f"+ {n_pair:,} locus vectors, LayerNorm: {Pg['summ_norm']}",
        f"encoder, 2 layers, 4 heads, FF 256: {Pg['fusion']:,}",
        f"self-attention, {lmax:,} × {lmax:,} per head"], fc=LF_FILL, ec=LF)
    box(ax, X[1], t3 - H3, W_, H3, f"Readout: {shape(1, 1, 128)}", [
        f"drug_queries {shape(1, 128)}: {Pg['drug_queries']}",
        f"pool_attn, 4 heads, over {lmax:,} slots: {Pg['pool_attn']:,}",
        f"LayerNorm: {Pg['norm']}",
        f"fc1 128→256, ReLU: {Pg['fc1']:,}",
        f"fc_out 256→1: {Pg['fc_out']}"], fc=LF_TINT, ec=LF)
    box(ax, X[2], t3 - H3, W_, H3, f"Parameters: {P['total']:,}", [
        f"locus_residual: {Pg['locus_residual']:,}  ({100 * Pg['locus_residual'] / P['total']:.1f}%)",
        f"encoder + fusion: {Pg['encoder'] + Pg['fusion']:,}  ({100 * (Pg['encoder'] + Pg['fusion']) / P['total']:.1f}%)",
        f"pool_attn + fc1 + fc_out: {Pg['pool_attn'] + Pg['fc1'] + Pg['fc_out']:,}  "
        f"({100 * (Pg['pool_attn'] + Pg['fc1'] + Pg['fc_out']) / P['total']:.1f}%)",
        f"all others: {P['total'] - Pg['locus_residual'] - Pg['encoder'] - Pg['fusion'] - Pg['pool_attn'] - Pg['fc1'] - Pg['fc_out']:,}"], ec=AXIS)
    arrow(ax, (X[0] + W_ / 2, t2 - H2), (X[0] + W_ / 2, t3))
    arrow(ax, (X[0] + W_, t3 - H3 / 2), (X[1], t3 - H3 / 2))

    # ---- c: protocol and output ---------------------------------------------------------
    yc = 0.358
    panel(fig, 0.005, yc, "c", "Output and training protocol")
    tx = blank(fig, [0.01, yc - 0.135, 0.98, 0.125], (0, 1), (5.6, -0.5))
    pr = T["protocol"]
    rows = [("logit", shape(1, 1), "one model per drug; sigmoid gives P(resistant): y = 1 is resistant here, the opposite of LocusFusion"),
            ("loss", "masked weighted BCE", "per-drug class weights fitted on the development rows"),
            ("optimiser", f"lr {pr['lr']:.2e}", f"weight decay {pr['weight_decay']:g}, seed {pr['seed']}"),
            ("batch", f"{pr['batch_size']} isolates", f"at most {pr['locus_budget']:,} isolate × widest-isolate slots: {C['batch']['isolates_below_requested']:,} of "
             f"{C['n']:,} isolates fall in batches below {pr['batch_size']}, minimum {C['batch']['min_batch']}"),
            ("stopping", f"max {pr['epochs']} epochs", f"patience {pr['patience']}, min_epochs {pr['min_epochs']}, on validation {pr['monitor'].upper()}; checkpoint is the best epoch from epoch 1"),
            ("reported", "5 lineage folds", "mean of fold AUCs per drug, macro over 11 drugs; 55 paired folds for significance")]
    for i, (a, b, c) in enumerate(rows):
        text(tx, 0.0, i, a, size=5.2, weight="bold")
        text(tx, 0.09, i, b, size=5.2)
        text(tx, 0.255, i, c, size=4.9, color=INK2)
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"fig_gf_tokens_to_model.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    return fig


# ==================================================================== cohort limits
def cohort_limits(T=None, save=True):
    T = T or load()
    C = T["cohort"]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.2), layout="constrained")
    (a, b), (c, d) = axes

    def tidy(ax):
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=5.5, length=2, width=0.5)

    for ax, key, hist, ttl, xl in ((a, "tokens", C["hist_tokens"], f"Tokens per isolate, {C['n']:,} isolates", "tokens"),
                                   (b, "active", C["hist_active"], f"Active loci per isolate, of {T['genome']['n_loci']:,}", "active loci")):
        h, e = hist
        ax.bar(e[:-1], h, width=np.diff(e) * 0.88, align="edge", color=LF_FILL)
        s = C[key]
        tidy(ax)
        ax.axvline(s["median"], color=LF, lw=1.0)
        ax.set_title(ttl, fontsize=6.6, loc="left", fontweight="bold")
        ax.set_xlabel(xl, fontsize=5.8)
        ax.set_ylabel("isolates", fontsize=5.8)
        ax.legend(handles=[Line2D([], [], color=LF, lw=1, label=f"median {s['median']:,.0f}"),
                           Line2D([], [], ls="", label=f"5th–95th {s['p5']:,.0f}–{s['p95']:,.0f}"),
                           Line2D([], [], ls="", label=f"max {s['max']:,}, min {s['min']:,}")],
                  fontsize=5.4, frameon=False, loc="upper right", handlelength=1.2)
    cut = C["batch"]["budget"] // C["batch"]["requested"] - 1
    b.axvline(cut, color=MUTED, lw=0.9, ls="--")
    h_, l_ = b.get_legend_handles_labels()
    b.get_legend().remove()
    s = C["active"]
    b.legend(handles=[Line2D([], [], color=LF, lw=1, label=f"median {s['median']:,.0f}"),
                      Line2D([], [], ls="", label=f"5th–95th {s['p5']:,.0f}–{s['p95']:,.0f}"),
                      Line2D([], [], ls="", label=f"max {s['max']:,}, min {s['min']:,}"),
                      Line2D([], [], color=MUTED, lw=0.9, ls="--", label=f"{cut:,}: batch budget reduces 32")],
             fontsize=5.4, frameon=False, loc="upper right", handlelength=1.6)

    # tokens by what they are
    tidy(c)
    kinds = sorted(((k, v) for k, v in C["consequence"].items() if v), key=lambda kv: kv[1])
    labels = [k.replace("_", " ") for k, _ in kinds]
    vals = [v for _, v in kinds]
    y = np.arange(len(vals))
    c.barh(y, vals, color=LF_FILL, height=0.68)
    for i, v in enumerate(vals):
        text(c, v * 1.02, i, f"{v:,}  {100 * v / C['tokens_total']:.1f}%", size=5.2)
    c.set_yticks(y, labels, fontsize=5.5)
    c.set_xlim(0, max(vals) * 1.42)
    c.set_title(f"Tokens by consequence channel, {C['tokens_total']:,} tokens", fontsize=6.6, loc="left", fontweight="bold")
    c.set_xlabel("tokens", fontsize=5.8)
    c.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M"))

    # busiest loci
    tidy(d)
    B = C["busiest_loci"][::-1]
    y = np.arange(len(B))
    d.barh(y, [x["tokens"] / C["n"] for x in B], color=LF_FILL, height=0.68)
    for i, x in enumerate(B):
        text(d, x["tokens"] / C["n"] * 1.02, i, f"{x['tokens'] / C['n']:.1f}  ({100 * x['isolates'] / C['n']:.0f}% of isolates)", size=5.2)
    d.set_yticks(y, [x["locus"] for x in B], fontsize=5.5)
    d.set_xlim(0, B[-1]["tokens"] / C["n"] * 1.6)
    d.set_title("Loci holding the most tokens", fontsize=6.6, loc="left", fontweight="bold")
    d.set_xlabel("mean tokens per isolate", fontsize=5.8)
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"fig_gf_cohort_limits.{ext}", dpi=300)
    return fig


# ==================================================================== regulatory edges
EDGE_COL = {"who": MOD["regulatory"], "plus": MUTED, "minus": MUTED}
BRANCH = {"who": "WHO name", "plus": "next + gene", "minus": "next − gene", "none": "no edge"}


def regulatory_edges(T=None, R=None, save=True):
    """Where regulatory relationships come from, and how each edge is decided."""
    T = T or load()
    R = R or json.loads((FIG / "regulatory_trace.json").read_text())
    fig = plt.figure(figsize=(9.0, 12.2))
    tok = {t["kind"]: t for t in T["example"]["tokens"]}

    # ---- a: the sources ------------------------------------------------------------------
    y = 0.984
    panel(fig, 0.004, y, "a", "Sources: fixed before training, the same for every isolate")
    tx = blank(fig, [0.012, y - 0.067, 0.98, 0.060], (0, 1), (5.5, -0.5))
    hdr = [(0.0, "input"), (0.20, "file or field"), (0.47, "values used")]
    for x, h in hdr:
        text(tx, x, -0.2, h, size=6.4, weight="bold")
    i_, k_ = R["source_row"]["inhA"], R["source_row"]["katG"]
    rows = [("WHO 2023 catalogue, Table 22", "promoter_windows_atg.json", f"{R['n_regions']} regions, offsets from the start codon; core span only, not the {R['pad']} bp pad"),
            ("  row filed under inhA", f"core {i_['core_start']:,}–{i_['core_end']:,}", f"strand {i_['sense']}, tss {i_['tss']:,}, atg {i_['atg']:,}, segments {i_['segments']}"),
            ("  row filed under katG", f"core {k_['core_start']:,}–{k_['core_end']:,}", f"strand {k_['sense']}, tss {k_['tss']:,}, atg {k_['atg']:,}, segments {k_['segments']}"),
            ("H37Rv GenBank, NC_000962.3", "gene name, span, strand", f"next start codon per strand; {R['named_cds']:,} CDS have a gene name, {R['unnamed_cds']:,} only an Rv tag"),
            ("VCF position of each token", "up_plus, up_minus per base", f"c.{MINUS}15: {R['upstream']['up_plus']} bp to next + start, {R['upstream']['up_minus']} bp to next − start")]
    for r, (a_, b_, c_) in enumerate(rows):
        text(tx, 0.0, r + 0.8, a_, size=6.1)
        text(tx, 0.20, r + 0.8, b_, size=6.0, color=INK2)
        text(tx, 0.47, r + 0.8, c_, size=6.0, color=INK2)

    # ---- b, c: real windows, one row per locus ---------------------------------------------
    def window(y0, letter, key, title, token, token_lab, row_h=0.0148):
        W = R["windows"][key]
        panel(fig, 0.004, y0, letter, title)
        ax = fig.add_axes([0.012, y0 - 0.113, 0.976, 0.103])
        lo, hi = W["lo"], W["hi"]
        ax.set_xlim(lo, hi)
        ax.set_ylim(-1.5, 4.25)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks(np.arange(np.ceil(lo / 250) * 250, hi + 1, 250))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.tick_params(axis="x", labelsize=5.8, length=2, width=0.5, pad=1)
        by = {L["locus"]: L for L in W["loci"]}
        seen = {}
        for r_ in W["regions"]:
            seen.setdefault((r_["start"], r_["end"]), []).append(r_["gene"])
        for (s_, e_), genes in seen.items():
            ax.add_patch(Rectangle((s_, -1.35), e_ - s_, 0.34, fc=MOD["regulatory"], alpha=0.55, ec="none"))
            text(ax, (s_ + e_) / 2, -1.18, f"WHO region: {' and '.join(genes)} rows, {e_ - s_:,} bp", size=5.8, ha="center", color=SURFACE, weight="bold")
        for L in W["loci"]:
            a, b = max(L["start"], lo), min(L["end"], hi)
            ax.add_patch(Rectangle((a, -0.5), b - a, 0.8, fc=KIND[L["kind"]],
                                   alpha=0.85 if L["kind"] != "cds" or L["locus"] in ("fabG1", "inhA", "katG") else 0.28,
                                   ec=INK if L["is_regulatory"] else "none", lw=0.8, zorder=2))
            if L["kind"] == "intergenic" and b - a > (hi - lo) * 0.03:
                text(ax, (a + b) / 2, -0.76, L["locus"], size=5.4, ha="center", color=INK, zorder=3,
                     bbox=dict(fc=SURFACE, ec="none", pad=0.4))
            elif b - a > (hi - lo) * 0.06:
                arrow_ = {1: " →", -1: " ←", 0: ""}[L["strand"]]
                text(ax, (a + b) / 2, -0.1, L["locus"] + arrow_, size=5.9, ha="center", style="italic" if L["kind"] != "intergenic" else "normal",
                     color=SURFACE if L["kind"] != "cds" or L["locus"] in ("fabG1", "inhA", "katG") else INK, zorder=3)
            t = by.get(L["edge_to"])
            if t is not None:
                m1 = (a + b) / 2
                m2 = (max(t["start"], lo) + min(t["end"], hi)) / 2
                rad = -0.32 if m2 > m1 else 0.32
                ax.add_patch(FancyArrowPatch((m1, 0.34), (m2, 0.34), connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                             mutation_scale=6.5, lw=1.0 if L["branch"] == "who" else 0.7,
                                             color=EDGE_COL[L["branch"]], zorder=1))
        ax.plot([token["pos"]] * 2, [-0.62, 3.75], color=LF, lw=0.9, zorder=4)
        text(ax, token["pos"], 3.98, token_lab, size=6.0, color=LF, weight="bold", ha="center")
        # the decision for every locus
        tb = blank(fig, [0.012, y0 - 0.113 - 0.018 - row_h * (len(W["loci"]) + 1), 0.976, row_h * (len(W["loci"]) + 1)],
                   (0, 1), (len(W["loci"]) + 0.5, -0.5))
        cols = [(0.0, "locus"), (0.135, "span"), (0.265, "WHO bases"), (0.335, "regulates (sorted)"), (0.485, "downstream +, −"), (0.645, "branch taken"), (0.755, "edge to"), (0.845, "distance")]
        for x, h in cols:
            text(tb, x, 0, h, size=5.9, weight="bold")
        for r, L in enumerate(W["loci"]):
            vals = [L["locus"], f"{L['start'] + 1:,}–{L['end']:,}", f"{L['reg_bases']:,}" if L["reg_bases"] else "—",
                    ", ".join(L["regulates"]) or "—", ", ".join(L["downstream"]), BRANCH[L["branch"]], L["edge_to"] or "—",
                    f"{L['edge_dist']:,} bp" if L["edge_dist"] is not None else "—"]
            for (x, _), v in zip(cols, vals):
                text(tb, x, r + 1, v, size=5.7, color=MOD["regulatory"] if (x == 0.645 and L["branch"] == "who") else INK,
                     weight="bold" if x == 0.645 and L["branch"] == "who" else "normal")
        return y0 - 0.113 - 0.018 - row_h * (len(W["loci"]) + 1)

    yb = y - 0.082
    end = window(yb, "b", "fabG1-inhA", "Decisions around fabG1-inhA, with this isolate's c-15t token",
                 tok["intergenic"], f"token: fabG1 c.{MINUS}15C>T")
    end = window(end - 0.020, "c", "katG", "Decisions around katG, with this isolate's S315T token",
                 tok["missense"], "token: katG S315T")
    y_leg = end - 0.006
    fig.legend(handles=[Patch(fc=MOD["regulatory"], alpha=0.55, ec="none", label="WHO region"),
                        Patch(fc=KIND["intergenic"], ec="none", label="intergenic locus"),
                        Patch(fc=KIND["cds"], ec="none", label="gene in the curated 19"),
                        Patch(fc=KIND["cds"], alpha=0.28, ec="none", label="other gene"),
                        Patch(fc=SURFACE, ec=INK, lw=0.8, label="tagged by a WHO region"),
                        Line2D([], [], color=MOD["regulatory"], lw=1.0, marker=">", ms=3, label="edge from a WHO name"),
                        Line2D([], [], color=MUTED, lw=0.7, marker=">", ms=3, label="edge from the next-gene fallback")],
               loc="center", bbox_to_anchor=(0.5, y_leg), ncol=7, fontsize=5.8, frameon=False, columnspacing=1.0, handlelength=1.4, handletextpad=0.4)

    # ---- d: the rule ------------------------------------------------------------------------
    yd = y_leg - 0.024
    panel(fig, 0.004, yd, "d", f"regulated_index: the rule, applied once per locus in order, with counts over {R['n_loci']:,} loci")
    tx = blank(fig, [0.012, yd - 0.098, 0.60, 0.090], (0, 1), (5.5, -0.5))
    B = R["by_kind"]
    C_ = R["counts"]
    for x, h in ((0.0, "step"), (0.585, "loci"), (0.685, "CDS"), (0.785, "RNA"), (0.885, "intergenic")):
        text(tx, x, 0, h, size=5.9, weight="bold")
    steps = [("who", "1. first of sorted regulates with a named locus, not itself"),
             ("plus", "2. next + strand start codon, if that gene is named"),
             ("minus", "3. next − strand start codon, if that gene is named"),
             ("none", "4. otherwise no edge")]
    for r, (k, lab) in enumerate(steps):
        text(tx, 0.0, r + 1, lab, size=5.9, color=MOD["regulatory"] if k == "who" else INK, weight="bold" if k == "who" else "normal")
        for x, v in ((0.585, C_[k]), (0.685, B["cds"][k]), (0.785, B["rna"][k]), (0.885, B["intergenic"][k])):
            text(tx, x, r + 1, f"{v:,}", size=5.9, color=MOD["regulatory"] if k == "who" else INK, weight="bold" if k == "who" and x == 0.585 else "normal")
    tag = R["tagged_by_branch"]
    text(tx, 0.0, 5.0, f"Of {R['n_regulatory']} WHO-tagged loci: {tag['who']} take step 1, {tag['plus'] + tag['minus']} fall through to steps 2–3, {tag['none']} get no edge", size=5.7, color=INK2)
    # the distances
    ax = fig.add_axes([0.665, yd - 0.092, 0.32, 0.080])
    D = R["intergenic_downstream"]
    h, e = D["hist"]
    ax.bar(e[:-1], h, width=np.diff(e) * 0.88, align="edge", color=LF_FILL)
    ax.set_xticks([0, 1, 2, 3, 4], ["≤1", "10", "100", "1,000", "10,000"], fontsize=5.4)
    ax.tick_params(axis="y", labelsize=5.2, length=1.5, width=0.4, pad=1)
    ax.tick_params(axis="x", length=1.5, width=0.4, pad=1)
    ax.set_xlabel("bp from intergenic locus to its target's start codon", fontsize=5.4, labelpad=1)
    ax.set_ylabel("intergenic loci", fontsize=5.4, labelpad=1)
    ax.set_title(f"Fallback edges: n = {D['n']:,}, {D['over_500']} over 500 bp, max {D['max']:,} bp", fontsize=5.8, loc="left", color=INK2, pad=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---- e: what the model does with an edge --------------------------------------------------
    ye = yd - 0.122
    panel(fig, 0.004, ye, "e", "Use in the model: fixed lookups at build time, read for each active locus in every forward pass")
    F_ = R["features"]
    own = F_["own_width"]
    names = F_["names"]
    mat = np.zeros((2, len(names)))
    for r, key in enumerate(("c15", "fabG1")):
        for it in F_[key]:
            mat[r, names.index(it["name"])] = it["value"]
    ax = fig.add_axes([0.125, ye - 0.058, 0.862, 0.040])
    ax.imshow(mat, aspect="auto", cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
    ax.axvline(own - 0.5, color=INK, lw=0.9)
    ax.set_yticks([0, 1], ["Rv1482c-Rv1483", "fabG1"], fontsize=5.6)
    short = [n.replace("reg:", "").replace("cat:", "").replace("kind:", "").replace("ess:", "ess ")[:24] for n in names]
    ax.set_xticks(range(len(names)), short, rotation=90, fontsize=3.9)
    ax.tick_params(length=0, pad=1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
        sp.set_color(AXIS)
    ax.set_title(f"locus_feat, 58 values: own {own} | {own} copied from the edge target", fontsize=5.8, color=INK2, loc="left", pad=2)
    tx = blank(fig, [0.012, ye - 0.200, 0.57, 0.074], (0, 1), (5.5, -0.5))
    uses = [("when", "where", "what"),
            ("partition built", "annotate_regulatory", "regions tag loci; regulates, reg_bases, downstream stored"),
            ("model built", "regulated_index, feature_matrix", "one target per locus; target's features copied into row"),
            ("model built", "remap to the locus set", "target outside the set: edge −1, copied features kept"),
            ("each forward pass", "locus_vectors", "feat_proj(row) + 0.1 × residual + reg_gate × residual[target]"),
            ("each isolate", "tokens", f"up channels: exp(−d/50): c.{MINUS}15 + {R['upstream']['ch_plus']:.3f}, − {R['upstream']['ch_minus']:.3f}")]
    for r, (a_, b_, c_) in enumerate(uses):
        w = "bold" if r == 0 else "normal"
        text(tx, 0.0, r, a_, size=5.8, weight=w)
        text(tx, 0.19, r, b_, size=5.6, weight=w, color=INK if r == 0 else INK2)
        text(tx, 0.495, r, c_, size=5.6, weight=w, color=INK if r == 0 else INK2)
    tx = blank(fig, [0.655, ye - 0.200, 0.335, 0.074], (0, 1), (5.5, -0.5))
    for x, h in ((0.0, "locus set"), (0.30, "loci"), (0.50, "edges kept"), (0.78, "from WHO")):
        text(tx, x, 0, h, size=5.8, weight="bold")
    for r, s_ in enumerate(("19", "19p", "19r", "minus19p", "genome")):
        v = R["sets"][s_]
        text(tx, 0.0, r + 1, s_, size=5.7)
        text(tx, 0.30, r + 1, f"{v['loci']:,}", size=5.7)
        text(tx, 0.50, r + 1, f"{v['edges_kept']:,}", size=5.7)
        text(tx, 0.78, r + 1, f"{v['who_edges']:,}", size=5.7, color=MOD["regulatory"], weight="bold")
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"fig_gf_regulatory_edges.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.05)
    return fig


# ==================================================================== forward pass
def forward_pass(T=None, F=None, path=None, save=True):
    """Trained values of one forward pass, from forward_trace.json."""
    T = T or load()
    F = F or json.loads((Path(path) if path else FIG / "forward_trace.json").read_text())
    S_ = F["showcase"]
    DIV = LinearSegmentedColormap.from_list("div", [PS.MODEL["cnn"], SURFACE, LF])
    fig = plt.figure(figsize=(7.5, 8.9))
    ex = {t["pos"]: t for t in T["example"]["tokens"]}

    def strip(rect, v, vmax=3.0):
        ax = fig.add_axes(rect)
        ax.imshow(np.asarray(v)[None, :], aspect="auto", cmap=DIV, vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)
            sp.set_color(AXIS)

    # ---- a: embedding terms -------------------------------------------------------------
    top = 0.975
    panel(fig, 0.005, top, "a", f"Token embedding, {F['isolate']}: katG S315T and the katG [NULL] token; "
          f"fold {F['fold']}, epoch {F['best_epoch']}")
    left = [("feat (57)", S_["feat"]), (f"sinusoid(coord = {S_['coord']:,.0f}) (32)", S_["sinusoid"]),
            ("tok_proj 57→128", S_["tok_proj"]), ("pos_proj 32→128", S_["pos_proj"]),
            ("locus identity", S_["locus"]), ("sum of the three", S_["sum"]),
            ("× (1 + scale) + shift", S_["film"]), ("LayerNorm: stage 1 input", S_["layernorm"])]
    stats = S_["stats"]
    right = [("feat_proj(locus_feat[katG]) 58→128", S_["feat_proj"]), ("0.1 × locus_residual[katG]", S_["residual"]),
             (f"reg_gate ({signed(S_['reg_gate'], 4)}) × residual of edge target", S_["edge"]),
             ("locus identity", S_["locus"]), ("null_emb", S_["null_emb"]),
             ("null_proj 4→128, input (" + ", ".join(f"{v:.3f}" for v in stats) + ")", S_["null_proj"]),
             ("× (1 + scale) + shift", S_["null_film"]), ("LayerNorm: stage 1 input", S_["null_layernorm"])]
    dy = 0.0262
    for x0, rows in ((0.005, left), (0.505, right)):
        for r, (lab, v) in enumerate(rows):
            y = top - 0.020 - r * dy
            fig.text(x0, y, lab, fontsize=4.9, color=INK2, va="top")
            strip([x0, y - dy + 0.0060, 0.41, 0.0115], v)
            fig.text(x0 + 0.415, y - dy + 0.0118, f"‖·‖ {np.linalg.norm(v):.2f}", fontsize=4.6, color=INK2, va="center")
    ya = top - 0.020 - len(left) * dy - 0.002
    cax = fig.add_axes([0.005, ya - 0.006, 0.14, 0.006])
    cax.imshow(np.linspace(-1, 1, 200)[None, :], aspect="auto", cmap=DIV, extent=(-3, 3, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([-3, 0, 3])
    cax.tick_params(axis="x", labelsize=4.6, length=1.5, width=0.4, pad=1)
    for sp in cax.spines.values():
        sp.set_linewidth(0.3)
    fig.text(0.155, ya - 0.003, "value of each channel", fontsize=4.8, color=INK2, va="center")

    def attn(ax, A, labels, title, vmax=None, annotate=True, size=4.5, labsize=4.5):
        A = np.asarray(A)
        vm = vmax or A.max()
        ax.imshow(A, cmap=SEQ, vmin=0, vmax=vm, interpolation="nearest", aspect="auto")
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=labsize)
        ax.set_yticks(range(len(labels)), labels, fontsize=labsize)
        ax.tick_params(length=0, pad=1)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=5.1, color=INK2, loc="left", pad=2)
        if annotate:
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    ax.text(j, i, f"{A[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=size,
                            color=SURFACE if A[i, j] > vm * 0.55 else INK)

    # ---- b: stage 1 ----------------------------------------------------------------------
    yb = ya - 0.040
    panel(fig, 0.005, yb, "b", "Stage 1: attention weights within a locus, averaged over 4 heads; rows attend to columns")
    pos_label = {}
    for t in T["example"]["tokens"]:
        pos_label[t["pos"]] = t
    xpos = 0.058
    for loc, w in (("katG", 0.062), ("Rv1482c-Rv1483", 0.062), ("ctpI", 0.062), ("Rv0387c", 0.165)):
        tr = F["traced"][loc]
        labs = ["[NULL]"] + [f"{p + 1:,}" for p in tr["positions"]]
        ann = len(labs) <= 6
        for li, a_ in enumerate(tr["attention"]):
            if loc == "Rv0387c" and li == 0:
                continue
            ax = fig.add_axes([xpos, yb - 0.117, w, 0.097])
            attn(ax, a_, labs, (f"{loc}, layer {li + 1}" if li == 0 else "layer 2") if loc != "Rv0387c" else f"{loc}, layer 2", vmax=1.0, annotate=ann or loc == "Rv0387c", size=3.3 if loc == "Rv0387c" else 4.5,
                 labsize=3.8)
            if li and loc != "Rv0387c":
                ax.set_yticks([])
            xpos += w + (0.006 if li == 0 and loc != "Rv0387c" else 0.052)

    # ---- c: stage 2 ----------------------------------------------------------------------
    yc = yb - 0.170
    panel(fig, 0.005, yc, "c", f"Stage 2: attention over {F['lmax']:,} slots; the traced slots and the most attended, and mean incoming weight")
    for li, st in enumerate(F["stage2"]):
        ax = fig.add_axes([0.075 + li * 0.300, yc - 0.200, 0.275, 0.180])
        attn(ax, st["block"], st["block_slots"], f"layer {li + 1}, maximum {st['max']:.3f}", annotate=False, labsize=4.0)
        if li:
            ax.set_yticks([])
    ax = fig.add_axes([0.745, yc - 0.200, 0.15, 0.180])
    rows = F["stage2"][-1]["incoming_mean"][:12][::-1]
    ax.barh(range(len(rows)), [r["weight"] for r in rows], color=LF_FILL, height=0.7)
    ax.set_yticks(range(len(rows)), [r["slot"] for r in rows], fontsize=4.2)
    ax.tick_params(axis="x", labelsize=4.4, length=1.5, width=0.4, pad=1)
    ax.tick_params(axis="y", length=0, pad=1)
    ax.set_title("mean incoming, layer 2", fontsize=5.1, color=INK2, loc="left", pad=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---- d: readout -----------------------------------------------------------------------
    yd = yc - 0.265
    panel(fig, 0.005, yd, "d", f"Readout: query attention over {F['lmax']:,} slots, and each step to the output")
    Q = F["query"]
    ax = fig.add_axes([0.07, yd - 0.135, 0.44, 0.115])
    rows = Q["top"][:20]
    ax.bar(range(len(rows)), [r["weight"] for r in rows], color=LF, width=0.72)
    ax.axhline(Q["uniform"], color=MUTED, lw=0.8, ls="--")
    ax.set_ylim(0, rows[0]["weight"] * 1.22)
    text(ax, len(rows) - 0.4, rows[0]["weight"] * 1.13, f"dashed line: uniform 1/{F['lmax']:,} = {Q['uniform']:.4f}", size=4.8, color=INK2, ha="right")
    ax.set_xticks(range(len(rows)), [r["slot"] for r in rows], rotation=90, fontsize=4.1)
    ax.set_ylabel("attention weight, top 20 slots", fontsize=4.9, labelpad=1)
    ax.tick_params(axis="y", labelsize=4.6, length=1.5, width=0.4, pad=1)
    ax.tick_params(axis="x", length=0, pad=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    tx = blank(fig, [0.54, yd - 0.150, 0.455, 0.14], (0, 1), (8.5, -0.5))
    steps = [("drug_queries", shape(1, 1, 128), f"‖·‖ {F['query_vector_norm']:.2f}"),
             ("pool_attn output", shape(1, 1, 128), f"‖·‖ {F['pooled_norm']:.2f}"),
             ("LayerNorm", shape(1, 1, 128), f"‖·‖ {F['layernorm_norm']:.2f}"),
             ("fc1, ReLU: 128→256", shape(1, 1, 256), f"{F['h_active']} of 256 units > 0"),
             ("fc_out: 256→1", shape(1, 1), f"logit {signed(F['logit'], 4)}"),
             ("sigmoid(logit)", "", f"P(resistant) = {F['P_R']:.4f}"),
             ("query on reference token", "", f"{Q['reference']:.4f}"),
             ("rescored against training", "", f"{F['checks']['rescored']} TEST isolates, max diff {F['checks']['max_diff']:.1e}"),
             (f"TEST, n = {F['test_n']:,} ({F['test_R']:,} R)", "", f"AUC {F['test_auc']:.4f}; 5-fold CV {F['cv_auc']:.4f}")]
    for r, (a_, b_, c_) in enumerate(steps):
        last = a_.startswith("sigmoid")
        text(tx, 0.0, r, a_, size=5.2, weight="bold" if last else "normal")
        text(tx, 0.43, r, b_, size=4.9, color=INK2)
        text(tx, 0.585, r, c_, size=5.2, color=LF if last else INK, weight="bold" if last else "normal")
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"fig_gf_forward_pass.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    return fig


if __name__ == "__main__":
    T = load()
    genome_to_tokens(T)
    tokens_to_model(T)
    cohort_limits(T)
    regulatory_edges(T)
    if (FIG / "forward_trace.json").exists():
        forward_pass(T)
    print(f"wrote fig_gf_genome_to_tokens, fig_gf_tokens_to_model, fig_gf_cohort_limits to {FIG}")

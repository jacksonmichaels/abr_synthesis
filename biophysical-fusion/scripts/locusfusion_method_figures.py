"""Method figures for LocusFusion v3, drawn from one real forward pass.

    python scripts/locusfusion_forward_trace.py     # batch: sbatch scripts/sbatch/locusfusion_forward_trace.sh
    python scripts/locusfusion_method_figures.py    # safe on a login node

Reads results/figures/manuscript/forward_pass_trace.json and writes
    fig_lf_tokens.{pdf,png}         19 loci -> the (1, 19, 17, 45) token tensor
    fig_lf_forward_pass.{pdf,png}   that tensor -> the logit

Every shape and value is read from the trace, which holds the model's own tensors
for one held-out isolate on the lineage-split, 19-locus, all-modalities ISONIAZID
checkpoint. Nothing is typed in by hand except the names of the modules.
"""
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
import paper_style as PS  # noqa: E402

PS.apply()
N = PS.NEUTRAL
SURFACE, INK, INK2, MUTED, AXIS, GRID = (N[k] for k in ("surface", "ink", "ink2", "muted", "axis", "grid"))
M = PS.MODALITY
LF = PS.MODEL["locusfusion"]
LF_TINT = PS.ramp(LF, 4, light=0.95)[3]
REG_T, DNA_T = PS.ramp(M["regulatory"], 4, 0.88)[3], PS.ramp(M["dna"], 4, 0.88)[3]
SEQ = LinearSegmentedColormap.from_list("copper_seq", [SURFACE, LF, "#5c2f14"])
DIV = LinearSegmentedColormap.from_list("div", [PS.MODEL["cnn"], SURFACE, LF])
MAN = PROJECT / "results" / "figures" / "manuscript"
T = json.loads((MAN / "forward_pass_trace.json").read_text())
LOCI = T["loci"]
ORDER = T["loci_order"]
LAY = T["layout"]
C = T["cohort"]
SL = LAY["slots"]
MINUS = "−"


def signed(v, d=2):
    return f"{v:+.{d}f}".replace("-", MINUS)


def shape(s):
    return "(" + ", ".join(str(x) for x in s) + ")"


def text(ax, x, y, s, size=5.4, color=INK, weight="normal", ha="left", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, **kw)


def panel(fig, x, y, letter, title):
    fig.text(x, y, letter, fontsize=8.5, fontweight="bold", color=INK, va="baseline")
    fig.text(x + 0.018, y, title, fontsize=7.4, fontweight="bold", color=INK, va="baseline")


def token_name(g, t):
    b = int(round(t["coord_codons"] * 3))
    if t["ref_aa"] and t["alt_aa"]:
        return f"{g} {t['ref_aa']}{t['codon'] + 1}{t['alt_aa']}"
    alt = {"-": "del", "N": "N"}.get(t["alt"], t["alt"])
    if b < 0:
        return f"{g} c.{MINUS}{-b}{t['ref']}>{alt}"
    prefix = "n" if not next(L for L in LOCI if L["locus"] == g)["has_cds"] else "c"
    return f"{g} {prefix}.{b + 1}{t['ref']}>{alt}"


def token_colour(t):
    if t["alt"] == "N":
        return M["uncalled"]
    return M["regulatory"] if t["region"] == "promoter" else M["dna"]


TOKENS = [(L["locus"], t) for L in LOCI for t in L["tokens"]]
NUMBER = {(g, t["slot"]): i + 1 for i, (g, t) in enumerate(TOKENS)}


# =============================================================== tokens figure
fig = plt.figure(figsize=(7.5, 7.1))
top, mid = 0.965, 0.455

# ---- a: the 19 loci at true scale, their block shapes, slots and [NULL] inputs ---
panel(fig, 0.005, top, "a", f"{T['isolate']} on the 19 loci: alignment blocks and token slots")
n = len(LOCI)
ax = fig.add_axes([0.065, mid + 0.035, 0.30, top - mid - 0.095])
lo = min(L["coord_min"] for L in LOCI) * 3
hi = max(L["coord_max"] for L in LOCI) * 3
ax.set_xlim(lo - 60, hi + 60)
ax.set_ylim(n - 0.5, -0.5)
ax.axvline(0, color=AXIS, lw=0.5, zorder=0)
for i, L in enumerate(LOCI):
    text(ax, lo - 110, i, L["locus"], size=5.8, style="italic", weight="bold", ha="right")
    x0, x1 = L["coord_min"] * 3, L["coord_max"] * 3
    if L["promoter_columns"]:
        rec = -int(L["cds_start"] or 0)
        ax.add_patch(Rectangle((x0, i - 0.27), rec - x0, 0.54, fc=REG_T, ec="none"))
        ax.add_patch(Rectangle((rec, i - 0.27), x1 - rec, 0.54, fc=DNA_T, ec="none"))
    else:
        ax.add_patch(Rectangle((x0, i - 0.27), x1 - x0, 0.54, fc=DNA_T, ec="none"))
    prev = None
    for t in L["tokens"]:
        x = t["coord_codons"] * 3
        ax.plot([x, x], [i - 0.38, i + 0.38], color=token_colour(t), lw=1.1, solid_capstyle="butt", zorder=3)
        dx = 95 if (prev is not None and x - prev < 170) else (-95 if any(
            0 < o["coord_codons"] * 3 - x < 170 for o in L["tokens"]) else 0)
        text(ax, x + dx, i - 0.60, str(NUMBER[(L["locus"], t["slot"])]), size=4.4, color=INK, ha="center")
        prev = x
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.set_yticks([])
ax.tick_params(axis="x", labelsize=5, length=2, width=0.5, pad=1)
ax.set_xlabel("bases from the start codon", fontsize=5.4, labelpad=1)

# block shapes + cohort columns
tx = fig.add_axes([0.37, mid + 0.035, 0.40, top - mid - 0.095])
tx.set_xlim(0, 1)
tx.set_ylim(n - 0.5, -0.5)
tx.axis("off")
COLS = [(0.00, "dna"), (0.205, "protein"), (0.41, "biophysical"), (0.60, "regulatory")]
for x, m in COLS:
    text(tx, x, -1.15, m, size=5.0, weight="bold", color=M[m] if m != "protein" else INK2)
text(tx, 0.815, -1.75, "tokens, TEST", size=5.0, weight="bold")
text(tx, 0.815, -1.05, "median", size=4.5, color=INK2)
text(tx, 0.965, -1.05, ">16", size=4.5, color=INK2, ha="right")
for i, L in enumerate(LOCI):
    for x, m in COLS:
        b = L["blocks"].get(m)
        text(tx, x, i, f"{b['shape'][1]}×{b['shape'][2]:,}" if b else "—", size=5.0,
             color=INK if b else AXIS)
    g = L["locus"]
    text(tx, 0.86, i, f"{C['tokens_median_by_locus'][g]:.0f}", size=5.0, ha="center")
    text(tx, 0.965, i, f"{C['loci_over_cap'][g]}", size=5.0, ha="right")

# the slot mask: the real (19, 17) valid tensor
sx = fig.add_axes([0.775, mid + 0.035, 0.135, top - mid - 0.095])
sx.set_xlim(-0.5, 16.5)
sx.set_ylim(n - 0.5, -0.5)
sx.axis("off")
V = np.array(T["valid"])
for i, row in enumerate(V):
    for j, v in enumerate(row):
        fc = SURFACE
        ec = GRID
        if j == 0:
            fc, ec = LF_TINT, LF
        elif v:
            t = next(t for t in LOCI[i]["tokens"] if t["slot"] == j)
            fc, ec = token_colour(t), token_colour(t)
        sx.add_patch(Rectangle((j - 0.42, i - 0.40), 0.84, 0.80, fc=fc, ec=ec, lw=0.35))
text(sx, -0.42, -1.75, "valid slots", size=5.0, weight="bold")
text(sx, 0, -1.05, "0", size=4.5, color=INK2, ha="center")
text(sx, 15.5, -1.05, "16", size=4.5, color=INK2, ha="center")

rx = fig.add_axes([0.915, mid + 0.035, 0.08, top - mid - 0.095])
rx.set_xlim(0, 1)
rx.set_ylim(n - 0.5, -0.5)
rx.axis("off")
text(rx, 0, -1.75, "[NULL] input", size=5.0, weight="bold")
text(rx, 0.44, -1.05, "log1p n", size=4.3, color=INK2, ha="right")
text(rx, 1.0, -1.05, "n/kept", size=4.5, color=INK2, ha="right")
for i, L in enumerate(LOCI):
    text(rx, 0.42, i, f"{L['stats'][0]:.2f}", size=4.8, ha="right")
    text(rx, 1.0, i, f"{L['stats'][1]:.4f}", size=4.8, ha="right")

fig.text(0.065, mid - 0.0, "■", color=REG_T, fontsize=7, va="center")
fig.text(0.078, mid - 0.0, "regulatory block", fontsize=5.0, color=INK2, va="center")
fig.text(0.165, mid - 0.0, "■", color=DNA_T, fontsize=7, va="center")
fig.text(0.178, mid - 0.0, "gene record", fontsize=5.0, color=INK2, va="center")
fig.text(0.245, mid - 0.0, "|", color=M["dna"], fontsize=7, va="center", fontweight="bold")
fig.text(0.252, mid - 0.0, "record token", fontsize=5.0, color=INK2, va="center")
fig.text(0.320, mid - 0.0, "|", color=M["regulatory"], fontsize=7, va="center", fontweight="bold")
fig.text(0.327, mid - 0.0, "regulatory token", fontsize=5.0, color=INK2, va="center")
fig.text(0.415, mid - 0.0, "|", color=M["uncalled"], fontsize=7, va="center", fontweight="bold")
fig.text(0.422, mid - 0.0, "uncalled base", fontsize=5.0, color=INK2, va="center")
fig.text(0.50, mid - 0.0, f"median, >16: tokens per isolate over {C['n']:,} TEST isolates", fontsize=5.0,
         color=INK2, va="center")
fig.text(0.775, mid - 0.0, f"slot 0 = [NULL]; {int(V.sum())} of {V.size} valid", fontsize=5.0,
         color=INK2, va="center")

# ---- b: the feature tensor, every channel of every real token -----------------
panel(fig, 0.005, mid - 0.045, "b",
      f"Token features: feat {shape(T['shapes']['feat'])}, the {len(TOKENS)} variant tokens and one [NULL]")
ALPHA_NT, ALPHA_AA = T["alphabets"]["nt"], T["alphabets"]["protein"]
BLOCKS = [("base", SL["dna"], ALPHA_NT, M["dna"]), ("residue", SL["protein"], ALPHA_AA, M["protein"]),
          ("props", SL["biophysical"], ["M", "p", "H"], M["biophysical"]),
          ("promoter", SL["regulatory"], ALPHA_NT, M["regulatory"]),
          ("Δ props", LAY["d_bio"], ["M", "p", "H"], M["biophysical"]),
          ("", [LAY["f_is_null"], LAY["f_is_null"] + 1], ["∅"], INK2),
          ("phase", [LAY["f_phase"], LAY["f_phase"] + 3], ["0", "1", "2"], INK2)]
bx = fig.add_axes([0.135, 0.015, 0.50, mid - 0.105])
rows = [("katG", {"slot": 0, "feat": next(L for L in LOCI if L["locus"] == "katG")["null_feat"]}, True)] + \
       [(g, t, False) for g, t in TOKENS]
nr, nc = len(rows), LAY["c_tok"]
bx.set_xlim(0, nc)
bx.set_ylim(nr, -2.15)
bx.axis("off")
props_max = max(abs(v) for _, t, _ in rows for a, b_ in (SL["biophysical"], LAY["d_bio"]) for v in t["feat"][a:b_]) or 1
chan_col = {}
for name, (a, b_), alpha, col in BLOCKS:
    bx.plot([a + 0.12, b_ - 0.12], [-1.4, -1.4], color=col, lw=1.3, solid_capstyle="butt")
    text(bx, (a + b_) / 2, -1.85, name, size=4.7, ha="center", color=INK2)
    for k, s in enumerate(alpha):
        chan_col[a + k] = col
        text(bx, a + k + 0.5, -0.75, s, size=3.9, ha="center", color=MUTED)
for r, (g, t, is_null) in enumerate(rows):
    f = t["feat"]
    for c in range(nc):
        v = f[c]
        col = chan_col[c]
        is_prop = SL["biophysical"][0] <= c < SL["biophysical"][1] or LAY["d_bio"][0] <= c < LAY["d_bio"][1]
        a = min(1, abs(v) / props_max) if is_prop else (1 if v else 0)
        fc = SURFACE
        if v:
            fc = M["uncalled"] if (not is_prop and c < SL["dna"][1] and ALPHA_NT[c - SL["dna"][0]] == "N") else col
        bx.add_patch(Rectangle((c + 0.05, r + 0.08), 0.90, 0.84, fc=SURFACE, ec=GRID, lw=0.3))
        if v:
            bx.add_patch(Rectangle((c + 0.05, r + 0.08), 0.90, 0.84, fc=fc, ec="none", alpha=max(a, 0.18)))
    if is_null:
        text(bx, -0.5, r + 0.5, "[NULL] katG", size=5.2, ha="right", color=INK2)
    else:
        text(bx, -0.5, r + 0.5, f"{NUMBER[(g, t['slot'])]}  {token_name(g, t)}", size=5.2, ha="right")
        dp = f[LAY["d_bio"][0]:LAY["d_bio"][1]]
        pr = f[SL["biophysical"][0]:SL["biophysical"][1]]
        bits = [f"{signed(t['coord_codons'])} codons"]
        if t["phase"] is not None:
            bits.append(f"phase {t['phase']}")
        if t["alt_aa"]:
            bits.append("M,p,H " + " ".join(signed(v) for v in pr))
            bits.append("Δ " + " ".join(signed(v) for v in dp))
        text(bx, nc + 0.5, r + 0.5, "  ".join(bits), size=4.5, color=INK2)
text(bx, nc + 0.5, 0.5, "coord 0 codons; all other channels 0", size=4.5, color=INK2)

for ext in ("pdf", "png"):
    fig.savefig(MAN / f"fig_lf_tokens.{ext}", dpi=300)
plt.close(fig)


# =============================================================== forward pass figure
def strip(ax, v, vmax, cmap=DIV):
    ax.imshow(np.asarray(v)[None, :], aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
        sp.set_color(AXIS)


def norm(v):
    return float(np.linalg.norm(v))


fig = plt.figure(figsize=(7.5, 9.3))
S = T["showcase"]
NS = T["null_show"]
kid = ORDER.index("katG")
K = LOCI[kid]
s315 = next(t for t in K["tokens"] if t["slot"] == S["slot"])
TOP = 0.972

# ---- a: embedding terms, at full width, for one variant token and the [NULL] token ----
panel(fig, 0.005, TOP, "a", f"Token embedding, katG: slot {S['slot']} ({token_name('katG', s315)}) and slot 0 ([NULL])")
vmax = 3.0
left = [(f"feat[katG, {S['slot']}]  (45)", s315["feat"], None),
        (f"sinusoid(coord = {S['coord_codons']:.2f})  (32)", S["sinusoid"], None),
        ("tok_proj: Linear 45→128", S["tok_proj"], 1),
        ("pos_proj: Linear 32→128", S["pos_proj"], 1),
        ("locus_emb[katG]", S["locus_emb"], 1),
        ("sum of the three", S["sum"], 1),
        ("× (1 + film_scale) + film_shift", S["film"], 1),
        ("LayerNorm: the stage 1 input", S["layernorm"], 1)]
right = [("feat[katG, 0]  (45)", K["null_feat"], None),
         ("sinusoid(coord = 0.00)  (32)", [0.0] * 16 + [1.0] * 16, None),
         ("tok_proj + pos_proj + locus_emb", np.add(np.add(NS["tok_proj"], NS["pos_proj"]), NS["locus_emb"]), 1),
         ("null_emb[katG]", NS["null_emb"], 1),
         (f"null_proj: Linear 2→128, input ({K['stats'][0]:.4f}, {K['stats'][1]:.4f})", NS["null_proj"], 1),
         ("sum of the three", NS["sum"], 1),
         ("× (1 + film_scale) + film_shift", NS["film"], 1),
         ("LayerNorm: the stage 1 input", NS["layernorm"], 1)]
y0, dy = TOP - 0.022, 0.0255
for col, (x0, rows_) in enumerate(((0.005, left), (0.505, right))):
    for r, (lab, v, _) in enumerate(rows_):
        y = y0 - r * dy
        fig.text(x0, y, lab, fontsize=5.0, color=INK2, va="top")
        ax_ = fig.add_axes([x0, y - dy + 0.0065, 0.41, 0.0115])
        strip(ax_, v, vmax)
        fig.text(x0 + 0.415, y - dy + 0.012, f"‖·‖ {norm(v):.2f}", fontsize=4.7, color=INK2, va="center")
ya = y0 - len(left) * dy - 0.004
cax = fig.add_axes([0.005, ya - 0.006, 0.14, 0.006])
cax.imshow(np.linspace(-1, 1, 200)[None, :], aspect="auto", cmap=DIV, extent=(-vmax, vmax, 0, 1))
cax.set_yticks([])
cax.tick_params(axis="x", labelsize=4.6, length=1.5, width=0.4, pad=1)
cax.set_xticks([-vmax, 0, vmax])
for sp in cax.spines.values():
    sp.set_linewidth(0.3)
fig.text(0.155, ya - 0.003, "value of each channel", fontsize=4.8, color=INK2, va="center")


def attn_map(ax, A, labels, title, vmax=None, annotate=True, size=4.6, labsize=4.6):
    A = np.asarray(A)
    im = ax.imshow(A, cmap=SEQ, vmin=0, vmax=vmax if vmax else A.max(), interpolation="nearest")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=labsize)
    ax.set_yticks(range(len(labels)), labels, fontsize=labsize)
    ax.tick_params(length=0, pad=1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=5.2, color=INK2, loc="left", pad=2)
    if annotate:
        lim = (vmax if vmax else A.max()) * 0.55
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax.text(j, i, f"{A[i, j]:.3f}".lstrip("0"), ha="center", va="center", fontsize=size,
                        color=SURFACE if A[i, j] > lim else INK)
    return im


# ---- b: stage 1 ------------------------------------------------------------------
yb = ya - 0.034
panel(fig, 0.005, yb, "b",
      f"Within each locus: shared 2-layer encoder, {shape(T['shapes']['stage1_in'])} with padding masked; attention weights")
hb = 0.092
# laid out by hand: two maps per locus, sized by token count
xpos = 0.058
for g in ("katG", "gyrA", "gid"):
    L_ = next(L for L in LOCI if L["locus"] == g)
    labs = ["[NULL]"] + [token_name(g, t).split(" ", 1)[1] for t in L_["tokens"]]
    k = len(labs)
    w = {2: 0.062, 3: 0.078, 4: 0.098}[k]
    for li, a in enumerate(L_["attention_stage1"]):
        ax_ = fig.add_axes([xpos, yb - 0.112, w, hb])
        attn_map(ax_, a, labs, f"{g}, layer {li + 1}", vmax=1.0)
        if li:
            ax_.set_yticks([])
        xpos += w + (0.008 if li == 0 else 0.062)

# ---- c: locus summaries and stage 2 ----------------------------------------------
yc = yb - 0.168
panel(fig, 0.005, yc, "c",
      f"Across loci: summaries z {shape(T['shapes']['z'])}, keyed norm, 2-layer encoder over 19 tokens, fused {shape(T['shapes']['fused'])}")
hc = 0.158
yb0 = yc - 0.020 - hc
A2 = np.asarray(T["attention_stage2"])
vz = 3.0


def vectors(ax, Z, title, labels):
    ax.imshow(np.asarray(Z), aspect="auto", cmap=DIV, vmin=-vz, vmax=vz, interpolation="nearest")
    ax.set_yticks(range(19), ORDER if labels else [""] * 19, fontsize=4.0)
    ax.set_xticks([0, 127], ["1", "128"], fontsize=4.2)
    ax.tick_params(length=0, pad=1)
    ax.set_title(title, fontsize=5.0, color=INK2, loc="left", pad=2)
    ax.set_xlabel("channel", fontsize=4.6, labelpad=-3)
    for sp in ax.spines.values():
        sp.set_linewidth(0.3)
        sp.set_color(AXIS)


W, X = 0.145, [0.042, 0.198, 0.392, 0.543, 0.815]
vectors(fig.add_axes([X[0], yb0, W, hc]), T["z"], "z: stage 1 output, slot 0", True)
vectors(fig.add_axes([X[1], yb0, W, hc]), T["z_norm"], "z after KeyedTokenNorm", False)
for li in range(2):
    ax_ = fig.add_axes([X[2 + li], yb0, W, hc])
    im = attn_map(ax_, A2[li], ORDER, f"attention weights, layer {li + 1}", vmax=float(A2.max()),
                  annotate=False, labsize=4.0)
    ax_.set_aspect("auto")
    ax_.set_xlabel("attended locus", fontsize=4.6, labelpad=1)
    if li:
        ax_.set_yticks([])
    else:
        ax_.set_ylabel("attending locus", fontsize=4.6, labelpad=0.5)
cax = fig.add_axes([X[3] + W + 0.006, yb0, 0.006, hc])
cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=4.4, length=1.5, width=0.4, pad=1)
cb.outline.set_linewidth(0.3)
vectors(fig.add_axes([X[4], yb0, W, hc]), T["fused"], "fused: stage 2 output", True)

# ---- d: readout ------------------------------------------------------------------
yd = yb0 - 0.052
panel(fig, 0.005, yd, "d", "Readout: one learned query, 4 heads, cross-attention over the 19 fused tokens")
ax_ = fig.add_axes([0.05, yd - 0.135, 0.46, 0.112])
q = np.asarray(T["attention_query"])
ax_.bar(range(19), q, color=LF, width=0.72)
ax_.axhline(1 / 19, color=MUTED, lw=0.8, ls="--")
text(ax_, 18.6, 1 / 19 + 0.0028, f"uniform 1/19 = {1 / 19:.4f}", size=4.8, color=INK2, ha="right")
ax_.set_xticks(range(19), ORDER, rotation=90, fontsize=4.6)
ax_.tick_params(axis="y", labelsize=4.8, length=1.5, width=0.4, pad=1)
ax_.tick_params(axis="x", length=0, pad=1)
ax_.set_ylim(0, 0.066)
ax_.set_ylabel("attention weight", fontsize=5.0, labelpad=1)
for sp in ("top", "right"):
    ax_.spines[sp].set_visible(False)
tx = fig.add_axes([0.54, yd - 0.150, 0.455, 0.14])
tx.set_xlim(0, 1)
tx.set_ylim(0, 1)
tx.axis("off")
STEPS = [("drug_queries", shape(T["shapes"]["query"]), f"‖·‖ {norm(T['query']):.2f}"),
         ("pool_attn output", shape(T["shapes"]["pooled"]), f"‖·‖ {norm(T['pooled']):.2f}"),
         ("LayerNorm", shape(T["shapes"]["pooled"]), f"‖·‖ {norm(T['pooled_norm']):.2f}"),
         ("fc1, ReLU: Linear 128→256", shape(T["shapes"]["h"]), f"{T['h_active']} of 256 units > 0"),
         ("fc_out: Linear 256→1", shape(T["shapes"]["logit"]), f"logit {signed(T['logit'], 4)}"),
         ("sigmoid(logit)", "", f"P(susceptible) = {T['P_S']:.4f}"),
         ("1 − sigmoid(logit)", "", f"P(resistant) = {T['P_R']:.4f}"),
         ("TEST operating point", "", f"P(S) < {C['threshold_P_S']:.2f} calls resistant"),
         (f"TEST, n = {C['n']:,} ({C['n_resistant']:,} R)", "", f"AUC {C['auc_rescored']:.4f}, sens {C['sens']:.3f}, spec {C['spec']:.3f}")]
for r, (a_, b_, c_) in enumerate(STEPS):
    y = 0.97 - r * 0.112
    last = a_.startswith("1 −")
    text(tx, 0.0, y, a_, size=5.3, weight="bold" if last else "normal")
    text(tx, 0.44, y, b_, size=5.0, color=INK2)
    text(tx, 0.60, y, c_, size=5.3, color=LF if last else INK, weight="bold" if last else "normal")

# ---- e: parameters ---------------------------------------------------------------
ye = yd - 0.182
panel(fig, 0.005, ye, "e", f"Parameters: {T['n_params']:,} by module")
tx = fig.add_axes([0.02, ye - 0.100, 0.96, 0.092])
tx.set_xlim(0, 1)
tx.set_ylim(8.5, -0.5)
tx.axis("off")
P = T["params"]
for i, p_ in enumerate(P):
    col, r = (0, i) if i < 8 else (1, i - 8)
    x0 = 0.0 if col == 0 else 0.52
    text(tx, x0, r, p_["module"], size=5.3, weight="bold")
    text(tx, x0 + 0.12, r, p_["shape"], size=5.1, color=INK2)
    text(tx, x0 + 0.40, r, f"{p_['params']:,}", size=5.3, ha="right")
    text(tx, x0 + 0.46, r, f"{100 * p_['params'] / T['n_params']:.1f}%", size=5.0, color=INK2, ha="right")
text(tx, 0.52, 8, "total", size=5.3, weight="bold")
text(tx, 0.92, 8, f"{T['n_params']:,}", size=5.3, weight="bold", ha="right")

for ext in ("pdf", "png"):
    fig.savefig(MAN / f"fig_lf_forward_pass.{ext}", dpi=300)
print(f"wrote fig_lf_tokens and fig_lf_forward_pass to {MAN}")

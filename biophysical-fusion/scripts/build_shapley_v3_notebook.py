"""Generate notebooks/paper/shapley_locusfusion_v3.ipynb: exact Shapley over the LocusFusion v3
19-locus cells, then a comparison with the expected-gradients SHAP run.

    python scripts/build_shapley_v3_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/paper/shapley_locusfusion_v3.ipynb
    (or: sbatch scripts/sbatch/build_shapley_v3_notebook.sh)

Reads only CSVs: results/analysis/shap_locusfusion_v3/shapley/ (scripts/shapley_locusfusion_v3.py)
and results/analysis/shap_locusfusion_v3/ (scripts/shap_locusfusion_v3.py, EG). Figures go to
results/figures/shapley_v3/ (PDF + PNG). Colours come from paper_style.py.
"""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "notebooks" / "paper" / "shapley_locusfusion_v3.ipynb"
cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md("""
# Shapley attribution for LocusFusion v3 — and how it compares with SHAP

Exact Shapley values for the five 19-locus v3 cells × 11 drugs
(`scripts/shapley_locusfusion_v3.py`), on the same 300 explained test isolates
per drug as the SHAP run (54 for levofloxacin). Part 1 shows the Shapley
results alone; part 2 compares them with the expected-gradients SHAP run and
explains why the two differ.

> Generated — edit `scripts/build_shapley_v3_notebook.py`, not the `.ipynb`.
""")

code(r'''
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from scipy import stats

_here = Path.cwd()
PROJECT = next(p for p in [_here, *_here.parents] if (p / "bigtb_ref.py").exists())
sys.path.insert(0, str(PROJECT))
import results_helpers as RH
import paper_style as PS
from paper_style import MODEL, MODALITY, ramp
from results_helpers import tidy
PS.apply(RH)
N = PS.NEUTRAL
LF_C, LASSO_C = MODEL["locusfusion"], MODEL["lasso"]
mpl.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "axes.titlesize": 9,
                     "pdf.fonttype": 42})
W = 6.5

A = PROJECT / "results" / "analysis" / "shap_locusfusion_v3"
S = A / "shapley"
FIG = PROJECT / "results" / "figures" / "shapley_v3"
FIG.mkdir(parents=True, exist_ok=True)
def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.show()
def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

# Shapley
smod = pd.read_csv(S / "modality_shapley.csv")
sloc = pd.read_csv(S / "locus_shapley.csv")
sgate = pd.read_csv(S / "gate_shapley.csv")
sperm = pd.read_csv(S / "perm_check.csv")
scat = pd.read_csv(S / "who" / "shap_category_shares.csv")
side = pd.read_csv(S / "who" / "side_by_side.csv")
lcat = pd.read_csv(S / "who" / "locus_categories.csv")
# SHAP (expected gradients, training background)
emod = pd.read_csv(A / "modality_shares.csv").query("background == 'train'")
eloc = pd.read_csv(A / "locus_shares.csv").query("background == 'train'")
egate = pd.read_csv(A / "gate.csv").query("background == 'train'")
ecat = pd.read_csv(A / "who" / "shap_category_shares.csv").query("background == 'train'")

CELLS = ["sd19_dna_unk", "sd19_dna_protein_unk", "sd19_dna_biophysical_unk",
         "sd19_dna_regulatory_unk", "sd19_all_modalities_unk"]
CNAME = {"sd19_dna_unk": "DNA", "sd19_dna_protein_unk": "DNA + protein",
         "sd19_dna_biophysical_unk": "DNA + biophysical",
         "sd19_dna_regulatory_unk": "DNA + regulatory",
         "sd19_all_modalities_unk": "all modalities"}
SH = ["dna", "protein", "biophysical", "regulatory"]
DRUGS = sorted(smod.drug.unique())
ABBR = {"AMIKACIN": "AMK", "CAPREOMYCIN": "CAP", "ETHAMBUTOL": "EMB", "ETHIONAMIDE": "ETO",
        "ISONIAZID": "INH", "KANAMYCIN": "KAN", "LEVOFLOXACIN": "LFX", "MOXIFLOXACIN": "MFX",
        "PYRAZINAMIDE": "PZA", "RIFAMPICIN": "RIF", "STREPTOMYCIN": "STM"}
OWN = (lcat.assign(own=lcat.category.isin(["own_t1", "own_t2"]))
       .pivot(index="drug", columns="locus", values="own"))
UNIFORM = OWN.mean(axis=1).mean()               # own-WHO share if every locus got the same

# per-cell modality shares, both methods, one tidy frame
_s = (smod[smod.game == "modality"].pivot_table(index="cell", columns="player",
                                                 values="share", aggfunc="mean"))
_e = emod.groupby("cell")[[f"share_{m}" for m in SH]].mean()
_e.columns = SH
MOD = pd.concat({"Shapley": _s.reindex(columns=SH), "SHAP": _e}, names=["method"])
_unc = (smod[(smod.game == "called_uncalled") & smod.player.str.endswith(":uncalled")]
        .groupby(["drug", "cell"]).share.sum().groupby("cell").mean())
UNC = pd.DataFrame({"Shapley": _unc, "SHAP": emod.groupby("cell").share_uncalled.mean()})
print(f"Shapley: {len(sgate)} drug x cell runs; SHAP: {len(egate)}; figures -> {FIG}")
''')

# =================================================================== PART 1
md("""
## Part 1 — Shapley results

### 1.1 Checks

Each isolate's Shapley values must sum to its logit minus the all-absent logit
(efficiency). The modality game is exact (all 2^k coalitions); the locus game
uses 32 antithetic permutations, or an exact enumeration when an isolate
differs from H37Rv at nine or fewer loci.
""")

code(r'''
chk = sgate.groupby("cell").agg(
    runs=("drug", "size"), auc_gate_max_diff=("auc_abs_diff", "max"),
    same_isolates_as_SHAP=("pick_matches_eg", "all"),
    max_efficiency_err_modality=("modality_max_eff_err", "max"),
    max_efficiency_err_called_uncalled=("called_uncalled_max_eff_err", "max"),
    f_all_absent_median=("f_all_absent", "median"))
chk.index = [CNAME[c] for c in chk.index]
pc = sperm.groupby("n_perm").abs_diff.max()
print("largest locus-share change against the most permutations:",
      ", ".join(f"{k} perm {v:.4f}" for k, v in pc.items()))
chk.round(5)
''')

md("""
All 55 runs pass: the rescored AUC equals the reported one, the isolates are
the SHAP run's, and the worst efficiency error is below 1e-4 logit. Locus
shares at 32 permutations move by at most 0.004 against 128. The all-absent
input (every locus identical to H37Rv) scores −1.3 to −6.2 logit, i.e. firmly
susceptible, so positive values push toward resistance.
""")

md("""
### 1.2 Modality shares

Share of mean |φ| per modality, mean over the 11 drugs. Right: the share taken
by the uncalled-base channels (N in DNA and regulatory, `-`/`X` in protein)
when they are split off as their own players.
""")

code(r'''
M = MOD.loc["Shapley"].reindex(CELLS[::-1]).fillna(0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True,
                             gridspec_kw=dict(width_ratios=[4, 1.1], wspace=0.16))
y = np.arange(len(M)); left = np.zeros(len(M))
for m in SH:
    a1.barh(y, M[m], left=left, height=0.62, color=MODALITY[m], edgecolor=N["surface"],
            linewidth=1.0, label=m, zorder=2)
    left += M[m].values
a1.set_xlim(0, 1); a1.set_xlabel("share of mean |φ|")
a1.set_yticks(y); a1.set_yticklabels([CNAME[c] for c in M.index])
u = UNC.Shapley.reindex(M.index)
a2.barh(y, u, height=0.62, color=MODALITY["uncalled"], zorder=2)
for yi, v in zip(y, u):
    a2.annotate(f"{v:.3f}", (v, yi), xytext=(3, 0), textcoords="offset points",
                va="center", fontsize=7.4, color=N["ink2"])
a2.set_xlim(0, 0.15); a2.set_xticks([0.05, 0.10]); a2.set_xlabel("uncalled-base share")
for ax in (a1, a2):
    tidy(ax); clean(ax)
a1.legend(loc="upper center", bbox_to_anchor=(0.62, -0.32), ncol=4, handlelength=1.1)
save(fig, "shapley_modality_shares")
M.assign(uncalled=u).round(3).rename(index=CNAME)
''')

md("""
DNA takes 0.82–1.00 in every cell. In the all-modalities cell protein takes
0.09, regulatory 0.05 and biophysical 0.04. Uncalled bases take 0.05–0.07,
almost all of it DNA `N`.
""")

md("""
### 1.3 Loci

Locus share of mean |φ| per drug, for the best cell (DNA + regulatory) and the
all-modalities cell. Outlined: the drug's own WHO tier 1/2 genes.
""")

code(r'''
LOCI = (sloc[sloc.cell == "sd19_all_modalities_unk"].groupby("locus").share.mean()
        .sort_values(ascending=False).index.tolist())
own = OWN.reindex(index=DRUGS, columns=LOCI)
seq = LinearSegmentedColormap.from_list("lf_seq", [N["surface"], ramp(LF_C, 3)[1], ramp(LF_C, 3)[0]])
panels = ["sd19_dna_regulatory_unk", "sd19_all_modalities_unk"]
vmax = sloc[sloc.cell.isin(panels)].share.max()
fig, axes = plt.subplots(len(panels), 1, figsize=(W, 5.6), sharex=True)
for ax, c in zip(axes, panels):
    H = (sloc[sloc.cell == c].pivot(index="drug", columns="locus", values="share")
         .reindex(index=DRUGS, columns=LOCI))
    im = ax.imshow(H.values, cmap=seq, vmin=0, vmax=vmax, aspect="auto")
    for (i, j), o in np.ndenumerate(own.values):
        if o:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=N["ink"], lw=1.1))
    ax.set_yticks(range(len(DRUGS))); ax.set_yticklabels([ABBR[d] for d in DRUGS])
    ax.set_title(CNAME[c], loc="left", color=N["ink2"])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[-1].set_xticks(range(len(LOCI)))
axes[-1].set_xticklabels(LOCI, rotation=60, ha="right", fontstyle="italic")
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("locus share of mean |φ|"); cb.outline.set_visible(False)
save(fig, "shapley_locus_heatmap")
''')

md("""
The outlined cells carry most of the colour: rpoB for rifampicin (0.51), rrs for
kanamycin (0.54) and capreomycin (0.37), katG for isoniazid (0.38), gyrA for the
fluoroquinolones, pncA, embB and rpsL for their drugs. The rest spreads over the
MDR-linked genes (rpoB, katG, gyrA) whatever the drug.
""")

md("""
### 1.4 Own WHO genes against other drugs' genes

All 19 loci are WHO tier 1/2 genes for some drug, so each locus is either the
drug's own WHO gene or another drug's. Bars: the share on each; the 19-locus
LASSO's |coefficient| mass for comparison. Tick: the share of variant tokens
that sit on own genes; dashed line: an equal share per locus.
""")

code(r'''
def prof(cat, cell):
    s = cat[cat.cell == cell]
    return s[["own_t1", "own_t2", "other_who"]].mean(), s.own_who_null_tokens.mean()
lasso = side[(side.source == "lasso") & (side.cell == "lasso_19locus_vcf_full")]
rows = {CNAME[c]: prof(scat, c) for c in CELLS}
rows["LASSO, 19 loci"] = (lasso[["own_t1", "own_t2", "other_who"]].mean(), np.nan)
names = list(rows)[::-1]
fig, ax = plt.subplots(figsize=(W, 2.5))
for yi, nm in enumerate(names):
    v, null = rows[nm]
    shades = ramp(LASSO_C if nm.startswith("LASSO") else LF_C, 3)
    left = 0
    for k, colour in zip(["own_t1", "own_t2", "other_who"], shades):
        ax.barh(yi, v[k], left=left, height=0.62, color=colour, edgecolor=N["surface"],
                linewidth=1.0, zorder=2)
        left += v[k]
    if not np.isnan(null):
        ax.plot([null, null], [yi - 0.36, yi + 0.36], color=N["ink"], lw=1.4, zorder=3)
ax.axvline(UNIFORM, color=N["muted"], lw=0.9, ls=(0, (3, 3)), zorder=1)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
ax.set_xlim(0, 1); ax.set_xlabel("share of mean |φ| (LASSO: |coefficient| mass)")
tidy(ax); clean(ax)
handles = ([Patch(color=c) for c in ramp(LF_C, 3)]
           + [Line2D([0], [0], color=N["ink"], lw=1.4),
              Line2D([0], [0], color=N["muted"], lw=0.9, ls=(0, (3, 3)))])
ax.legend(handles, ["own WHO tier 1", "own WHO tier 2", "other drugs' WHO genes",
                    "variant-token share on own genes", f"equal share per locus ({UNIFORM:.3f})"],
          loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=3)
save(fig, "shapley_who_categories")
own_hit = scat.assign(hit=scat.top_locus_category.isin(["own_t1", "own_t2"])).groupby("cell").hit.sum()
pd.DataFrame({nm: dict(own_who=v[["own_t1", "own_t2"]].sum(), other_who=v["other_who"],
                       token_null=null) for nm, (v, null) in rows.items()}).T.assign(
    top_locus_own=[f"{int(own_hit[c])}/11" for c in CELLS] + [
        f"{int(lasso.top_locus_category.isin(['own_t1', 'own_t2']).sum())}/11"]).round(3)
''')

md("""
v3 puts 0.37–0.39 of its attribution on each drug's own WHO genes in every
cell, above the 19-locus LASSO (0.34) and four to five times the
variant-token share (0.07–0.09). The top locus is an own WHO gene for 10–11 of
11 drugs. The remaining ~0.62 is on other drugs' genes.
""")

code(r'''
top3 = (sloc[sloc.cell.isin(["sd19_all_modalities_unk", "sd19_dna_regulatory_unk"]) & (sloc["rank"] <= 3)]
        .assign(item=lambda d: d.locus + " " + d.share.map(lambda v: f"{v:.2f}"))
        .groupby(["drug", "cell"]).item.apply(", ".join).unstack()
        [["sd19_dna_regulatory_unk", "sd19_all_modalities_unk"]].rename(columns=CNAME))
top3
''')

md("""
One entry is an artefact: tlyA is levofloxacin's top locus in the
all-modalities cell (0.25), but every explained isolate differs from H37Rv
there and gets the same +0.83 logit, so it is a constant offset from the
all-H37Rv baseline rather than a signal that separates resistant from
susceptible isolates.
""")

# =================================================================== PART 2
md("""
## Part 2 — SHAP against Shapley

### 2.1 What the two methods compute

Both split a prediction among input features, and both are "Shapley-style" in
name, but they answer different questions.

- **SHAP here is expected gradients** (`shap.GradientExplainer`): pick a
  background isolate, slide the input along a straight line from it to the
  explained isolate, and add up gradients along the way, averaged over 100
  training isolates as backgrounds and 64 sampled points. It attributes
  individual input columns; we then sum columns into modalities and loci. It
  is fast, but its sums equal the prediction only if the model is smooth along
  every path.
- **Shapley values** treat a small number of *players* (here 4 modalities,
  or 19 loci) and average each player's marginal contribution over coalitions
  of the others, with "absent" meaning the player's input set to the H37Rv
  reference. The modality game is computed exactly; the locus game by
  permutation sampling. The sums equal f(isolate) − f(all reference) by
  construction.

### 2.2 Why they disagree here

1. **LocusFusion is not smooth along the SHAP path.** It builds tokens
   inside the forward pass: a column becomes a token only if it differs from
   H37Rv, and each locus keeps its first 16. Interpolating between two isolates
   makes tokens appear and disappear abruptly; gradients never see those jumps,
   so part of the score goes unattributed. The symptom is completeness: SHAP's
   per-isolate sums correlate with the logit gap at a median r of 0.35, with
   several runs near zero or negative.
2. **Different reference points.** SHAP compares each isolate with real
   training isolates, so a variant shared with the background gets little
   credit; Shapley compares with H37Rv, so every variant the isolate carries is
   credited. That is why constant offsets such as levofloxacin's tlyA appear
   only under Shapley, and why near-universal lineage columns move between the
   two.
3. **Interactions are split differently.** Protein and biophysical features
   are only read at positions the DNA input marks as variant, so they act
   almost entirely through an interaction with DNA. Shapley splits an
   interaction evenly between the players involved; gradients assign it to
   whichever input the gradient is steepest in, which here was the dense
   biophysical block.

### 2.3 Which to trust in this setting

Shapley. Its attributions add up exactly, its notion of "absent" (identical to
H37Rv) is the model's own notion of "no variant" on delta input, and it does not
depend on interpolating across token boundaries. Its results also agree with
independent evidence: the AUC ablations (protein and biophysical add ~0 at 19
loci) and the WHO catalogue (own genes on top for 10–11 of 11 drugs). Its
limitation is that zeroing a modality the model was trained with is out of
distribution, so a modality's value means "what the model loses without it",
not "what it reads". SHAP stays useful as a column-level map within a locus,
not for shares across modalities or loci.
""")

md("""
### 2.4 Modality shares, side by side

Open circles: SHAP; filled: Shapley. One row per modality in each cell.
""")

code(r'''
rows = [(c, m) for c in CELLS[::-1] for m in SH if not np.isnan(MOD.loc[("Shapley", c), m])]
fig, ax = plt.subplots(figsize=(W, 0.21 * len(rows) + 0.9))
y = np.arange(len(rows))[::-1]
for yi, (c, m) in zip(y, rows):
    e, s = MOD.loc[("SHAP", c), m], MOD.loc[("Shapley", c), m]
    ax.plot([e, s], [yi, yi], color=N["axis"], lw=1.4, zorder=1)
    ax.scatter([e], [yi], s=34, color=N["surface"], edgecolor=MODALITY[m], linewidth=1.6, zorder=3)
    ax.scatter([s], [yi], s=34, color=MODALITY[m], edgecolor=N["surface"], linewidth=0.8, zorder=3)
ax.set_yticks(y); ax.set_yticklabels([f"{CNAME[c]} · {m}" for c, m in rows], fontsize=7.4)
breaks = [i for i in range(1, len(rows)) if rows[i][0] != rows[i - 1][0]]
for b in breaks:
    ax.axhline(y[b] + 0.5, color=N["grid"], lw=0.8, zorder=0)
ax.set_xlim(-0.02, 1.02); ax.set_xlabel("modality share")
tidy(ax); clean(ax)
ax.legend([Line2D([0], [0], marker="o", lw=0, markersize=6, markerfacecolor=N["surface"],
                  markeredgecolor=N["ink2"], markeredgewidth=1.4),
           Line2D([0], [0], marker="o", lw=0, markersize=6, color=N["ink2"])],
          ["SHAP (expected gradients)", "Shapley"], loc="upper center",
          bbox_to_anchor=(0.5, -0.12), ncol=2)
save(fig, "shap_vs_shapley_modalities")
MOD.unstack(0).swaplevel(axis=1).sort_index(axis=1).reindex(CELLS).rename(index=CNAME).round(3)
''')

md("""
SHAP gives biophysical 0.50–0.62 and protein 0.19–0.54; Shapley gives them
0.04–0.10 and DNA 0.82–0.93. Regulatory is the one modality Shapley credits
more (0.05–0.07 vs 0.01–0.02). The uncalled-base share is similar under both
(Shapley 0.05–0.07, SHAP 0.03–0.08).
""")

md("""
### 2.5 Locus shares, side by side

Each point is one drug × locus in the all-modalities cell. Copper: the drug's
own WHO gene; grey: another drug's.
""")

code(r'''
c = "sd19_all_modalities_unk"
J = (eloc[eloc.cell == c][["drug", "locus", "share"]]
     .merge(sloc[sloc.cell == c][["drug", "locus", "share"]], on=["drug", "locus"],
            suffixes=("_shap", "_shapley")))
J["own"] = [bool(OWN.loc[d, l]) for d, l in zip(J.drug, J.locus)]
rho, p = stats.spearmanr(J.share_shap, J.share_shapley)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.6))
lim = max(J.share_shap.max(), J.share_shapley.max()) * 1.05
a1.plot([0, lim], [0, lim], color=N["axis"], lw=0.9, zorder=1)
for own_, col, z in ((False, N["muted"], 2), (True, LF_C, 3)):
    s = J[J.own == own_]
    a1.scatter(s.share_shap, s.share_shapley, s=18 if own_ else 12, color=col,
               edgecolor=N["surface"], linewidth=0.5, zorder=z,
               label="own WHO gene" if own_ else "other drug's WHO gene")
a1.set_xlim(0, lim); a1.set_ylim(0, lim); a1.set_aspect("equal")
a1.set_xlabel("SHAP locus share"); a1.set_ylabel("Shapley locus share")
a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=7.2, handletextpad=0.2)
tidy(a1, ygrid=True); clean(a1)
# own-WHO share per cell, both methods, LASSO reference
yy = np.arange(len(CELLS))[::-1]
e_own = ecat.groupby("cell").own_who.mean().reindex(CELLS)
s_own = scat.groupby("cell").own_who.mean().reindex(CELLS)
for yi, cc in zip(yy, CELLS):
    a2.plot([e_own[cc], s_own[cc]], [yi, yi], color=N["axis"], lw=1.4, zorder=1)
a2.scatter(e_own, yy, s=34, color=N["surface"], edgecolor=LF_C, linewidth=1.6, zorder=3, label="SHAP")
a2.scatter(s_own, yy, s=34, color=LF_C, edgecolor=N["surface"], linewidth=0.8, zorder=3, label="Shapley")
lasso_own = lasso.own_who.mean()
a2.axvline(lasso_own, color=LASSO_C, lw=1.1, ls=(0, (3, 3)), zorder=0)
a2.axvline(UNIFORM, color=N["muted"], lw=0.9, ls=(0, (1, 2)), zorder=0)
a2.set_yticks(yy); a2.set_yticklabels([CNAME[cc] for cc in CELLS])
a2.set_xlim(0, 0.5); a2.set_xlabel("share on own WHO genes")
tidy(a2); clean(a2)
a2.legend([Line2D([0], [0], marker="o", lw=0, markersize=6, markerfacecolor=N["surface"],
                  markeredgecolor=LF_C, markeredgewidth=1.4),
           Line2D([0], [0], marker="o", lw=0, markersize=6, color=LF_C),
           Line2D([0], [0], color=LASSO_C, lw=1.1, ls=(0, (3, 3))),
           Line2D([0], [0], color=N["muted"], lw=0.9, ls=(0, (1, 2)))],
          ["SHAP", "Shapley", f"LASSO, 19 loci ({lasso_own:.2f})", f"equal share ({UNIFORM:.2f})"],
          loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, fontsize=7.2)
save(fig, "shap_vs_shapley_loci")
print(f"locus shares, all-modalities cell: Spearman rho {rho:+.2f} (p {p:.1e}, n {len(J)})")
pd.DataFrame({"SHAP": e_own, "Shapley": s_own}).rename(index=CNAME).round(3)
''')

md("""
The two broadly agree on which loci matter (Spearman ρ = 0.66 over drug ×
locus), but Shapley moves attribution onto own WHO genes (copper points above
the diagonal): own-WHO share 0.37–0.39 against SHAP's 0.16–0.23, and above the
LASSO in every cell. The top locus is an own WHO gene for 10 of 11 drugs under
Shapley against 4 under SHAP.
""")

code(r'''
tl = pd.DataFrame({
    "SHAP": ecat[ecat.cell == c].set_index("drug").top_locus,
    "Shapley": scat[scat.cell == c].set_index("drug").top_locus}).reindex(DRUGS)
tl["own WHO (SHAP / Shapley)"] = [
    f"{'yes' if OWN.loc[d, a] else 'no'} / {'yes' if OWN.loc[d, b] else 'no'}"
    for d, a, b in zip(tl.index, tl.SHAP, tl.Shapley)]
tl
''')

md("""
### 2.6 Do the attributions add up?
""")

code(r'''
comp = pd.DataFrame({
    "SHAP: median completeness r": egate.groupby("cell").completeness_r.median(),
    "SHAP: runs with r < 0.1": egate.groupby("cell").completeness_r.apply(lambda x: int((x < 0.1).sum())),
    "Shapley: max |sum φ − Δf| (logit)": sgate.groupby("cell").modality_max_eff_err.max(),
}).reindex(CELLS).rename(index=CNAME)
comp.round(5)
''')

md("""
SHAP's per-isolate sums track the prediction loosely (median r 0.35; 13 of 55
runs below 0.1). Shapley's sums match it to within 1e-4 logit in every run.
""")

md("""
## Caveats

- Shapley's "absent" (input identical to H37Rv) is out of distribution for a
  model trained with that input: values mean what the model loses without a
  player, not what it reads. Modality shares follow the model's design —
  protein and biophysical act only through DNA-marked positions.
- Shapley's H37Rv reference credits variants that every isolate carries
  (levofloxacin tlyA, +0.83 logit in all 54 isolates) as constant offsets.
- Locus values use 32 permutations (max share change 0.004 against 128).
- One fold (best_fold) per drug, one seed; levofloxacin explains 54 isolates.
- The uncalled-base share (0.05–0.07) answers a different question from its
  0.012 AUC cost, which came from retraining without the channel.
- LASSO |coefficient| mass and attribution shares are different quantities;
  compare their profiles, not their absolute values.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                                                  "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(cells)} cells)")

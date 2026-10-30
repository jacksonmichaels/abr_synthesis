"""Generate notebooks/shap_locusfusion_v3.ipynb: SHAP over the LocusFusion v3 19-locus cells.

    python scripts/build_shap_v3_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/shap_locusfusion_v3.ipynb

Reads only CSVs written by scripts/shap_locusfusion_v3.py (--aggregate) and
scripts/shap_v3_who_compare.py, plus the v3 result JSONs for CV AUC. Figures go
to results/figures/shap_v3/ (PDF + PNG). Colours come from paper_style.py.
"""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "notebooks" / "shap_locusfusion_v3.ipynb"
cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md("""
# SHAP over LocusFusion v3 — modalities and loci

Expected-gradients attributions for the five 19-locus v3 cells × 11 drugs
(`scripts/shap_locusfusion_v3.py`; 300 explained test isolates, 64 samples,
100-isolate training background, best fold). Two questions: what share of the
attribution each input modality takes, and which loci the model reads relative
to each drug's WHO tier 1+2 genes. The training background is primary; the
all-zero (`wt`) background was run for two cells as a check.

> Generated — edit `scripts/build_shap_v3_notebook.py`, not the `.ipynb`.
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
sys.path.insert(0, str(PROJECT / "results" / "experiments" / "locusfusion_v3"))
import results_helpers as RH
import paper_style as PS
from paper_style import MODEL, MODALITY, ramp
from results_helpers import tidy
from compare import read_cells
PS.apply(RH)
N = PS.NEUTRAL
LF_C, LASSO_C = MODEL["locusfusion"], MODEL["lasso"]
mpl.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "axes.titlesize": 9,
                     "pdf.fonttype": 42})
W = 6.5

D = PROJECT / "results" / "analysis" / "shap_locusfusion_v3"
FIG = PROJECT / "results" / "figures" / "shap_v3"
FIG.mkdir(parents=True, exist_ok=True)
def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.show()

mod = pd.read_csv(D / "modality_shares.csv")
loc = pd.read_csv(D / "locus_shares.csv")
top = pd.read_csv(D / "top_features.csv")
gate = pd.read_csv(D / "gate.csv")
cat = pd.read_csv(D / "who" / "shap_category_shares.csv")
side = pd.read_csv(D / "who" / "side_by_side.csv")
lcat = pd.read_csv(D / "who" / "locus_categories.csv")

CELLS = ["sd19_dna_unk", "sd19_dna_protein_unk", "sd19_dna_biophysical_unk",
         "sd19_dna_regulatory_unk", "sd19_all_modalities_unk"]
CNAME = {"sd19_dna_unk": "DNA", "sd19_dna_protein_unk": "DNA + protein",
         "sd19_dna_biophysical_unk": "DNA + biophysical",
         "sd19_dna_regulatory_unk": "DNA + regulatory",
         "sd19_all_modalities_unk": "all modalities"}
DRUGS = sorted(mod.drug.unique())
ABBR = {"AMIKACIN": "AMK", "CAPREOMYCIN": "CAP", "ETHAMBUTOL": "EMB", "ETHIONAMIDE": "ETO",
        "ISONIAZID": "INH", "KANAMYCIN": "KAN", "LEVOFLOXACIN": "LFX", "MOXIFLOXACIN": "MFX",
        "PYRAZINAMIDE": "PZA", "RIFAMPICIN": "RIF", "STREPTOMYCIN": "STM"}
TR = mod[mod.background == "train"]

# CV AUC per drug x cell, from the v3 result JSONs
_v = read_cells("locusfusion_v3", {"sd19_": "19"}, "v3")
CV = _v.assign(cell="sd19_" + _v.modset + "_unk").set_index(["drug", "cell"]).cv
print(f"{len(mod)} attribution runs, {mod.drug.nunique()} drugs; figures -> {FIG}")
''')

# ------------------------------------------------------------------ jobs
md("""
## 1. Jobs and the reproduction gate

Every checkpoint was rebuilt and rescored before attribution; a run is refused
if its rescored test AUC differs from the reported one by more than 0.005.
`completeness_r` is the per-isolate correlation between the attribution sum and
the logit gap to the background.
""")

code(r'''
g = gate.groupby(["background", "cell"]).agg(
    runs=("gate_pass", "size"), passed=("gate_pass", "sum"),
    max_auc_diff=("auc_abs_diff", "max"), completeness_median=("completeness_r", "median"),
    completeness_min=("completeness_r", "min"),
    runs_below_0_1=("completeness_r", lambda x: int((x < 0.1).sum())))
g.index = g.index.set_levels([g.index.levels[0], [CNAME.get(c, c) for c in g.index.levels[1]]])
g.round(3)
''')

md("""
All 77 runs pass the gate with rescored AUC identical to the reported value
(55 train, 22 wt; SLURM jobs 65428040, 65428628, 65428629, no failed tasks).
Completeness is loose: median 0.35 on the training background, and 13 of 55
runs fall below 0.1, a few of them negative. Shares below say where attribution
went; they are not an exact decomposition of the score.
""")

# ------------------------------------------------------------------ modalities
md("""
## 2. Modality shares

Share of total |attribution| per input modality, mean over the 11 drugs. The
uncalled-base channels (N in DNA and regulatory, `-`/`X` in protein) are a
subset of their modality and are shown separately on the right.
""")

code(r'''
SH = ["dna", "protein", "biophysical", "regulatory"]
M = TR.groupby("cell")[[f"share_{m}" for m in SH] + ["share_uncalled"]].mean().reindex(CELLS[::-1])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True,
                             gridspec_kw=dict(width_ratios=[4, 1.1], wspace=0.16))
y = np.arange(len(M))
left = np.zeros(len(M))
for m in SH:
    v = M[f"share_{m}"].fillna(0).values
    a1.barh(y, v, left=left, height=0.62, color=MODALITY[m], edgecolor=N["surface"],
            linewidth=1.0, label=m, zorder=2)
    left += v
a1.set_xlim(0, 1); a1.set_xlabel("share of |attribution|")
a1.set_yticks(y); a1.set_yticklabels([CNAME[c] for c in M.index])
a2.barh(y, M.share_uncalled, height=0.62, color=MODALITY["uncalled"], zorder=2)
for yi, v in zip(y, M.share_uncalled):
    a2.annotate(f"{v:.3f}", (v, yi), xytext=(3, 0), textcoords="offset points",
                va="center", fontsize=7.4, color=N["ink2"])
a2.set_xlim(0, 0.15); a2.set_xticks([0.05, 0.10]); a2.set_xlabel("uncalled-base share")
for ax in (a1, a2):
    tidy(ax)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
a1.legend(loc="upper center", bbox_to_anchor=(0.62, -0.32), ncol=4, handlelength=1.1)
save(fig, "shap_v3_modality_shares")
M.round(3)
''')

md("""
Wherever biophysical is offered it takes the largest share: 0.62 with DNA, 0.50
in the all-modalities cell. Protein takes 0.54 next to DNA. Regulatory takes
0.02 even in the DNA + regulatory cell, the best cell by AUC. The uncalled-base
channels take 0.03–0.08; regulatory N is effectively zero.
""")

code(r'''
per = TR.pivot(index="drug", columns="cell", values="share_uncalled")[CELLS]
per.columns = [CNAME[c] for c in per.columns]
per.index = [ABBR[d] for d in per.index]
per.round(3)
''')

code(r'''
unk = (top[(top.background == "train") & top.uncalled]
       .groupby(["locus", "channel"]).agg(rows=("rank", "size"), drugs=("drug", "nunique"),
                                          cells=("cell", "nunique"))
       .sort_values("rows", ascending=False))
print(f"uncalled-base rows among top-30 features: {int(unk.rows.sum())} of "
      f"{int((top.background == 'train').sum())}")
unk
''')

md("""
The largest uncalled-base share is isoniazid in the DNA-only cell (0.17). Of the
1,650 top-30 feature rows, 20 are uncalled bases, half of them katG N.
""")

# ------------------------------------------------------------------ share vs gain
md("""
## 3. Share taken against AUC gained

Each point is one drug × cell: the non-DNA attribution share against the cell's
CV AUC minus the DNA-only cell's, same drug.
""")

code(r'''
rows = []
for _, r in TR[TR.cell != "sd19_dna_unk"].iterrows():
    rows.append(dict(drug=r.drug, cell=r.cell, taken=1 - r.share_dna,
                     gain=CV[(r.drug, r.cell)] - CV[(r.drug, "sd19_dna_unk")]))
X = pd.DataFrame(rows)
rho, p = stats.spearmanr(X.taken, X.gain)
COL = {"sd19_dna_protein_unk": MODALITY["protein"],
       "sd19_dna_biophysical_unk": MODALITY["biophysical"],
       "sd19_dna_regulatory_unk": MODALITY["regulatory"],
       "sd19_all_modalities_unk": N["ink2"]}
MRK = {c: ("D" if c == "sd19_all_modalities_unk" else "o") for c in COL}
fig, ax = plt.subplots(figsize=(W * 0.62, 2.8))
ax.axhline(0, color=N["axis"], lw=0.8, zorder=1)
for c, s in X.groupby("cell"):
    ax.scatter(s.taken, s.gain, s=26, color=COL[c], marker=MRK[c], edgecolor=N["surface"],
               linewidth=0.8, zorder=3, label=CNAME[c])
ax.set_xlabel("non-DNA share of |attribution|"); ax.set_ylabel("Δ CV AUC vs. DNA only")
ax.set_xlim(-0.03, 1.0)
tidy(ax, ygrid=True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
save(fig, "shap_v3_share_vs_gain")
print(f"Spearman rho {rho:+.3f}  p {p:.2f}  n {len(X)}")
X.groupby("cell")[["taken", "gain"]].mean().rename(index=CNAME).round(4)
''')

md("""
No relationship: Spearman ρ = −0.10 (p = 0.50, n = 44). Regulatory takes 2% of
the attribution and is the only addition with a positive mean gain (+0.0015);
biophysical takes 62% and gains −0.0002.
""")

# ------------------------------------------------------------------ loci
md("""
## 4. Loci against WHO tier 1+2 genes

All 19 loci are a WHO tier 1 or 2 gene for some drug, so the split is the
drug's own WHO genes against other drugs' (`who/locus_categories.csv`; mapping
notes there). Heatmaps: locus share per drug; outlined cells are the drug's own
WHO genes.
""")

code(r'''
LOCI = (loc[(loc.background == "train") & (loc.cell == "sd19_all_modalities_unk")]
        .groupby("locus").share.mean().sort_values(ascending=False).index.tolist())
own = (lcat.assign(own=lcat.category.isin(["own_t1", "own_t2"]))
       .pivot(index="drug", columns="locus", values="own").reindex(index=DRUGS, columns=LOCI))
seq = LinearSegmentedColormap.from_list("lf_seq", [N["surface"], ramp(LF_C, 3)[1], ramp(LF_C, 3)[0]])
panels = ["sd19_dna_regulatory_unk", "sd19_all_modalities_unk"]
fig, axes = plt.subplots(len(panels), 1, figsize=(W, 5.6), sharex=True)
vmax = loc[(loc.background == "train") & loc.cell.isin(panels)].share.max()
for ax, c in zip(axes, panels):
    H = (loc[(loc.background == "train") & (loc.cell == c)]
         .pivot(index="drug", columns="locus", values="share").reindex(index=DRUGS, columns=LOCI))
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
cb.set_label("locus share of |attribution|"); cb.outline.set_visible(False)
save(fig, "shap_v3_locus_heatmap")
''')

md("""
In the all-modalities cell, attribution is spread across the MDR-linked genes
(rpoB, katG, gid, rpsL, gyrA) whatever the drug; the top locus takes 0.13–0.24.
The DNA + regulatory cell is more on target: gyrA for levofloxacin (0.29), katG
for isoniazid (0.26), rpoB for rifampicin (0.23), rrs for kanamycin (0.20).
""")

code(r'''
T = cat[cat.background == "train"].copy()
uniform = own.mean(axis=1)                          # share if every locus got the same
def prof(cell):
    s = T[T.cell == cell].set_index("drug")
    return s[["own_t1", "own_t2", "other_who"]].mean(), s.own_who_null_tokens.mean()
lasso = side[(side.source == "lasso") & (side.cell == "lasso_19locus_vcf_full")]
rows = {CNAME[c]: prof(c) for c in CELLS}
rows["LASSO, 19 loci"] = (lasso[["own_t1", "own_t2", "other_who"]].mean(), np.nan)
names = list(rows)[::-1]
fig, ax = plt.subplots(figsize=(W, 2.5))
for yi, nm in enumerate(names):
    v, null = rows[nm]
    base = LASSO_C if nm.startswith("LASSO") else LF_C
    shades = ramp(base, 3)
    left = 0
    for k, colour in zip(["own_t1", "own_t2", "other_who"], shades):
        ax.barh(yi, v[k], left=left, height=0.62, color=colour, edgecolor=N["surface"],
                linewidth=1.0, zorder=2)
        left += v[k]
    if not np.isnan(null):
        ax.plot([null, null], [yi - 0.36, yi + 0.36], color=N["ink"], lw=1.4, zorder=3)
ax.axvline(uniform.mean(), color=N["muted"], lw=0.9, ls=(0, (3, 3)), zorder=1)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
ax.set_xlim(0, 1); ax.set_xlabel("share of |attribution| (LASSO: |coefficient| mass)")
tidy(ax)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
handles = ([Patch(color=c) for c in ramp(LF_C, 3)]
           + [Line2D([0], [0], color=N["ink"], lw=1.4),
              Line2D([0], [0], color=N["muted"], lw=0.9, ls=(0, (3, 3)))])
ax.legend(handles, ["own WHO tier 1", "own WHO tier 2", "other drugs' WHO genes",
                    "variant-token share on own genes", f"uniform over loci ({uniform.mean():.3f})"],
          loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=3)
save(fig, "shap_v3_who_categories")
pd.DataFrame({nm: dict(own_who=v[["own_t1", "own_t2"]].sum(), other_who=v["other_who"],
                       token_null=null) for nm, (v, null) in rows.items()}).T.round(3)
''')

md("""
v3 puts 0.16–0.23 of its attribution on the drug's own WHO genes and 0.77–0.84
on other drugs' WHO genes. That is above both nulls — uniform over loci (0.129)
and the share of variant tokens that sit on own genes (0.07–0.09, enrichment
2.0–3.7×) — but well below the 19-locus LASSO (0.34 own). The best cell
(DNA + regulatory) is the most on-target v3 cell; all modalities the least.
""")

code(r'''
tl = (T.pivot(index="drug", columns="cell", values="top_locus")[CELLS]
      .rename(columns=CNAME))
own_hit = T.assign(hit=T.top_locus_category.isin(["own_t1", "own_t2"])).groupby("cell").hit.sum()
tl.loc["own-WHO top locus"] = [f"{int(own_hit[c])}/11" for c in CELLS]
lt = lasso.set_index("drug").top_locus
tl["LASSO, 19 loci"] = list(lt.reindex(DRUGS)) + [f"{int(lasso.top_locus_category.isin(['own_t1','own_t2']).sum())}/11"]
tl
''')

md("""
The top locus is one of the drug's own WHO genes for 2–6 of 11 drugs across the
v3 cells, against 10 of 11 for the LASSO.
""")

code(r'''
best = "sd19_dna_regulatory_unk"
tf = (top[(top.background == "train") & (top.cell == best) & (top["rank"] <= 3)]
      [["drug", "rank", "locus", "modality", "codon", "genome_pos", "channel", "n_carriers",
        "carriers_R", "carriers_S", "share_of_total"]])
tf.round(4).set_index(["drug", "rank"])
''')

# ------------------------------------------------------------------ background check
md("""
## 5. Background check

Top locus per drug under the two backgrounds, for the two cells run on both.
""")

code(r'''
bc = (cat[cat.cell.isin(["sd19_dna_regulatory_unk", "sd19_all_modalities_unk"])]
      .pivot_table(index="drug", columns=["cell", "background"], values="top_locus", aggfunc="first"))
bc.columns = [f"{CNAME[c]} | {b}" for c, b in bc.columns]
bc
''')

md("""
Against the all-zero reference, gyrA is the top locus for 10 of 11 drugs in the
all-modalities cell: the reference is a genome no isolate is, so near-universal
columns where H37Rv differs from the population take attribution. The training
background does not show this and is the one reported above.
""")

# ------------------------------------------------------------------ caveats
md("""
## Caveats

- Completeness is loose (median r = 0.35; 13 of 55 runs < 0.1, some negative):
  shares are where attribution went, not an exact decomposition of the score.
- One fold (best_fold) per drug, one seed; levofloxacin explains only its 54
  test isolates.
- All 19 loci are WHO tier 1/2 genes for some drug, so "outside the catalogue"
  cannot occur at the locus level; neighbour-gene fragments inside a locus's
  ±100 bp flank are credited to the named locus.
- LASSO |coefficient| mass and SHAP share are different quantities; compare
  profiles, not absolute values.
- Read-out attention is flat (≈1/19 per locus) and is not used as an
  importance measure.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                                                  "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(cells)} cells)")

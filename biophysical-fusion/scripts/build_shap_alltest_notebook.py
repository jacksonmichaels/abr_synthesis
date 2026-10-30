"""Generate notebooks/paper/shap_shapley_v3_alltest.ipynb: SHAP (expected gradients) and exact
Shapley for the LocusFusion v3 19-locus cells, on every test isolate.

    python scripts/build_shap_alltest_notebook.py
    sbatch scripts/sbatch/build_shap_alltest_notebook.sh      # build + execute

Reads the per-cell files as they land, so it can be re-executed while the array jobs run:
  results/analysis/shap_locusfusion_v3_alltest/locusfusion_v3/locusfusion_v2/{DRUG}/  (EG, 65716236)
  results/analysis/shap_locusfusion_v3_alltest/shapley/per_cell/{DRUG}/               (Shapley, 65716237)
The 300-isolate run (results/analysis/shap_locusfusion_v3/) is read only for the
all-test vs 300-isolate comparison and the locus -> WHO category map. Figures go to
results/figures/shap_v3_alltest/ (PDF + PNG). Colours come from paper_style.py.
Replaces notebooks/shap_locusfusion_v3.ipynb and notebooks/shapley_locusfusion_v3.ipynb.

    python scripts/build_shap_alltest_notebook.py --split lineage

builds notebooks/paper/shap_shapley_lineage_alltest.ipynb instead, from the lineage-split
checkpoints (results/analysis/shap_locusfusion_lineage_alltest/, written by
scripts/sbatch/shap_alltest_lineage.sh and shapley_alltest_lineage.sh) with the
lineage-trained 19-locus LASSO as the reference line (.../who/side_by_side.csv), into
results/figures/shap_lineage_alltest/. Same plotting code; there is no 300-isolate
lineage run, so section 5 is left out, and the read-outs are left to fill in.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["random", "lineage"], default="random")
SPLIT = ap.parse_args().split
LINEAGE = SPLIT == "lineage"

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "notebooks" / "paper" / ("shap_shapley_lineage_alltest.ipynb" if LINEAGE
                               else "shap_shapley_v3_alltest.ipynb")
cells = []

PENDING = "_Read-out to write once the lineage run has landed; the numbers are in the table above._"


def ro(random_text, lineage_text=PENDING):
    """Read-out prose: the random-split text, or the lineage placeholder."""
    return lineage_text if LINEAGE else random_text


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


if LINEAGE:
    PATHS = """\
A = PROJECT / "results" / "analysis" / "shap_locusfusion_lineage_alltest"
EG_DIR = A / "lineage_split" / "locusfusion_v2"
SV_DIR = A / "shapley" / "per_cell"
R300 = None                                   # no 300-isolate run on the lineage split
WHO = A / "who"                               # lineage-trained 19-locus LASSO reference
FIG = PROJECT / "results" / "figures" / "shap_lineage_alltest\""""
else:
    PATHS = """\
A = PROJECT / "results" / "analysis" / "shap_locusfusion_v3_alltest"
EG_DIR = A / "locusfusion_v3" / "locusfusion_v2"
SV_DIR = A / "shapley" / "per_cell"
R300 = PROJECT / "results" / "analysis" / "shap_locusfusion_v3"      # 300-isolate run
WHO = R300 / "who"
FIG = PROJECT / "results" / "figures" / "shap_v3_alltest\""""


def code(src):
    src = src.replace("__PATHS__", PATHS)
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(("""
# LocusFusion v3 attribution on every lineage-split test isolate — Shapley and SHAP

Same analysis as `shap_shapley_v3_alltest.ipynb`, on the checkpoints retrained on
lineage-grouped splits (`model_weights_lineage/lineage_split/`, test split and CV
folds from `results/experiments/lineage_split/splits/`): exact Shapley (modalities
exact, loci by 32 antithetic permutations) and SHAP (expected gradients, 100-isolate
training background, 64 samples) for the five 19-locus v3 cells × 11 drugs, best CV
fold, on **all** lineage test isolates. LASSO reference: the 19-locus one-hot LASSO
trained on the same splits. Section 1 shows what has landed.
""" if LINEAGE else """
# LocusFusion v3 attribution on every test isolate — Shapley and SHAP

Exact Shapley (modalities exact, loci by 32 antithetic permutations) and SHAP
(expected gradients, 100-isolate training background, 64 samples) for the five
19-locus v3 cells × 11 drugs, best fold, on **all** test isolates — the same
isolates the LASSO is evaluated on. The notebook reads per-cell files, so cells
still running are simply missing; section 1 shows what has landed.
""") + """
> Generated — edit `scripts/build_shap_alltest_notebook.py`, not the `.ipynb`.
""")

code(r'''
import sys, json
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

__PATHS__
FIG.mkdir(parents=True, exist_ok=True)
def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.show()
def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

CELLS = ["sd19_dna_unk", "sd19_dna_protein_unk", "sd19_dna_biophysical_unk",
         "sd19_dna_regulatory_unk", "sd19_all_modalities_unk"]
CNAME = {"sd19_dna_unk": "DNA", "sd19_dna_protein_unk": "DNA + protein",
         "sd19_dna_biophysical_unk": "DNA + biophysical",
         "sd19_dna_regulatory_unk": "DNA + regulatory",
         "sd19_all_modalities_unk": "all modalities"}
SH = ["dna", "protein", "biophysical", "regulatory"]
DRUGS = ["AMIKACIN", "CAPREOMYCIN", "ETHAMBUTOL", "ETHIONAMIDE", "ISONIAZID", "KANAMYCIN",
         "LEVOFLOXACIN", "MOXIFLOXACIN", "PYRAZINAMIDE", "RIFAMPICIN", "STREPTOMYCIN"]
ABBR = {"AMIKACIN": "AMK", "CAPREOMYCIN": "CAP", "ETHAMBUTOL": "EMB", "ETHIONAMIDE": "ETO",
        "ISONIAZID": "INH", "KANAMYCIN": "KAN", "LEVOFLOXACIN": "LFX", "MOXIFLOXACIN": "MFX",
        "PYRAZINAMIDE": "PZA", "RIFAMPICIN": "RIF", "STREPTOMYCIN": "STM"}

def _cat(paths):
    paths = list(paths)
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()
def _meta(paths):
    return pd.DataFrame([json.loads(Path(p).read_text()) for p in paths])

# Shapley, all test isolates
smeta = _meta(sorted(SV_DIR.glob("*/*__meta.json")))
smod_all = _cat(sorted(SV_DIR.glob("*/*__modality.csv")))
smod = smod_all[smod_all.game == "modality"]
sloc = _cat(sorted(SV_DIR.glob("*/*__locus.csv")))
# SHAP (expected gradients), all test isolates, training background
emeta = _meta(sorted(EG_DIR.glob("*/*__train__meta.json")))
eloc = _cat(sorted(EG_DIR.glob("*/*__train__loci.csv")))
# 300-isolate run, for comparison (random split only)
if R300 is not None:
    smod300 = pd.read_csv(R300 / "shapley" / "modality_shapley.csv").query("game == 'modality'")
    sloc300 = pd.read_csv(R300 / "shapley" / "locus_shapley.csv")
    emod300 = pd.read_csv(R300 / "modality_shares.csv").query("background == 'train'")
    eloc300 = pd.read_csv(R300 / "locus_shares.csv").query("background == 'train'")
# locus -> own / other-drug WHO gene (isolate-independent); 19-locus LASSO reference
lcat = pd.read_csv(WHO / "locus_categories.csv")
side = pd.read_csv(WHO / "side_by_side.csv")
OWN = (lcat.assign(own=lcat.category.isin(["own_t1", "own_t2"]))
       .pivot(index="drug", columns="locus", values="own"))
UNIFORM = OWN.mean(axis=1).mean()
LASSO = side[(side.source == "lasso") & (side.cell == "lasso_19locus_onehot_top40")].set_index("drug")

def own_share(loc):
    """Share of attribution on each drug's own WHO tier 1/2 genes, per drug x cell."""
    d = loc.assign(own=[bool(OWN.loc[a, b]) for a, b in zip(loc.drug, loc.locus)])
    return d[d.own].groupby(["drug", "cell"]).share.sum().rename("own_who")

EMOD = (emeta.set_index(["drug", "cell"])[[f"share_{m}" for m in SH]]
        .rename(columns=lambda c: c.replace("share_", "")) if len(emeta) else pd.DataFrame())
SMOD = smod.pivot_table(index=["drug", "cell"], columns="player", values="share").reindex(columns=SH)
print(f"Shapley: {len(smeta)}/55 drug x cell runs; SHAP: {len(emeta)}/55; figures -> {FIG}")
''')

# ------------------------------------------------------------------ progress
md("""
## 1. What has landed

One row per drug: which cells have results. `S` = Shapley, `E` = SHAP; `·` = not yet.
""")

code(r'''
def have(meta):
    return set(zip(meta.drug, meta.cell)) if len(meta) else set()
hs, he = have(smeta), have(emeta)
grid = pd.DataFrame([[("S" if (d, c) in hs else "·") + ("E" if (d, c) in he else "·")
                      for c in CELLS] for d in DRUGS],
                    index=[ABBR[d] for d in DRUGS], columns=[CNAME[c] for c in CELLS])
grid.loc["done (S / E)"] = [f"{sum((d, c) in hs for d in DRUGS)} / {sum((d, c) in he for d in DRUGS)}"
                            for c in CELLS]
grid
''')

md("""
Checks per run. Shapley: rescored test AUC against the reported one, worst
efficiency error (Σφ against f(isolate) − f(all H37Rv)). SHAP: the same AUC gate,
and completeness r (per-isolate correlation of Σattribution with the logit gap
to the background).
""")

code(r'''
chk = pd.DataFrame({
    "Shapley runs": smeta.groupby("cell").size(),
    "Shapley isolates": smeta.groupby("cell").n_explained.sum(),
    "Shapley max |AUC diff|": smeta.groupby("cell").auc_abs_diff.max(),
    "Shapley max eff. err": smeta.groupby("cell").modality_max_eff_err.max(),
})
if len(emeta):
    chk = chk.join(pd.DataFrame({
        "SHAP runs": emeta.groupby("cell").size(),
        "SHAP gate pass": emeta.groupby("cell").gate_pass.sum(),
        "SHAP completeness r (median)": emeta.groupby("cell").completeness_r.median(),
        "SHAP runs r < 0.1": emeta.groupby("cell").completeness_r.apply(lambda x: int((x < 0.1).sum())),
    }))
chk.reindex(CELLS).rename(index=CNAME).round(5)
''')

md(ro("""
Shapley is complete: 55/55 runs on 17,433 explained isolates per cell, rescored
AUC within 1e-4 of the reported value, worst efficiency error below 1e-4 logit.
SHAP runs that have landed all pass the AUC gate; completeness stays loose
(median r 0.14–0.47 by cell), as on 300 isolates.
"""))

# ------------------------------------------------------------------ Shapley modalities
md("""
## 2. Shapley modality shares

Share of mean |φ| per modality, mean over drugs. Right: the uncalled-base
channels (N in DNA and regulatory, `-`/`X` in protein) split off as their own players.
""")

code(r'''
M = SMOD.groupby("cell").mean().reindex(CELLS[::-1]).fillna(0)
unc = (smod_all[(smod_all.game == "called_uncalled") & smod_all.player.str.endswith(":uncalled")]
       .groupby(["drug", "cell"]).share.sum().groupby("cell").mean().reindex(M.index))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True,
                             gridspec_kw=dict(width_ratios=[4, 1.1], wspace=0.16))
y = np.arange(len(M)); left = np.zeros(len(M))
for m in SH:
    a1.barh(y, M[m], left=left, height=0.62, color=MODALITY[m], edgecolor=N["surface"],
            linewidth=1.0, label=m, zorder=2)
    left += M[m].values
a1.set_xlim(0, 1); a1.set_xlabel("share of mean |φ|")
a1.set_yticks(y); a1.set_yticklabels([CNAME[c] for c in M.index])
a2.barh(y, unc, height=0.62, color=MODALITY["uncalled"], zorder=2)
for yi, v in zip(y, unc):
    a2.annotate(f"{v:.3f}", (v, yi), xytext=(3, 0), textcoords="offset points",
                va="center", fontsize=7.4, color=N["ink2"])
a2.set_xlim(0, 0.15); a2.set_xticks([0.05, 0.10]); a2.set_xlabel("uncalled-base share")
for ax in (a1, a2):
    tidy(ax); clean(ax)
a1.legend(loc="upper center", bbox_to_anchor=(0.62, -0.32), ncol=4, handlelength=1.1)
save(fig, "shapley_alltest_modality_shares")
M.assign(uncalled=unc).round(3).rename(index=CNAME)
''')

md(ro("""
Unchanged from the 300-isolate run: DNA takes 0.80–1.00 in every cell; in the
all-modalities cell protein 0.09, regulatory 0.05, biophysical 0.04. Uncalled
bases take 0.05–0.07.
"""))

# ------------------------------------------------------------------ Shapley loci
md("""
## 3. Shapley locus shares

Locus share of mean |φ| per drug, DNA + regulatory and all-modalities cells.
Outlined: the drug's own WHO tier 1/2 genes.
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
save(fig, "shapley_alltest_locus_heatmap")
top3 = (sloc[sloc.cell.isin(panels) & (sloc["rank"] <= 3)]
        .assign(item=lambda d: d.locus + " " + d.share.map(lambda v: f"{v:.2f}"))
        .groupby(["drug", "cell"]).item.apply(", ".join).unstack()[panels].rename(columns=CNAME))
top3.index = [ABBR[d] for d in top3.index]
top3
''')

md(ro("""
The drug's own WHO gene is the top locus for 10 of 11 drugs in each cell: rpoB
for rifampicin (0.53 all modalities), rrs for kanamycin (0.52) and capreomycin
(0.40), katG for isoniazid (0.37), gyrA for the fluoroquinolones. Levofloxacin's
tlyA (0.25, all modalities) is the H37Rv-reference offset seen on 300 isolates.
"""))

# ------------------------------------------------------------------ own WHO
md("""
## 4. Share on each drug's own WHO genes

Mean over drugs of the attribution share on the drug's own WHO tier 1/2 genes, per
cell, Shapley and SHAP; SHAP is averaged over the drugs that have landed. Dashed:
the 19-locus one-hot LASSO's share of its 40 largest |coefficients| on own genes
(as in the paper's Fig. A1); dotted: an equal share per locus.
""")

code(r'''
s_own = own_share(sloc)
e_own = own_share(eloc) if len(eloc) else pd.Series(dtype=float)
O = pd.DataFrame({"Shapley": s_own.groupby("cell").mean(),
                  "SHAP": e_own.groupby("cell").mean() if len(e_own) else np.nan,
                  "SHAP drugs": e_own.groupby("cell").size() if len(e_own) else 0}).reindex(CELLS)
lasso_own = LASSO.own_who.mean()
fig, ax = plt.subplots(figsize=(W * 0.75, 2.3))
yy = np.arange(len(CELLS))[::-1]
for yi, c in zip(yy, CELLS):
    if not np.isnan(O.loc[c, "SHAP"]):
        ax.plot([O.loc[c, "SHAP"], O.loc[c, "Shapley"]], [yi, yi], color=N["axis"], lw=1.4, zorder=1)
ax.scatter(O.SHAP, yy, s=34, color=N["surface"], edgecolor=LF_C, linewidth=1.6, zorder=3)
ax.scatter(O.Shapley, yy, s=34, color=LF_C, edgecolor=N["surface"], linewidth=0.8, zorder=3)
ax.axvline(lasso_own, color=LASSO_C, lw=1.1, ls=(0, (3, 3)), zorder=0)
ax.axvline(UNIFORM, color=N["muted"], lw=0.9, ls=(0, (1, 2)), zorder=0)
ax.set_yticks(yy); ax.set_yticklabels([CNAME[c] for c in CELLS])
ax.set_xlim(0, 0.6); ax.set_xlabel("share on own WHO genes")
tidy(ax); clean(ax)
ax.legend([Line2D([0], [0], marker="o", lw=0, markersize=6, markerfacecolor=N["surface"],
                  markeredgecolor=LF_C, markeredgewidth=1.4),
           Line2D([0], [0], marker="o", lw=0, markersize=6, color=LF_C),
           Line2D([0], [0], color=LASSO_C, lw=1.1, ls=(0, (3, 3))),
           Line2D([0], [0], color=N["muted"], lw=0.9, ls=(0, (1, 2)))],
          ["LocusFusion, SHAP", "LocusFusion, Shapley", f"LASSO, 19 loci, top 40 ({lasso_own:.2f})",
           f"equal share per locus ({UNIFORM:.2f})"],
          loc="upper left", bbox_to_anchor=(1.01, 1.0))
save(fig, "alltest_own_who_share")
top_own = (sloc[sloc["rank"] == 1].assign(hit=lambda d: [bool(OWN.loc[a, b]) for a, b in zip(d.drug, d.locus)])
           .groupby("cell").hit.agg(lambda x: f"{int(x.sum())}/{len(x)}"))
O.assign(**{"Shapley top locus own": top_own}).rename(index=CNAME).round(3)
''')

md(ro("""
Shapley puts 0.375–0.387 on own WHO genes in every cell, three times an equal
split (0.13) and a little below the 19-locus LASSO's 40 largest coefficients
(0.42). Both have the drug's own gene on top for 10 of 11 drugs. SHAP (8–9 drugs so far)
gives 0.12–0.20, close to the equal split in the biophysical and all-modalities
cells.
"""))

# ------------------------------------------------------------------ vs 300
_n5 = len(cells)
md("""
## 5. All test isolates against the 300-isolate run

Each point is one drug × cell × locus (left) or drug × cell × modality (right):
the share on all test isolates against the share on the earlier 300 explained
isolates, same checkpoint. Drugs with ≤ 300 test isolates are identical by
construction and are left out.
""")

code(r'''
n_test = smeta.set_index(["drug", "cell"]).n_test
big = set(n_test[n_test > 300].index)
def paired(new, old, key, val="share"):
    j = new.merge(old, on=["drug", "cell", key], suffixes=("_all", "_300"))[["drug", "cell", key, f"{val}_all", f"{val}_300"]]
    return j[[(d, c) in big for d, c in zip(j.drug, j.cell)]]
rows = []
for name, new_l, old_l in (("Shapley", sloc, sloc300), ("SHAP", eloc, eloc300)):
    if len(new_l):
        rows.append((name, "locus", paired(new_l, old_l, "locus")))
rows.append(("Shapley", "modality", paired(smod, smod300, "player")))
if len(emeta):
    e_long = EMOD.reset_index().melt(id_vars=["drug", "cell"], var_name="player", value_name="share").dropna()
    o_long = emod300.melt(id_vars=["drug", "cell"], value_vars=[f"share_{m}" for m in SH],
                          var_name="player", value_name="share").dropna()
    o_long["player"] = o_long.player.str.replace("share_", "")
    rows.append(("SHAP", "modality", paired(e_long, o_long, "player")))
fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw=dict(wspace=0.35))
style = {"Shapley": dict(color=LF_C, edgecolor=N["surface"], linewidth=0.5),
         "SHAP": dict(color=N["surface"], edgecolor=LF_C, linewidth=1.0)}
summ = []
for ax, level in zip(axes, ("locus", "modality")):
    lim = 0
    for name, lv, j in rows:
        if lv != level or not len(j):
            continue
        ax.scatter(j.share_300, j.share_all, s=12, zorder=3, label=name, **style[name])
        lim = max(lim, j.share_300.max(), j.share_all.max())
        r, _ = stats.pearsonr(j.share_300, j.share_all)
        summ.append(dict(method=name, level=level, n=len(j), pearson_r=r,
                         max_abs_diff=(j.share_all - j.share_300).abs().max()))
    lim *= 1.05
    ax.plot([0, lim], [0, lim], color=N["axis"], lw=0.9, zorder=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel(f"{level} share, 300 isolates"); ax.set_ylabel(f"{level} share, all test isolates")
    tidy(ax, ygrid=True); clean(ax)
axes[0].legend(loc="upper left")
save(fig, "alltest_vs_300")
pd.DataFrame(summ).round(4)
''')

md("""
Going from 300 to all test isolates changes little: Pearson r ≥ 0.997 for loci
and ≥ 0.9999 for modalities, worst single change 0.034 (a Shapley locus share).
The 300-isolate conclusions stand.
""")

if LINEAGE:
    del cells[_n5:]                 # no 300-isolate run on the lineage split

# ------------------------------------------------------------------ SHAP vs Shapley
md("""
## 6. SHAP against Shapley, modality shares

Open circles: SHAP; filled: Shapley; mean over the drugs that have both.
""")

code(r'''
both = EMOD.index.intersection(SMOD.index) if len(EMOD) else []
ME = EMOD.loc[both].groupby("cell").mean() if len(both) else pd.DataFrame(columns=SH)
MS = SMOD.loc[both].groupby("cell").mean() if len(both) else pd.DataFrame(columns=SH)
pairs = [(c, m) for c in CELLS[::-1] if c in MS.index for m in SH if not np.isnan(MS.loc[c, m])]
fig, ax = plt.subplots(figsize=(W, 0.21 * len(pairs) + 0.9))
y = np.arange(len(pairs))[::-1]
for yi, (c, m) in zip(y, pairs):
    e, s = ME.loc[c, m], MS.loc[c, m]
    ax.plot([e, s], [yi, yi], color=N["axis"], lw=1.4, zorder=1)
    ax.scatter([e], [yi], s=34, color=N["surface"], edgecolor=MODALITY[m], linewidth=1.6, zorder=3)
    ax.scatter([s], [yi], s=34, color=MODALITY[m], edgecolor=N["surface"], linewidth=0.8, zorder=3)
ax.set_yticks(y); ax.set_yticklabels([f"{CNAME[c]} · {m}" for c, m in pairs], fontsize=7.4)
for b in [i for i in range(1, len(pairs)) if pairs[i][0] != pairs[i - 1][0]]:
    ax.axhline(y[b] + 0.5, color=N["grid"], lw=0.8, zorder=0)
ax.set_xlim(-0.02, 1.02); ax.set_xlabel("modality share")
tidy(ax); clean(ax)
ax.legend([Line2D([0], [0], marker="o", lw=0, markersize=6, markerfacecolor=N["surface"],
                  markeredgecolor=N["ink2"], markeredgewidth=1.4),
           Line2D([0], [0], marker="o", lw=0, markersize=6, color=N["ink2"])],
          ["LocusFusion, SHAP", "LocusFusion, Shapley"], loc="upper center",
          bbox_to_anchor=(0.5, -0.12), ncol=2)
save(fig, "alltest_shap_vs_shapley_modalities")
T = pd.concat({"SHAP": ME, "Shapley": MS}, axis=1).swaplevel(axis=1).sort_index(axis=1)
T[("drugs", "")] = pd.Series(both.get_level_values("cell")).value_counts() if len(both) else 0
T.reindex([c for c in CELLS if c in MS.index]).rename(index=CNAME).round(3)
''')

md(ro("""
Same disagreement as on 300 isolates: SHAP gives biophysical 0.52–0.62 and
protein 0.20–0.56 where Shapley gives 0.05–0.10; Shapley gives DNA 0.80–0.93.
Regulatory is the one modality Shapley credits more (0.06–0.08 vs 0.01–0.02).
"""))

md("""
## Caveats

- Shapley's "absent" is the H37Rv reference, out of distribution for a model
  trained with that input: a value means what the model loses without a player.
  Variants every isolate carries get credited as constant offsets.
- SHAP's sums track the prediction only loosely (completeness, section 1), because
  tokens are built inside the forward pass and appear or vanish along the
  interpolation path; use its shares as a map, not a decomposition.
- Locus Shapley uses 32 permutations (or exact enumeration when ≤ 9 loci differ).
- One fold (best_fold) per drug, one seed.
- LASSO |coefficient| mass and attribution shares are different quantities;
  compare profiles, not absolute values.
- SHAP numbers are provisional until all 55 runs have landed (section 1).
""" + ("""- Lineage split: levofloxacin has 59 test isolates, so its shares are the
  noisiest; the best CV fold can be fold 4, the Beijing-only fold, whose
  checkpoint was trained without Beijing isolates.
""" if LINEAGE else ""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                                                  "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(cells)} cells)")

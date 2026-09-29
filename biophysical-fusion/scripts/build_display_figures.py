"""Generate notebooks/display_figures.ipynb — the paper's figures, one cell each.

    python scripts/build_display_figures.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/display_figures.ipynb

Every figure is saved to results/figures/paper/ as it is drawn. The story and
numbers these figures carry are in PAPER.md; the significance family is
results_helpers.paper_family(), shared with results.ipynb §9.

Conventions: titles name the subject and slice, never the finding; nothing
explanatory is drawn inside a chart; colour follows model identity (blue =
dense-sequence CNNs, rust = LocusFusion, light rust = locusfusion_v2, orange =
variant-token aggregators, aqua = LASSO / non-network baselines).
"""
import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("-o", "--output", type=Path, default=None,
                 help="notebook path (default: <project>/notebooks/display_figures.ipynb)")
_args = _ap.parse_args()
OUT = (_args.output.expanduser().resolve() if _args.output
       else PROJECT / "notebooks" / "display_figures.ipynb")

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.rstrip("\n").splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md("""
# Paper figures

One cell per figure, in the paper's order. Each is saved to
`results/figures/paper/` when it is drawn. Numbers and framing: `PAPER.md`.
Single-drug cells, 5-fold CV AUC, macro over the 11 shared drugs; networks at
`all_modalities` and the LASSO at `dna` unless a figure says otherwise.

> Generated — edit `scripts/build_display_figures.py`, not the `.ipynb`.
""")

code('''
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

_here = Path.cwd()
PROJECT = next(p for p in [_here, *_here.parents] if (p / "bigtb_ref.py").exists())
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from results_frame import load_frame, STD_MODSETS
from bigtb_baselines import SD_BASE_CV, MD_BASE_CV, ALL_DRUGS
from results_helpers import (SURFACE, INK, INK2, MUTED, AXIS, GRID, BLUE, ORANGE, RUST,
                             RUST_LIGHT, AQUA, AQUA_RAMP, PALE_BLUE, tidy, label_points,
                             paper_family, plot_sig_by_claim)

frame, meta = load_frame()
# the old-tokenizer locusfusion rows are a different model: kept apart, named
_old = ((frame.model == "locusfusion") & (frame.run == "newmodels_full")
        & (frame.task != "multi-drug"))
frame_lf_old = frame[_old].assign(model="locusfusion_old")
frame = frame[~_old].reset_index(drop=True)

PD, L19 = "single-drug", "single-drug, all loci"
A, MODS5 = "all_modalities", STD_MODSETS
EST3 = ["mdcnn", "cisfusion", "late_fusion"]
V1, V2 = "locusfusion", "locusfusion_v2"
AGGREGATORS = ["catalogue", "additive", "noisyor", "gatedpool", "deepsets", "fm"]
BASELINES = {"sparse_baseline", "sparse_baseline_trainvocab",
             "lasso_onehot", "lasso_delta", "lasso_genome"}
LABEL = {"lasso_onehot": "LASSO", "mdcnn": "MD-CNN (ours)", "cisfusion": "cis-fusion",
         "late_fusion": "late-fusion", V1: "LocusFusion", V2: "LocusFusion v2"}
OUTDIR = PROJECT / "results" / "figures" / "paper"
OUTDIR.mkdir(parents=True, exist_ok=True)


def hue(model):
    if model in BASELINES:
        return AQUA
    if model == V1:
        return RUST
    if model == V2:
        return RUST_LIGHT
    return ORANGE if model in AGGREGATORS else BLUE


def drug_cv(model, task, modset):
    s = frame[(frame.model == model) & (frame.task == task) & (frame.modset == modset)]
    return s.groupby("drug").cv.mean().reindex(ALL_DRUGS)


def cell(model, task):
    """The pinned modality set: dna for the LASSO, all_modalities for networks."""
    return drug_cv(model, task, "dna" if model in BASELINES else A)


def save(fig, name):
    fig.savefig(OUTDIR / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUTDIR / f"{name}.svg", bbox_inches="tight")
    plt.show()


N_BY_DRUG = frame[frame.task == PD].groupby("drug").n_valid.max().reindex(ALL_DRUGS)
BY_N = N_BY_DRUG.sort_values().index.tolist()          # smallest cohort first
SD_REF, MD_REF = SD_BASE_CV.mean(), MD_BASE_CV.mean()
print(f"SD-CNN {SD_REF:.4f}   MD-CNN {MD_REF:.4f}   figures -> {OUTDIR}")
''')

# ------------------------------------------------------------------ figure 0
md("""
## Figure 0 — LocusFusion

Schematic of the paper's model (the fixed tokenizer). Drawn by
`scripts/locusfusion_schematic.py`, not by this notebook; rerun that script to
change it.
""")

code('''
from IPython.display import Image, display
display(Image(filename=str(OUTDIR / "fig0_locusfusion_architecture.png"), width=1100))
''')

# ------------------------------------------------------------------ figure 1
md("""
## Figure 1 — models at two locus sets

Macro CV AUC for each model on the drug's own 2–3 loci and on all 19 curated
loci. Reference lines are the published baselines' CV AUC.
""")

code('''
MODELS = ["lasso_onehot", "late_fusion", "cisfusion", "mdcnn", V1]
m = pd.DataFrame({mo: {"per-drug": cell(mo, PD).mean(), "19": cell(mo, L19).mean()}
                  for mo in MODELS}).T.sort_values("19")
fig, ax = plt.subplots(figsize=(8.0, 3.6))
y = np.arange(len(m))
for yi, (mo, r) in zip(y, m.iterrows()):
    ax.plot([r["per-drug"], r["19"]], [yi, yi], color=AXIS, lw=2.0, zorder=1,
            solid_capstyle="round")
    ax.scatter([r["per-drug"]], [yi], s=70, zorder=3, color=SURFACE, edgecolor=hue(mo),
               linewidth=2.0)
    ax.scatter([r["19"]], [yi], s=70, zorder=3, color=hue(mo), edgecolor=SURFACE,
               linewidth=1.4)
    ax.annotate(f"{r['19']:.4f}", (r["19"], yi), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=7.8, color=INK2)
for x, lab in ((SD_REF, f"SD-CNN (published) {SD_REF:.4f}"),
               (MD_REF, f"MD-CNN (published) {MD_REF:.4f}")):
    ax.axvline(x, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=0)
    ax.annotate(lab, (x, len(m) - 0.45), xytext=(3, 0), textcoords="offset points",
                fontsize=7.4, color=MUTED, va="bottom", rotation=0)
ax.set_yticks(y); ax.set_yticklabels([LABEL[i] for i in m.index], fontsize=9)
ax.set_ylim(-0.6, len(m) + 0.1)
ax.set_xlabel("macro CV AUC, 11 drugs")
ax.set_title("Macro CV AUC by model, per-drug loci vs. 19 loci", loc="left")
ax.legend([Line2D([0], [0], marker="o", lw=0, markersize=8, markerfacecolor=SURFACE,
                  markeredgecolor=INK2, markeredgewidth=2),
           Line2D([0], [0], marker="o", lw=0, markersize=8, color=INK2)],
          ["per-drug loci (2–3 genes)", "19 loci"], loc="upper center",
          bbox_to_anchor=(0.5, -0.2), ncol=2)
tidy(ax); fig.tight_layout(); save(fig, "fig1_models_by_locus_set")
print(m.round(4).to_string())
''')

md("""
At 19 loci the LASSO (0.9241) sits with MD-CNN (ours, 0.9246) and the published
MD-CNN; on per-drug loci it is the lowest model. LocusFusion is highest at 19
loci (0.9343). Significance for each pair is Figure 6.
""")

# ------------------------------------------------------------------ figure 2
md("""
## Figure 2 — LocusFusion against the LASSO and MD-CNN, per drug

Per-drug CV AUC difference at 19 loci, drugs ordered by labelled cohort size.
""")

code('''
lf = cell(V1, L19)
d = pd.DataFrame({"LASSO": lf - cell("lasso_onehot", L19),
                  "MD-CNN (ours)": lf - cell("mdcnn", L19)}).reindex(BY_N)
fig, ax = plt.subplots(figsize=(7.6, 4.4))
y = np.arange(len(d))
ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
for col, colour, dy in (("LASSO", AQUA, 0.13), ("MD-CNN (ours)", BLUE, -0.13)):
    ax.scatter(d[col], y + dy, s=54, color=colour, edgecolor=SURFACE, linewidth=1.3,
               zorder=3, label=f"LocusFusion − {col}  ({int((d[col] > 0).sum())}/11 drugs > 0)")
ax.set_yticks(y)
ax.set_yticklabels([f"{dr.title()} (n = {int(N_BY_DRUG[dr]):,})" for dr in d.index], fontsize=8)
ax.set_xlabel("Δ CV AUC, 19 loci")
ax.set_title("LocusFusion − LASSO and − MD-CNN (ours), per drug, 19 loci", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=7.8)
tidy(ax); fig.tight_layout(); save(fig, "fig2_locusfusion_per_drug")
print(d.round(4).to_string())
''')

md("""
Ahead of both on 10 of 11 drugs; the largest and least stable differences sit on
the small cohorts at the top.
""")

# ------------------------------------------------------------------ figure 3
md("""
## Figure 3 — what the extra modalities add, per drug

All four modalities minus DNA only, averaged over the three dense-sequence CNNs
(MD-CNN, cis-fusion, late-fusion), at both locus sets.
""")

code('''
def mod_gain(task):
    return pd.concat([drug_cv(mo, task, A) - drug_cv(mo, task, "dna") for mo in EST3],
                     axis=1).mean(axis=1)
g = pd.DataFrame({"per-drug loci": mod_gain(PD), "19 loci": mod_gain(L19)}) \\
      .sort_values("per-drug loci")
fig, ax = plt.subplots(figsize=(7.6, 4.4))
y = np.arange(len(g))
ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
for yi, dr in zip(y, g.index):
    ax.plot(g.loc[dr].values, [yi, yi], color=GRID, lw=2.0, zorder=1)
ax.scatter(g["per-drug loci"], y, s=60, color=BLUE, edgecolor=SURFACE, linewidth=1.3,
           zorder=3, label=f"per-drug loci  (macro {g['per-drug loci'].mean():+.4f})")
ax.scatter(g["19 loci"], y, s=60, color=SURFACE, edgecolor=BLUE, linewidth=1.8,
           zorder=3, label=f"19 loci  (macro {g['19 loci'].mean():+.4f})")
ax.set_yticks(y); ax.set_yticklabels([dr.title() for dr in g.index], fontsize=8.2)
ax.set_xlabel("Δ CV AUC, all modalities − DNA")
ax.set_title("All modalities − DNA by drug, CNN mean, two locus sets", loc="left")
ax.legend(loc="lower right", fontsize=7.8)
tidy(ax); fig.tight_layout(); save(fig, "fig3_modalities_per_drug")
print(g.round(4).to_string())
''')

md("""
At per-drug loci the gain is concentrated in ethionamide and isoniazid (the
*fabG1–inhA* promoter, via the regulatory modality) and pyrazinamide; at 19 loci
it largely disappears. Per-drug effects rest on five folds each and are sizes,
not tested differences.
""")

# ------------------------------------------------------------------ figure 4
md("""
## Figure 4 — where the LASSO puts its weight

Left: the 19-locus LASSO's 40 largest coefficients, split by whether the locus is
in that drug's own per-drug gene map. Right: the whole-genome LASSO's full
coefficient mass, split by WHO catalogue (Table 21, tier 1 + 2) membership.
""")

code('''
LW = PROJECT / "results/experiments/lasso_wholeseq_20260901"
own_share = {}
for dr in ALL_DRUGS:
    top = json.load(open(next((LW / "onehot_all").glob(f"{dr}__*.json"))))["top_features"]
    own = set(json.load(open(next((LW / "onehot_perdrug").glob(f"{dr}__*.json"))))["genes"])
    tot = sum(abs(t["coef"]) for t in top)
    own_share[dr] = 100 * sum(abs(t["coef"]) for t in top
                              if t["feature"].split(":")[0] in own) / tot
l19 = pd.DataFrame({"drug's own loci": pd.Series(own_share)})
l19["other drugs' loci"] = 100 - l19["drug's own loci"]
who = (pd.read_csv(PROJECT / "results/experiments/lasso_wholegenome_20260903/who_mass.csv")
       .set_index("drug")
       .rename(columns={"this drug %": "drug's own WHO genes",
                        "other drug %": "other drugs' WHO genes",
                        "outside %": "outside the WHO catalogue"}))
who = who[["drug's own WHO genes", "other drugs' WHO genes", "outside the WHO catalogue"]]
order = l19.sort_values("drug's own loci").index

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3), sharey=True)
y = np.arange(len(order))
for ax, tab, cols, title in (
        (a1, l19, AQUA_RAMP[:2], "19-locus LASSO, top-40 |coefficient|"),
        (a2, who, AQUA_RAMP, "Whole-genome LASSO, all |coefficient|")):
    left = np.zeros(len(order))
    for c, colour in zip(tab.columns, cols):
        v = tab.loc[order, c].values
        ax.barh(y, v, left=left, height=0.66, color=colour, edgecolor=SURFACE,
                linewidth=1.0, zorder=2, label=c)
        left += v
    ax.set_xlim(0, 100); ax.set_xlabel("share of |coefficient| mass (%)")
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(tab.columns),
              fontsize=7.6)
    tidy(ax)
a1.set_yticks(y); a1.set_yticklabels([dr.title() for dr in order], fontsize=8.2)
fig.suptitle("LASSO coefficient mass by locus membership, per drug", x=0.01, ha="left",
             fontsize=11, fontweight="bold")
fig.tight_layout(); save(fig, "fig4_lasso_coefficient_mass")
print(pd.concat([l19, who], axis=1).loc[order].round(1).to_string())
''')

md("""
At 19 loci, 26–52% of the LASSO's largest coefficients sit on the drug's own
genes; the rest are on other drugs' targets. Genome-wide, 62% of the mass
(macro over drugs) is outside the WHO catalogue altogether. Coefficient mass is
not a variance decomposition, and none of this separates co-resistance from
lineage; it is why the locus gain is read as probably not causal.
""")

# ------------------------------------------------------------------ figure 5
md("""
## Figure 5 — parameter count and accuracy

Every complete network cell at 19 loci (five modality sets per model), against
its parameter count. The LASSO's macro is a reference line.
""")

code('''
cells19 = (frame[(frame.task == L19) & ~frame.model.isin(BASELINES | {V2})
                 & frame.n_params.notna()]
           .groupby(["model", "modset"])
           .agg(cv=("cv", "mean"), n=("drug", "nunique"), params=("n_params", "mean"))
           .reset_index().query("n == 11"))
rho, p = stats.spearmanr(cells19.params, cells19.cv)
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.scatter(cells19.params, cells19.cv, s=46, color=[hue(mo) for mo in cells19.model],
           edgecolor=SURFACE, linewidth=1.2, zorder=3)
lasso19 = cell("lasso_onehot", L19).mean()
ax.axhline(lasso19, color=AQUA, lw=1.2, ls=(0, (4, 3)), zorder=1)
ax.annotate(f"LASSO {lasso19:.4f}", (cells19.params.min() / 2.4, lasso19), va="top",
            xytext=(0, -4), textcoords="offset points", fontsize=7.6, color=INK2)
ax.set_xscale("log")
ax.set_xlim(cells19.params.min() / 2.5, cells19.params.max() * 2.5)
ax.set_ylim(cells19.cv.min() - 0.006, cells19.cv.max() + 0.010)
lab = cells19.sort_values("cv").groupby("model").tail(1)       # one label per model
label_points(ax, lab.params, lab.cv, [LABEL.get(mo, mo) for mo in lab.model], fontsize=7.4)
ax.set_xlabel("parameters (log scale)"); ax.set_ylabel("macro CV AUC")
ax.set_title("Parameter count vs. macro CV AUC, network cells, 19 loci", loc="left")
ax.annotate(f"Spearman ρ = {rho:+.2f}, p = {p:.2f}, n = {len(cells19)} cells",
            (0.0, -0.2), xycoords="axes fraction", fontsize=8, color=INK2)
ax.legend([Line2D([0], [0], marker="o", lw=0, markersize=7, color=c)
           for c in (BLUE, ORANGE, RUST)],
          ["dense-sequence CNNs", "variant-token aggregators", "LocusFusion"],
          loc="lower right", fontsize=7.8)
tidy(ax, ygrid=True); fig.tight_layout(); save(fig, "fig5_params_vs_auc")
''')

md("""
No relationship across 0.02M–59M parameters. The two largest CNNs are the same
size and 0.012 apart; the best model is 0.66M.
""")

# ------------------------------------------------------------------ figure 6
md("""
## Figure 6 — paired comparisons by claim

Hodges–Lehmann Δ and exact 95% CI over the 55 fold-paired differences; filled =
significant after Holm over the paper's 14 comparisons (`SIGNIFICANCE_TESTS.md`).
Three tested rows are not drawn: the tokenizer fix, onehot − delta, and whole
genome − 19 loci.
""")

code('''
sig = paper_family(pd.concat([frame, frame_lf_old], ignore_index=True), ALL_DRUGS, MODS5)
fig = plot_sig_by_claim(sig); save(fig, "fig6_significance_by_claim")
sig.to_csv(OUTDIR / "sig_paper_family.csv", index=False)
print(sig.verdict.value_counts().to_string())
''')

# ------------------------------------------------------------------ appendix
md("""
## Appendix figure — LocusFusion v2 against LocusFusion

Macro CV AUC per modality set at per-drug loci. Only cells where both models
have all 11 drugs are drawn.
""")

code('''
rows = []
for ms in MODS5:
    a, b = drug_cv(V1, PD, ms), drug_cv(V2, PD, ms)          # v2 ahead = drugs, CV means
    if a.notna().all() and b.notna().all():
        rows.append((ms, a.mean(), b.mean(), int((b > a).sum())))
v = pd.DataFrame(rows, columns=["modset", V1, V2, "v2 ahead"]).set_index("modset")
fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(v) + 1.3))
y = np.arange(len(v))[::-1]
for yi, (ms, r) in zip(y, v.iterrows()):
    ax.plot([r[V1], r[V2]], [yi + 0.12, yi - 0.12], color=AXIS, lw=1.6, zorder=1)
for mo, dy in ((V1, 0.12), (V2, -0.12)):
    ax.scatter(v[mo], y + dy, s=66, color=hue(mo), edgecolor=SURFACE, linewidth=1.4,
               zorder=3, label=LABEL[mo])
ax.set_yticks(y)
ax.set_yticklabels([f"{ms.replace('_', ' ')}  ({k}/11)" for ms, k in
                    zip(v.index, v["v2 ahead"])], fontsize=8.4)
ax.set_xlabel("macro CV AUC, per-drug loci")
ax.set_title("LocusFusion v2 vs. LocusFusion by modality set, per-drug loci", loc="left")
ax.set_ylim(-0.5, len(v) - 0.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.8)
tidy(ax); fig.tight_layout(); save(fig, "figA1_v2_vs_v1")
print(v.round(4).to_string())
''')

md("""
v2 ties v1 on DNA and DNA + protein and is 0.017 behind whenever the regulatory
modality is loaded (0/11 drugs). `dna_biophysical` is missing because v2 has 8
of 11 drugs there.

---
### Caveats

- Single seed and one CV partition throughout.
- Figure 6's pooled Wilcoxon treats the 55 overlapping fold differences as
  independent; its p-values are optimistic.
- Figure 1's published baselines are not fold-paired with the other points.
- Figure 4 (left) uses each drug's 40 largest coefficients, the only ones the
  run kept.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                                                  "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(cells)} cells)")

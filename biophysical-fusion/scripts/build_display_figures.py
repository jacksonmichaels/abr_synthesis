"""Generate notebooks/display_figures.ipynb — the manuscript's figures, in paper order.

    python scripts/build_display_figures.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/display_figures.ipynb
    (or: sbatch scripts/sbatch/build_display_figures.sh)

Every figure is saved to results/figures/manuscript/ as PDF (for LaTeX) and PNG
as it is drawn; Tables 1 and 2 are written there as LaTeX (tab_family.tex,
tab_bigtb.tex). The last cell prints the tables and the LaTeX figure blocks, in
paper order, ready to paste.

Evaluation setting (--split, default lineage). The paper's setting is the
lineage-grouped split: Figures 2-4 and Table 1 are built from
results/experiments/lineage_split, loaded and tested by
results/analysis/lineage_paper_family/paper_family.py (imported, so every number
agrees with paper_family_lineage.csv / paper_family.log). Lineage cells are never
pooled with the random-split `frame` (lineage_split stays in
results_frame.DROP_RUNS). Table 2 is the one random-split table: the comparison
with the published Big-TB models on their own protocol (80/20 split at seed 42 +
five-fold CV).
--split random redraws Figures 2-4 from the random-split runs, for reference
only (the paper does not use them; Table 1 is lineage-only and is skipped).

Conventions: manuscript figures carry no title (the caption does that job) and
nothing explanatory inside the plot; white background; sized for a 6.5in text
width. Every colour comes from paper_style.py and has one meaning: model
identity (copper = LocusFusion v3, steel blue = CNNs, teal = LASSO) or
modality (plum DNA, mustard protein, pink biophysical, moss regulatory). The variant-token aggregators,
LocusFusion v1/v2 and the whole-genome LASSO are left out of every figure.
"""
import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("-o", "--output", type=Path, default=None,
                 help="notebook path (default: <project>/notebooks/display_figures.ipynb)")
_ap.add_argument("--split", choices=("lineage", "random"), default="lineage",
                 help="evaluation setting of Figures 2-4 (default: lineage, the paper's)")
_args = _ap.parse_args()
SPLIT = _args.split
OUT = (_args.output.expanduser().resolve() if _args.output
       else PROJECT / "notebooks" / "display_figures.ipynb")

cells = []


def md(src):
    src = src.replace("@SPLITNAME@", {"lineage": "lineage-grouped splits",
                                       "random": "random stratified split"}[SPLIT])
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.rstrip("\n").splitlines(keepends=True)})


def code(src):
    src = src.replace("@SPLIT@", SPLIT)
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md("""
# Manuscript figures

One cell per figure or table, in the order they appear in the paper; appendix
figures last, then the LaTeX to paste. Figures are saved to
`results/figures/manuscript/` (PDF + PNG), tables as `.tex`. Single-drug cells,
5-fold CV AUC, macro over the 11 shared drugs; networks at `all_modalities` and
the LASSO at `dna` unless a figure says otherwise. Figures 2–4 and Table 1:
@SPLITNAME@. Table 2: Big-TB's random-split protocol.

> Generated — edit `scripts/build_display_figures.py`, not the `.ipynb`.
""")

code(r'''
import sys, json, shutil
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats

_here = Path.cwd()
PROJECT = next(p for p in [_here, *_here.parents] if (p / "bigtb_ref.py").exists())
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import results_helpers as RH
from results_frame import load_frame, STD_MODSETS
from bigtb_baselines import SD_BASE_CV, MD_BASE_CV, ALL_DRUGS
from results_helpers import tidy, label_points, paper_family
# every colour comes from paper_style.py: change it there, every figure follows
import paper_style as PS
from paper_style import MODEL, MODALITY, model_color, ramp, diverging_cmap
PS.apply(RH)
SURFACE, INK, INK2, MUTED, AXIS, GRID = (PS.NEUTRAL[k] for k in
                                         ("surface", "ink", "ink2", "muted", "axis", "grid"))
LF_C, CNN_C, LASSO_C = MODEL["locusfusion"], MODEL["cnn"], MODEL["lasso"]

# manuscript type sizes for a 6.5in column
mpl.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.titlesize": 9, "axes.titleweight": "normal",
    "pdf.fonttype": 42,                     # editable text in the PDF
})
W = 6.5                                     # text width, inches

frame, meta = load_frame()
# LocusFusion is v3 only (results/experiments/locusfusion_v3, read as its compare.py
# does). v1 and v2 rows are dropped. v3 takes the "locusfusion" slot and v3 without
# the uncalled-base channel takes the "locusfusion_old" slot, so paper_family()
# runs unchanged with the ablation as its tokenizer row.
sys.path.insert(0, str(PROJECT / "results" / "experiments" / "locusfusion_v3"))
from compare import read_cells
_PD, _L19 = "single-drug", "single-drug, all loci"
frame = frame[~frame.model.isin(["locusfusion", "locusfusion_v2"])]
_v3 = read_cells("locusfusion_v3", {"sd19_": _L19, "sd_": _PD}, "locusfusion")
frame_lf_old = read_cells("locusfusion_v3", {"ablation_sd19_": _L19}, "locusfusion_old")
frame = pd.concat([frame, _v3], ignore_index=True)
# the paper's LASSOs are the 09-01 alignment run and the 09-03 whole-genome run;
# later sweeps (lasso_wholegenome_sweep_20260929) would add a second cell per drug
_lasso = frame.model.str.startswith("lasso")
frame = frame[~_lasso | frame.run.isin(["lasso_wholeseq_20260901",
                                        "lasso_wholegenome_20260903"])].reset_index(drop=True)
# the whole-genome run is only the 14th row of the Holm family (Table A1); no figure uses it

PD, L19 = "single-drug", "single-drug, all loci"
A, MODS5 = "all_modalities", STD_MODSETS
EST3 = ["mdcnn", "cisfusion", "late_fusion"]
V1 = "locusfusion"                          # v3, see above
BASELINES = {"sparse_baseline", "sparse_baseline_trainvocab",
             "lasso_onehot", "lasso_delta", "lasso_genome"}
LABEL = {"lasso_onehot": "LASSO", "mdcnn": "stacked CNN", "cisfusion": "cis-fusion",
         "late_fusion": "late-fusion", V1: "LocusFusion"}
OUTDIR = PROJECT / "results" / "figures" / "manuscript"
OUTDIR.mkdir(parents=True, exist_ok=True)


def hue(model):
    return LASSO_C if model in BASELINES else model_color(model)


# ---- evaluation setting of Figures 2-4 and Table 1
# The lineage cells and Table 1's tests come from paper_family.py itself, so the
# figures agree with paper_family_lineage.csv / paper_family.log to the digit.
# LFRAME is never concatenated with `frame` (random split).
EVAL_SPLIT = "@SPLIT@"            # not SPLIT: the appendix A2 cell uses that name
import importlib.util
_pf = PROJECT / "results" / "analysis" / "lineage_paper_family" / "paper_family.py"
_spec = importlib.util.spec_from_file_location("lineage_paper_family", _pf)
PF = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(PF)
LFRAME = PF.load_lineage()
assert set(LFRAME[LFRAME.model == "locusfusion"].src.str.split("/").str[1]
           .str.contains("_unk__locusfusion_v2")), "lineage LocusFusion must be v3 (_unk)"
FIG = LFRAME if EVAL_SPLIT == "lineage" else frame


def drug_cv(model, task, modset, fr=None):
    """Per-drug CV AUC of one or more (model, modset) cells, averaged. On the
    lineage split this is paper_family.drugmean (mean fold AUC)."""
    fr = FIG if fr is None else fr
    if fr is LFRAME:
        return PF.drugmean(LFRAME, task, model, modset).reindex(ALL_DRUGS)
    s = fr[(fr.model == model) & (fr.task == task) & (fr.modset == modset)]
    return s.groupby("drug").cv.mean().reindex(ALL_DRUGS)


def cell(model, task, fr=None):
    """The pinned modality set: dna for the LASSO, all_modalities for networks."""
    return drug_cv(model, task, "dna" if model in BASELINES else A, fr)


def mean5(model, task=None, fr=None):
    """Per-drug CV AUC matched over the five modality sets."""
    task, fr = task or L19, FIG if fr is None else fr
    if fr is LFRAME:
        return PF.drugmean(LFRAME, task, model, MODS5).reindex(ALL_DRUGS)
    s = fr[(fr.model == model) & (fr.task == task) & fr.modset.isin(MODS5)]
    return s.groupby(["drug", "modset"]).cv.mean().groupby("drug").mean().reindex(ALL_DRUGS)


def save(fig, name):
    fig.savefig(OUTDIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.show()


# LaTeX number formats shared by Tables 1 and 2
def _s4(v):
    return f"{'+' if v >= 0 else '-'}{abs(v):.4f}".replace("0.", ".", 1)
def fmt_d(v):
    return f"${_s4(v)}$"
def fmt_ci(hl, lo, hi):
    return f"${_s4(hl)}$ [${_s4(lo)}$, ${_s4(hi)}$]"
def fmt_p(v, s):
    """.12 / .052 / .0006 / $<$.0001, bold when significant."""
    t = ("$<$.0001" if v < 1e-4 else f"{v:.2f}" if v >= 0.1 else f"{v:.2g}" if v >= 1e-3
         else f"{v:.1g}").lstrip("0")
    return rf"\textbf{{{t}}}" if s else t
def fmt_auc(v):
    return f"{v:.4f}".lstrip("0")


def bare(ax):
    """No frame on a categorical axis."""
    for sp in ("left", "top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)


N_BY_DRUG = frame[frame.task == PD].groupby("drug").n_valid.max().reindex(ALL_DRUGS)
BY_N = N_BY_DRUG.sort_values().index.tolist()          # smallest cohort first
SD_REF, MD_REF = SD_BASE_CV.mean(), MD_BASE_CV.mean()     # Table 2 only
print(f"split {EVAL_SPLIT}   SD-CNN {SD_REF:.4f}   MD-CNN {MD_REF:.4f}   figures -> {OUTDIR}")
''')

# ------------------------------------------------------------------ figure 1
md("""
## Figure 1 — LocusFusion forward pass

`images/detailed_foward_pass.png` (`fig:forward`) is made outside this repo and is
not drawn here. The old schematic (`results/figures/paper/fig0_*`) is v1 and is not
used.
""")

# ------------------------------------------------------------------ figure 2
md("""
## Figure 2 — locus set and parameter count

One figure for two Results subsections, @SPLITNAME@. (a) Macro CV AUC per
model on each drug's own loci (open) and on all 19 loci (filled). (b) Every
network × modality set at 19 loci against its parameter count; dashed line is
the LASSO. The published Big-TB models are not drawn: they were scored on a
random split, so they appear only in Table 2.
""")

code(r'''
MODELS = ["lasso_onehot", "late_fusion", "cisfusion", "mdcnn", V1]
m = pd.DataFrame({mo: {"per-drug": cell(mo, PD).mean(), "19": cell(mo, L19).mean()}
                  for mo in MODELS}).T.sort_values("19")
if EVAL_SPLIT == "lineage":      # paper_family.spearman_cells: mean CV AUC, median n_params
    _g, _r, _rc = PF.spearman_cells(LFRAME)
    cells19 = _g.rename(columns={"n": "params"}).reset_index()
    (rho, p), (rho_cnn, p_cnn) = (_r.statistic, _r.pvalue), (_rc.statistic, _rc.pvalue)
else:
    cells19 = (frame[(frame.task == L19) & frame.model.isin(EST3 + [V1])
                     & frame.n_params.notna()]
               .groupby(["model", "modset"])
               .agg(cv=("cv", "mean"), n=("drug", "nunique"), params=("n_params", "mean"))
               .reset_index().query("n == 11"))
    rho, p = stats.spearmanr(cells19.params, cells19.cv)
    _cnn = cells19[cells19.model.isin(EST3)]
    rho_cnn, p_cnn = stats.spearmanr(_cnn.params, _cnn.cv)
assert len(cells19) == 20, len(cells19)
lasso19 = cell("lasso_onehot", L19).mean()

fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, 2.75),
                             gridspec_kw=dict(width_ratios=[1.12, 1]))
# (a) locus set
y = np.arange(len(m))
for yi, (mo, r) in zip(y, m.iterrows()):
    ax.plot([r["per-drug"], r["19"]], [yi, yi], color=hue(mo), alpha=0.35, lw=2.0,
            zorder=1, solid_capstyle="round")
    ax.scatter([r["per-drug"]], [yi], s=34, zorder=3, color=SURFACE, edgecolor=hue(mo),
               linewidth=1.6)
    ax.scatter([r["19"]], [yi], s=34, zorder=3, color=hue(mo), edgecolor=SURFACE,
               linewidth=1.0)
    _right = r["19"] >= r["per-drug"]          # label on the side away from the bar
    ax.annotate(f"{r['19']:.3f}", (r["19"], yi), xytext=(5 if _right else -5, 0),
                textcoords="offset points", va="center", ha="left" if _right else "right",
                fontsize=7, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([LABEL[i] for i in m.index])
ax.set_ylim(-0.6, len(m) - 0.4)
ax.set_xlim(m.values.min() - 0.006, m.values.max() + 0.011)
ax.set_xlabel("macro CV AUC")
ax.legend([Line2D([0], [0], marker="o", lw=0, markersize=5.5, markerfacecolor=SURFACE,
                  markeredgecolor=INK2, markeredgewidth=1.4),
           Line2D([0], [0], marker="o", lw=0, markersize=5.5, color=INK2)],
          ["per-drug loci", "19 loci"], loc="upper center",
          bbox_to_anchor=(0.5, -0.24), ncol=2, handletextpad=0.3)
tidy(ax); bare(ax)
# (b) parameter count
bx.scatter(cells19.params, cells19.cv, s=26, color=[hue(mo) for mo in cells19.model],
           edgecolor=SURFACE, linewidth=0.8, zorder=3)
bx.axhline(lasso19, color=LASSO_C, lw=1.0, ls=(0, (3, 3)), zorder=1)
bx.set_xscale("log")
bx.set_xlim(cells19.params.min() / 2.5, cells19.params.max() * 2.5)
bx.set_ylim(cells19.cv.min() - 0.005, cells19.cv.max() + 0.009)
bx.annotate(f"LASSO {lasso19:.4f}", (cells19.params.min() / 2.4, lasso19), va="top",
            xytext=(0, -3), textcoords="offset points", fontsize=6.6, color=INK2)
lab = cells19.sort_values("cv").groupby("model").tail(1)       # one label per model
label_points(bx, lab.params, lab.cv, [LABEL.get(mo, mo) for mo in lab.model], fontsize=6.6)
bx.set_xlabel("parameters (log scale)"); bx.set_ylabel("macro CV AUC")
tidy(bx, ygrid=True)
fig.tight_layout(w_pad=1.5)
for a, k in ((ax, "a"), (bx, "b")):
    pos = a.get_position()
    fig.text(pos.x0 - (0.17 if a is ax else 0.075), pos.y1 + 0.02, k, fontsize=10,
             fontweight="bold", color=INK, va="bottom")
save(fig, "fig2_locus_size")
print(m.round(4).to_string()); print()
print(f"all cells: rho {rho:+.3f}  p {p:.2g}  n {len(cells19)}  "
      f"params {cells19.params.min():,.0f}-{cells19.params.max():,.0f}")
print(f"established models only: rho {rho_cnn:+.3f}  p {p_cnn:.4f}  "
      f"n {int(cells19.model.isin(EST3).sum())}")
''')

md("""
At 19 loci LocusFusion is highest (0.9134), then the stacked CNN (0.9017), the
LASSO (0.8985), cis-fusion (0.8978) and late-fusion (0.8771). On per-drug loci
the four networks sit within 0.004 (0.8852–0.8891) and the LASSO is lowest
(0.8316). In (b), ρ = −0.82 over all 20 cells and −0.60 (p = 0.017) among the
established models alone.
""")

# ------------------------------------------------------------------ figure 3
md("""
## Figure 3 — what each modality adds, per drug

Gain in CV AUC over DNA only, per drug, from adding one modality (bars) or all
modalities (◆), @SPLITNAME@. Rows: established-model mean (stacked CNN,
cis-fusion, late-fusion) and LocusFusion; columns: per-drug loci and 19 loci.
""")

code(r'''
ADDS = {"dna_protein": ("+ protein", MODALITY["protein"]),
        "dna_biophysical": ("+ biophysical", MODALITY["biophysical"]),
        "dna_regulatory": ("+ regulatory", MODALITY["regulatory"])}
MS = [*ADDS, A]
def gain(models, task, ms):
    return pd.concat([drug_cv(mo, task, ms) - drug_cv(mo, task, "dna") for mo in models],
                     axis=1).mean(axis=1)
EST = "established models"
G = {(who, t): pd.DataFrame({ms: gain(models, t, ms) for ms in MS})
     for who, models in ((EST, EST3), ("LocusFusion", [V1])) for t in (PD, L19)}
order = G[(EST, PD)][A].sort_values(ascending=False).index
ABBR = {"AMIKACIN": "AMK", "CAPREOMYCIN": "CAP", "ETHAMBUTOL": "EMB",
        "ETHIONAMIDE": "ETO", "ISONIAZID": "INH", "KANAMYCIN": "KAN",
        "LEVOFLOXACIN": "LFX", "MOXIFLOXACIN": "MFX", "PYRAZINAMIDE": "PZA",
        "RIFAMPICIN": "RIF", "STREPTOMYCIN": "STM"}
PANELS = [(EST, PD), (EST, L19), ("LocusFusion", PD), ("LocusFusion", L19)]
fig, axes = plt.subplots(2, 2, figsize=(W, 2.95), sharex=True, sharey=True)
x = np.arange(len(order)); w = 0.27
for ax, (who, t) in zip(axes.flat, PANELS):
    g = G[(who, t)].loc[order]
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    for k, (ms, (lab, colour)) in enumerate(ADDS.items()):
        ax.bar(x + (k - 1) * w, g[ms], width=w, color=colour, edgecolor=SURFACE,
               linewidth=0.5, zorder=2, label=lab)
    ax.scatter(x, g[A], marker="D", s=10, color=INK2, edgecolor=SURFACE, linewidth=0.5,
               zorder=3, label="all modalities")
    ax.set_title(f"{who}, {'per-drug loci' if t == PD else '19 loci'}", loc="left",
                 fontsize=7.8, color=CNN_C if who == EST else LF_C, fontweight="bold", pad=2)
    tidy(ax, xgrid=False, ygrid=True)
    ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", length=0)
for ax in axes[-1]:
    ax.set_xticks(x); ax.set_xticklabels([ABBR[d] for d in order], fontsize=6.8)
for ax in axes[:, 0]:
    ax.set_ylabel("Δ CV AUC vs. DNA", fontsize=7.4)
fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=0.4, w_pad=0.8)
h, l = axes[0, 0].get_legend_handles_labels()
h, l = zip(*sorted(zip(h, l), key=lambda t: t[1] == "all modalities"))   # diamond last
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=7.2,
           handlelength=1.1, columnspacing=1.2, frameon=False)
save(fig, "fig3_modalities")
macro = pd.DataFrame({f"{who} | {'per-drug' if t == PD else '19'}": G[(who, t)].mean()
                      for who, t in PANELS}).T
print(macro.round(4).to_string()); print()
print(pd.concat({f"{who} | {t}": G[(who, t)] for who, t in PANELS}, axis=1)
      .loc[order].round(4).to_string())
''')

md("""
At per-drug loci the established models gain +0.0215 (macro, all modalities),
where the mechanism predicts: regulatory carries ethionamide (+0.085; +0.103
with all modalities) and isoniazid (+0.038; +0.049), the *fabG1–inhA* promoter;
protein and biophysical carry pyrazinamide (+0.036 / +0.043, *pncA*); rifampicin
+0.003. At 19 loci the all-modality gain is −0.0021 for the established models
and +0.0009 for LocusFusion.
""")

# ------------------------------------------------------------------ figure 4
md("""
## Figure 4 — LocusFusion per drug

Per-drug CV AUC difference at 19 loci, @SPLITNAME@, drawn at its printed size
(half the text width, wrapped beside the text). Column 1 is LocusFusion (all
modalities) − LASSO (DNA); column 2 is LocusFusion − stacked CNN matched over the
five modality sets; both as in Table 1, so each macro value is the table's Δ.
""")

code(r'''
lf = cell(V1, L19)
cols = {"− LASSO": lf - cell("lasso_onehot", L19),
        "− stacked\nCNN": mean5(V1) - mean5("mdcnn")}
if EVAL_SPLIT == "random":                      # the ablation was run on the random split only
    cols["− no uncalled-\nbase channel"] = (mean5(V1)
                                            - mean5("locusfusion_old", fr=frame_lf_old))
d = pd.DataFrame(cols).reindex(BY_N[::-1])
d.loc["macro"] = d.mean()
lim = float(np.ceil(np.abs(d.values).max() / 0.005) * 0.005)
cmap = diverging_cmap()                        # LocusFusion ahead = copper, behind = grey
fig, ax = plt.subplots(figsize=(3.25, 3.85))
ax.imshow(d.values, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
for (r, c), v in np.ndenumerate(d.values):
    ax.text(c, r, f"{v:+.4f}", ha="center", va="center", fontsize=6.4,
            color=SURFACE if abs(v) > 0.6 * lim else INK,
            fontweight="bold" if d.index[r] == "macro" else "normal")
ax.axhline(len(d) - 1.5, color=SURFACE, lw=3)                 # macro row apart
for c in range(1, d.shape[1]):
    ax.axvline(c - 0.5, color=SURFACE, lw=2)
ax.set_xticks(range(d.shape[1]))
ax.set_xticklabels([f"{c}\n({int((d[c].iloc[:-1] > 0).sum())}/11)" for c in d.columns],
                   fontsize=6.4)
ax.xaxis.tick_top()
ax.set_yticks(range(len(d)))
ax.set_yticklabels([f"{dr.title()} ({int(N_BY_DRUG[dr]):,})" if dr in N_BY_DRUG
                    else "macro" for dr in d.index], fontsize=6.6)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
fig.tight_layout(); save(fig, "fig4_locusfusion_per_drug")
print(d.round(4).to_string())
''')

md("""
LocusFusion is ahead of the LASSO on all 11 drugs (macro +0.0149) and of the
stacked CNN on 6 (+0.0114). Levofloxacin (n = 269) carries most of the margin
over the stacked CNN (+0.094); over the other ten drugs it averages +0.0032, and
LocusFusion is behind on amikacin, capreomycin, ethambutol, kanamycin and
rifampicin (−0.0004 to −0.0029).
""")

# ------------------------------------------------------------------ table 1
# Figure 7 (random vs lineage) and tab_lineage_split.tex were removed from the
# build on 2026-10-26: the paper no longer compares the two splits. The old files
# stay on disk in results/figures/manuscript/ but are not regenerated.
md("""
## Table 1 — the comparison family, lineage-grouped splits

The 11 rows of `results/analysis/lineage_paper_family/paper_family_lineage.csv`,
recomputed here by `paper_family.family()` and checked against the CSV. Δ is the
difference in macro CV AUC. Written to `results/figures/manuscript/tab_family.tex`.
""")

code(r'''
TAB_FAMILY = None
if EVAL_SPLIT == "lineage":
    fam = PF.family(LFRAME)
    _csv = pd.read_csv(PF.OUT / "paper_family_lineage.csv")
    assert list(fam.comparison) == list(_csv.comparison)
    for c_ in ("delta", "hl", "lo", "hi", "p", "p_holm"):
        assert np.allclose(fam[c_], _csv[c_], rtol=0, atol=1e-15), c_
    assert list(fam.wins) == list(_csv.wins) and list(fam.verdict) == list(_csv.verdict)
    # model-size labels from the median parameter count over the five modality sets
    _np = (LFRAME[(LFRAME.task == L19) & LFRAME.modset.isin(MODS5)]
           .groupby("model").n_params.median() / 1e6)
    SZ = {k: f"{_np[mo]:.1f}M" for k, mo in (("st", "mdcnn"), ("cis", "cisfusion"),
                                               ("late", "late_fusion"))}
    FROWS = {   # comparison -> (group, label)
        "lasso − stacked · 19": ("LASSO vs.\\ CNN", r"LASSO $-$ stacked, 19 loci"),
        "lasso − stacked · per-drug": ("", r"LASSO $-$ stacked, per-drug"),
        "LF − lasso · 19": ("LocusFusion (ours)", r"LF $-$ LASSO, 19"),
        "LF − stacked · 19 · 5 modsets": ("", r"LF $-$ stacked, 19$^a$"),
        "locus set · established · 5 modsets": ("Locus set", r"Established$^{a,b}$"),
        "locus set · LF · 5 modsets": ("(19 $-$ per-drug)", r"LF$^a$"),
        "locus set · lasso": ("", "LASSO"),
        "modalities · established · all − dna · per-drug":
            ("Modalities (per-drug)", r"Established$^b$, all $-$ DNA"),
        "stacked − late · 19 · 5 modsets":
            ("Model size (19 loci)", rf"stacked ({SZ['st']}) $-$ late ({SZ['late']})$^a$"),
        "stacked − cis · 19 · 5 modsets":
            ("", rf"stacked ({SZ['st']}) $-$ cis ({SZ['cis']})$^a$"),
        "cis − late · 19 · 5 modsets":
            ("", rf"cis ({SZ['cis']}) $-$ late ({SZ['late']})$^a$"),
    }
    assert set(fam.comparison) == set(FROWS), set(fam.comparison) ^ set(FROWS)
    lines, prev = [], None
    for r in fam.itertuples():
        g, lab = FROWS[r.comparison]
        if g and prev is not None and not g.startswith("("):
            lines.append(r"  \midrule")
        lines.append(f"  {('\\textit{' + g + '}') if g else ''} & {lab} & {fmt_d(r.delta)} & "
                     f"{r.wins} & "
                     f"{fmt_p(r.p_holm, r.verdict == 'significant')} \\\\")
        prev = g or prev
    TAB_FAMILY = "\n".join([
        r"\begin{table}[tbp]", r"  \centering\footnotesize", r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{llrrr}", r"  \toprule",
        r"   & Comparison & $\Delta$ & Wins & $p_\text{Holm}$\\", r"  \midrule",
        *lines, r"  \bottomrule", r"  \end{tabular}",
        r"\caption{Paired comparisons of CV AUC over 55 fold differences (11 drugs $\times$ 5 "
        r"lineage-grouped folds). stacked = stacked CNN, cis = cis-fusion, late = late-fusion, "
        r"LF = LocusFusion; 19 = all 19 loci, per-drug = each drug's own loci. $\Delta$: "
        r"difference in macro CV AUC. Wins: drugs on which the first "
        r"model has the higher mean fold AUC. $p_\text{Holm}$: two-sided exact Wilcoxon "
        r"signed-rank, Holm-adjusted over the 11 comparisons shown; bold: $p_\text{Holm} < 0.05$. "
        r"$^a$Matched over the five modality sets. $^b$Pools stacked, cis and late-fusion "
        r"models.}",
        r"\label{tab:family}", r"\end{table}"])
    (OUTDIR / "tab_family.tex").write_text(TAB_FAMILY + "\n")
    print(TAB_FAMILY)
    print(fam[["comparison", "delta", "hl", "lo", "hi", "wins", "p", "p_holm"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
else:
    print("Table 1 is lineage-only; not built with --split random")
''')

md("""
Nine of the 11 rows are significant after Holm. The two that are not: LASSO −
stacked at 19 loci (−0.0032, 2/11, p_Holm .052) and stacked − cis (+0.0048,
8/11, .12). LF − stacked is +0.0114 on 6/11 drugs (.037), most of it from
levofloxacin (Figure 4).
""")

# ------------------------------------------------------------------ table 2
md("""
## Table 2 — Big-TB comparison on its own protocol

The one random-split table. Big-TB's models were scored on a random 80/20 split
(seed 42) with five-fold CV on the training portion; our random-split runs
(`full_run_v2`, `alllocus_run_v2`, `locusfusion_v3`, `lasso_wholeseq_20260901`)
follow that protocol with stratified folds. Published SD-/MD-CNN macro CV AUCs
come from `bigtb_baselines` (MD-CNN: the authors' own run log, `ABR_MDCNN_RUN_DIR`);
our rerun of the authors' MD-CNN code is `mdcnn_eval/training_output/
repro_filter12_epoch150/auc.csv`. Written to
`results/figures/manuscript/tab_bigtb.tex`.
""")

code(r'''
RAND = {mo: {"per-drug": cell(mo, PD, fr=frame).mean(), "19": cell(mo, L19, fr=frame).mean()}
        for mo in (V1, "mdcnn", "lasso_onehot")}
# stacked CNN trained jointly on all drugs, DNA, Big-TB's 19 loci (full_run_v2)
_j = frame[(frame.model == "mdcnn") & (frame.task == "multi-drug") & (frame.modset == "dna")]
assert _j.run.unique().tolist() == ["full_run_v2"] and _j.drug.nunique() == 11
JOINT = _j.groupby("drug").cv.mean().reindex(ALL_DRUGS).mean()
_rr = pd.read_csv(PROJECT.parent / "mdcnn_eval" / "training_output"
                  / "repro_filter12_epoch150" / "auc.csv")
_rr = _rr[_rr.Drug.isin(ALL_DRUGS)].groupby("Drug").AUC
assert (_rr.size() == 5).all() and _rr.ngroups == 11
RERUN = _rr.mean().reindex(ALL_DRUGS).mean()

na = "--"
brows = [
    (r"\textit{Ours, one model per drug}", None, None, None),
    ("LocusFusion", "all modalities", RAND[V1]["per-drug"], RAND[V1]["19"]),
    ("stacked CNN", "all modalities", RAND["mdcnn"]["per-drug"], RAND["mdcnn"]["19"]),
    ("LASSO", "DNA", RAND["lasso_onehot"]["per-drug"], RAND["lasso_onehot"]["19"]),
    (r"\textit{Ours, one model for all drugs}", None, None, None),
    ("stacked CNN", "DNA", None, JOINT),
    (r"MD-CNN, authors' code, our rerun", "DNA", None, RERUN),
    (r"\textit{Big-TB, published}", None, None, None),
    ("SD-CNN", "DNA", SD_REF, None),
    ("MD-CNN", "DNA", None, MD_REF),
]
bl = []
for name, inp, a_, b_ in brows:
    if inp is None:
        if bl:
            bl.append(r"  \midrule")
        bl.append(rf"  \multicolumn{{4}}{{l}}{{{name}}} \\")
        continue
    f = lambda v: na if v is None else fmt_auc(v)
    bl.append(rf"  \quad {name} & {inp} & {f(a_)} & {f(b_)} \\")
TAB_BIGTB = "\n".join([
    r"\begin{table}[tbp]", r"  \centering\footnotesize", r"  \setlength{\tabcolsep}{4pt}",
    r"  \begin{tabular}{llrr}", r"  \toprule",
    r"  Model & Input & per-drug loci & 19 loci \\", r"  \midrule", *bl,
    r"  \bottomrule", r"  \end{tabular}",
    r"\caption{Comparison with the published Big-TB models on their evaluation setting, a "
    r"random 80/20 train/test split at seed 42 with five-fold CV on the training portion, "
    r"not the lineage-grouped splits used everywhere else. Our models use stratified folds "
    r"at seed 42. Macro CV AUC over the 11 drugs; the per-drug-loci models see each "
    r"drug's own loci, the 19-loci models all 19 (Big-TB's set). The published SD-CNN and "
    r"MD-CNN, and our rerun of the authors' MD-CNN code, were scored on a different fold "
    r"assignment, so these comparisons are unpaired and are not tested.}",
    r"\label{tab:bigtb}", r"\end{table}"])
(OUTDIR / "tab_bigtb.tex").write_text(TAB_BIGTB + "\n")
print(TAB_BIGTB)
print(pd.DataFrame(RAND).T.round(4).to_string())
print(f"joint stacked CNN {JOINT:.4f}   MD-CNN rerun {RERUN:.4f}   "
      f"SD-CNN {SD_REF:.4f}   MD-CNN {MD_REF:.4f}")
''')

md("""
On Big-TB's protocol LocusFusion is 0.9323 at 19 loci against the published
MD-CNN's 0.9222 and our rerun of its code at 0.9212; on per-drug loci it is
0.9115 against the SD-CNN's 0.8636.
""")

# ------------------------------------------------------------------ appendix
md("""
## Appendix figures — Shapley attribution, Big-TB's protocol

The three appendix figures in the manuscript (`fig:shapley_modality`,
`fig:shapley_locus`, `fig:own_who`). They are drawn by
`notebooks/shap_shapley_v3_alltest.ipynb` (built by
`scripts/build_shap_alltest_notebook.py`) from the random-split LocusFusion v3
checkpoints, and are copied here unchanged so every manuscript file sits in one
folder.
""")

code(r'''
from IPython.display import Image, display
SHAPLEY = ["shapley_alltest_modality_shares", "shapley_alltest_locus_heatmap",
           "alltest_own_who_share"]
_sv = PROJECT / "results" / "figures" / "shap_v3_alltest"
for _n in SHAPLEY:
    for _ext in ("png", "pdf"):
        shutil.copy(_sv / f"{_n}.{_ext}", OUTDIR / f"{_n}.{_ext}")
    print(_n); display(Image(filename=str(OUTDIR / f"{_n}.png"), width=800))
''')

md("""
## Not in the manuscript

A1–A3 below are commented out in the manuscript and are not in the LaTeX blocks.
A1 is the random-split LASSO (coefficient mass); the text's 69% is the
lineage-split LASSO's 40 largest coefficients by count.

### A1 — where the LASSO puts its weight

The 19-locus LASSO's 40 largest coefficients per drug, split by whether the
locus is in that drug's own gene map. LocusFusion is not drawn: the figure is
about the LASSO's own coefficients.
""")

code(r'''
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
order = l19.sort_values("drug's own loci").index

fig, ax = plt.subplots(figsize=(4.55, 2.9))
y = np.arange(len(order)); left = np.zeros(len(order))
for c, colour in zip(l19.columns, ramp(LASSO_C, 3)[:2]):
    v = l19.loc[order, c].values
    ax.barh(y, v, left=left, height=0.68, color=colour, edgecolor=SURFACE,
            linewidth=0.8, zorder=2, label=c)
    left += v
ax.set_xlim(0, 100); ax.set_xlabel("|coefficient| mass (%)")
ax.set_yticks(y); ax.set_yticklabels([dr.title() for dr in order])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.2)
tidy(ax); bare(ax)
fig.tight_layout(); save(fig, "figA1_lasso_coefficients")
print(l19.loc[order].round(1).to_string())
''')

md("""
At 19 loci, 26–52% of the LASSO's largest coefficients sit on the drug's own
genes; the rest are on other drugs' targets.
""")

md("""
### A2 — SHAP modality shares, all modalities, per drug

LocusFusion, all modalities, 19 loci: share of total |attribution| per input
modality (expected gradients, 300 test isolates, 100-isolate training
background). Biophysical is split into the residue-properties slot (token
dims 29–31) and the change-from-reference slot (38–40) when the token-level
run (`results/analysis/shap_locusfusion_v3/token_split/`) is present; otherwise
the two are drawn as one.
""")

code(r'''
SHD = PROJECT / "results" / "analysis" / "shap_locusfusion_v3"
SPLIT = SHD / "token_split" / "modality_shares_split.csv"
ALLC = "sd19_all_modalities_unk"
if SPLIT.exists():
    # The two biophysical paths partly cancel (|residue| + |delta| > |total|), so the
    # biophysical segment keeps its undivided share and is split in proportion to the
    # two paths' magnitudes; DNA, protein and regulatory shares are unchanged.
    sh = pd.read_csv(SPLIT)
    _f = sh.alt_share_biophysical_residue / (sh.alt_share_biophysical_residue
                                             + sh.alt_share_biophysical_delta)
    sh = sh.assign(bio_res=sh.alt_share_biophysical * _f,
                   bio_delta=sh.alt_share_biophysical * (1 - _f),
                   dna=sh.alt_share_dna, protein=sh.alt_share_protein,
                   regulatory=sh.alt_share_regulatory)
    SEG = [("dna", "DNA", MODALITY["dna"]),
           ("protein", "protein", MODALITY["protein"]),
           ("bio_res", "biophysical (residue)", MODALITY["biophysical"]),
           ("bio_delta", "biophysical (change from H37Rv)",
            ramp(MODALITY["biophysical"], 3)[0]),
           ("regulatory", "regulatory", MODALITY["regulatory"])]
else:
    print("token-level split not found: biophysical drawn as one segment")
    sh = pd.read_csv(SHD / "modality_shares.csv")
    SEG = [("share_dna", "DNA", MODALITY["dna"]),
           ("share_protein", "protein", MODALITY["protein"]),
           ("share_biophysical", "biophysical", MODALITY["biophysical"]),
           ("share_regulatory", "regulatory", MODALITY["regulatory"])]
sh = sh[(sh.cell == ALLC) & (sh.background == "train")].set_index("drug").reindex(ALL_DRUGS)
S = sh[[c for c, _, _ in SEG]]
S = S.div(S.sum(axis=1), axis=0)
order = S.sort_values(SEG[0][0]).index                   # by DNA share
fig, ax = plt.subplots(figsize=(W, 3.0))
y = np.arange(len(order)); left = np.zeros(len(order))
for c, lab, colour in SEG:
    v = S.loc[order, c].values
    ax.barh(y, v, left=left, height=0.7, color=colour, edgecolor=SURFACE, linewidth=0.8,
            zorder=2, label=lab)
    for yi, (l0, vi) in enumerate(zip(left, v)):
        if vi >= 0.06:
            ax.text(l0 + vi / 2, yi, f"{100 * vi:.0f}%", ha="center", va="center",
                    fontsize=6.4, color=SURFACE)
    left += v
ax.set_xlim(0, 1); ax.set_xlabel("share of |attribution|")
ax.set_yticks(y); ax.set_yticklabels([dr.title() for dr in order])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=min(len(SEG), 3), fontsize=7,
          handlelength=1.1, columnspacing=1.0)
tidy(ax); bare(ax)
fig.tight_layout(); save(fig, "figA2_shap_modality_shares")
print(S.loc[order].round(3).to_string()); print(S.mean().round(3).to_string())
''')

md("""
### A3 — SHAP locus shares, all modalities, per drug

Same runs as A2: each locus's share of |attribution| per drug (a locus's
regulatory window is credited to the locus). Outlined: the drug's own WHO tier
1+2 genes.
""")

code(r'''
loc = pd.read_csv(SHD / "locus_shares.csv")
lcat = pd.read_csv(SHD / "who" / "locus_categories.csv")
L = loc[(loc.background == "train") & (loc.cell == ALLC)]
LOCI = L.groupby("locus").share.mean().sort_values(ascending=False).index.tolist()
H = L.pivot(index="drug", columns="locus", values="share").reindex(index=ALL_DRUGS, columns=LOCI)
own = (lcat.assign(own=lcat.category.isin(["own_t1", "own_t2"]))
       .pivot(index="drug", columns="locus", values="own").reindex(index=ALL_DRUGS, columns=LOCI))
seq = LinearSegmentedColormap.from_list("lf_seq", [SURFACE, ramp(LF_C, 3)[1], ramp(LF_C, 3)[0]])
fig, ax = plt.subplots(figsize=(W, 3.1))
im = ax.imshow(H.values, cmap=seq, vmin=0, vmax=float(np.nanmax(H.values)), aspect="auto")
for (i, j), o in np.ndenumerate(own.values):
    if o is True or o == 1:
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.0))
ax.set_yticks(range(len(ALL_DRUGS))); ax.set_yticklabels([ABBR[d] for d in ALL_DRUGS])
ax.set_xticks(range(len(LOCI)))
ax.set_xticklabels(LOCI, rotation=60, ha="right", fontstyle="italic")
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
cb.set_label("locus share of |attribution|"); cb.outline.set_visible(False)
fig.tight_layout(); save(fig, "figA3_shap_locus_shares")
print(H.round(3).to_string())
''')

md("""
### Caveats

- Single seed and one CV partition per cell. Figures 2–4 and Table 1 are on
  the lineage-grouped splits (Beijing, lineage 2.2.1, is a fold of its own and
  about twice the size of the others, with a higher resistant share);
  levofloxacin has 9–23 resistant isolates per partition and carries most of
  LF − stacked.
- The pooled Wilcoxon (Table 1) treats the 55 overlapping fold differences
  as independent; its p-values are optimistic. Per-drug effects (Figures 3, 4)
  rest on five folds each and are sizes, not tested differences.
- On the lineage split, per-drug and 19-locus runs read the same split file,
  so all 55 locus-set pairs match.
- Table 2 is on a different setting (Big-TB's random split). Its published
  SD-/MD-CNN values and our MD-CNN rerun are on other fold assignments and are
  unpaired.
- The uncalled-base ablation is not reported (it was run only on the random
  split; `--split random` still draws it as Figure 4's third column).
- Figure 7 and `tab_lineage_split.tex` (random vs lineage) are no longer built;
  the old files remain in `results/figures/manuscript/`.
- Figure 2a is the all-modalities slice: there late-fusion is lower at 19 loci
  (0.8886 → 0.8771). Matched over the five modality sets (Table 1 and the text)
  every family gains, late-fusion +0.0042.
- The Shapley appendix figures are on Big-TB's protocol (random-split
  checkpoints), not on lineage checkpoints; their captions say so.
""")

# ------------------------------------------------------------------ LaTeX
md("""
## LaTeX blocks

Printed in paper order, using the file names the manuscript already uses in
`images/`. The first lines of the output list which file in
`results/figures/manuscript/` to copy to which name.
""")

code(r'''
_ON = ", on lineage-grouped splits" if EVAL_SPLIT == "lineage" else ", on the random stratified split"
_FIG4 = ({"lineage":
          r"Per-drug difference in CV AUC at 19 loci@ON@, between LocusFusion and the LASSO "
          r"and between LocusFusion and the stacked CNN. The LASSO column compares LocusFusion "
          r"with all modalities to the LASSO on DNA; the stacked CNN column is matched over the "
          r"five modality sets. Drugs are ordered by labelled cohort size (in parentheses); the "
          r"bottom row is the macro mean, and column headers give the number of drugs on which "
          r"LocusFusion is ahead.",
          "random":
          r"Per-drug difference in CV AUC at 19 loci@ON@, between LocusFusion and the LASSO, the "
          r"stacked CNN, and LocusFusion without the uncalled-base channel. The LASSO column "
          r"compares LocusFusion with all modalities to the LASSO on DNA; the other two are "
          r"matched over the five modality sets. Drugs are ordered by labelled cohort size (in "
          r"parentheses); the bottom row is the macro mean, and column headers give the number "
          r"of drugs on which LocusFusion is ahead."}[EVAL_SPLIT])
# manuscript images/ name <- results/figures/manuscript/ source
FILES = {"locus_and_params.png": "fig2_locus_size.png",
         "modalities.png": "fig3_modalities.png",
         "lf.png": "fig4_locusfusion_per_drug.png",
         **{f"{n}.png": f"{n}.png" for n in SHAPLEY}}
BLOCKS = [
    ("locus_and_params.png", "fig:locus_size", "figure",
     r"Macro CV AUC by locus set and by parameter count@ON@. (a) Each model on each drug's "
     r"own loci (open) and on all 19 loci (filled); networks use all modalities and the LASSO "
     r"uses DNA. Table~\ref{tab:family} gives the locus-set gains matched over the five "
     r"modality sets. (b) Each network and modality set at 19 loci (n = @N@) against its parameter "
     r"count; dashed line: LASSO. Spearman $\rho = @RHO@$ ($p @P@$); among the established "
     r"models alone $\rho = @RHOC@$ ($p @PC@$)."),
    ("modalities.png", "fig:modality", "figure",
     r"Change in CV AUC from adding protein, biophysical or regulatory input to DNA, per "
     r"drug@ON@. Rows: mean over the established models (stacked CNN, cis-fusion and "
     r"late-fusion), and LocusFusion; columns: per-drug loci and all 19 loci. Bars: one "
     r"modality added; diamonds: all modalities. Drugs are ordered by the established "
     r"models' all-modality gain on per-drug loci. AMK amikacin, CAP capreomycin, EMB "
     r"ethambutol, ETO ethionamide, INH isoniazid, KAN kanamycin, LFX levofloxacin, MFX "
     r"moxifloxacin, PZA pyrazinamide, RIF rifampicin, STM streptomycin."),
    ("lf.png", "fig:lf", "wrapfigure", _FIG4),
]
# the manuscript's own appendix captions, with the setting stated and the dangling
# fig:lasso_coef reference removed
APPENDIX = [
    ("shapley_alltest_modality_shares.png", "fig:shapley_modality", "figure[H]",
     r"Shapley values divide a prediction among input groups by averaging each group's "
     r"marginal contribution over all orders of adding the groups. Shapley attribution share "
     r"by input modality for LocusFusion at 19 loci, trained on Big-TB's split protocol "
     r"(Section~\ref{sec:bigtb}), for each of the five modality sets, mean over the 11 drugs. "
     r"Shapley values are computed exactly over all coalitions of modalities, with an absent "
     r"modality set to the H37Rv reference, for every test isolate (17,433 per modality set, "
     r"54 to 3,517 per drug) using the best cross-validation fold's checkpoint. Share: a "
     r"modality's mean $|\phi|$ over the sum across modalities. Right: the share taken by the "
     r"uncalled-base symbols (\texttt{N} in DNA and regulatory; deleted and unreadable codons "
     r"in protein) when they are split off as their own players."),
    ("shapley_alltest_locus_heatmap.png", "fig:shapley_locus", "figure",
     r"Shapley attribution share by locus for LocusFusion at 19 loci, DNA + regulatory (top) "
     r"and all modalities (bottom), on the isolates of Figure~\ref{fig:shapley_modality}. "
     r"Locus values are estimated from 32 antithetic permutations, or enumerated exactly when "
     r"an isolate differs from H37Rv at nine or fewer loci; a locus's regulatory window is "
     r"credited to the locus. Outlined: the drug's own WHO tier 1 and 2 genes. Drug "
     r"abbreviations as in Figure~\ref{fig:modality}."),
    ("alltest_own_who_share.png", "fig:own_who", "figure",
     r"Share of attribution on each drug's own WHO tier 1 and 2 genes for LocusFusion at 19 "
     r"loci, by modality set, mean over the 11 drugs: Shapley (filled) and SHAP expected "
     r"gradients over a 100-isolate training background (open), both on all test isolates "
     r"(Big-TB's split protocol). Dashed line: the same share of the 19-locus LASSO's 40 "
     r"largest coefficients. Dotted line: the share if every locus received the same "
     r"attribution. Attribution shares and coefficient mass are different quantities."),
]
def _prel(v):                                     # "< 0.001" / "= 0.017"
    return "< 0.001" if v < 1e-3 else f"= {v:.2g}"
fill = {"@ON@": _ON, "@N@": str(len(cells19)), "@RHO@": f"{rho:+.2f}",
        "@P@": _prel(p), "@RHOC@": f"{rho_cnn:+.2f}", "@PC@": _prel(p_cnn)}

def block(fname, label, env, caption):
    for k, v in fill.items():
        caption = caption.replace(k, v)
    width = r"0.8\linewidth" if "own_who" in fname else r"\linewidth"
    if env == "wrapfigure":
        head = r"\begin{wrapfigure}{r}{0.5\linewidth}"
    else:
        env, _, place = env.partition("[")
        head = rf"\begin{{figure}}[{place.rstrip(']') or 'tbp'}]"
    return "\n".join([head, r"    \centering",
                      rf"    \includegraphics[width={width}]{{images/{fname}}}",
                      rf"    \caption{{{caption}}}", rf"    \label{{{label}}}",
                      rf"\end{{{env}}}", ""])

print("% ---------------- files: results/figures/manuscript/<source>  ->  images/<name>")
for _dst, _src in FILES.items():
    assert (OUTDIR / _src).exists(), _src
    print(f"%   {_src:42s} -> images/{_dst}")
print("%   tab_family.tex, tab_bigtb.tex             -> tables/ (or paste the blocks below)")
print()
print("% ---------------- main text")
if TAB_FAMILY:                   # also in results/figures/manuscript/tab_family.tex
    print(TAB_FAMILY); print()
print("% Table 2: place after tab:family so it numbers as Table 2")
print(TAB_BIGTB); print()        # also in results/figures/manuscript/tab_bigtb.tex
for b in BLOCKS:
    print(block(*b))
print("% ---------------- appendix")
for b in APPENDIX:
    print(block(*b))
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                                                  "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(cells)} cells)")

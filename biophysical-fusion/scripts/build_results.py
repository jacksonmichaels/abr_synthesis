"""Generate notebooks/results.ipynb — the presentation view of the project.

    python scripts/build_results.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb

`overview.ipynb` is the working view (every cell of the task x modality x model
cube). This one carries the findings in the order the paper makes them
(PAPER_OUTLINE.tex); everything the paper does not lean on is in the supplement
at the end.

Conventions, all enforced here rather than per figure:

* Plot style, table style and the paired tests live in `results_helpers.py`;
  the data comes from `results_frame.load_frame()` and `bigtb_baselines`.
* Figure titles name the subject and the slice, never the finding, and nothing
  explanatory is drawn inside a chart (see ~/.claude/CLAUDE.md).
* Tables set colour only on tinted cells, so plain cells inherit the reader's
  editor theme and stay legible in dark mode.
* Markdown is a short lead-in and a short read-out; the caveats are collected
  once at the end.
* `locusfusion_v2` numbers are computed live from whatever cells have finished;
  prose quotes only cells that are complete.
"""
import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("-o", "--output", type=Path, default=None,
                 help="notebook path (default: <project>/notebooks/results.ipynb)")
_args = _ap.parse_args()
OUT = (_args.output.expanduser().resolve() if _args.output
       else PROJECT / "notebooks" / "results.ipynb")

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.rstrip("\n").splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


# ================================================================== title
md("""
# Genotype → phenotype antibiotic resistance in *M. tuberculosis*

Per-drug resistance (R/S) for 11 antibiotics from gene sequence, 17,942 clinical
isolates, against the two published CNNs of the BIG-TB benchmark.

**The argument, in the paper's order.**

- An **L1-logistic lasso** on the raw alignment lands on `mdcnn` at 19 loci
  (0.9241 vs 0.9246, not significant) with 228 nonzero coefficients; at the
  drug's own loci it is behind (−0.034, 0/11) (§2).
- One architecture separates from it: **`locusfusion`**, 0.9343 macro CV AUC at
  19 loci and 0.66M parameters, +0.010 over the lasso and +0.012 over `mdcnn`
  (§3). The tokenizer fix behind it is +0.0095 on its own. Its second
  tokenizer, **`locusfusion_v2`**, ties it on DNA and protein and trails it
  wherever the regulatory modality is loaded (§4).
- What moves accuracy is **input, not capacity**. At 19 loci the locus set is
  the largest effect (§5), probably not causal (§6); at the drug's own loci the
  modalities are as large as the locus set (§7); parameter count shows no
  relationship with accuracy (§8).
- Of the paper's 14 comparisons, **11 are significant, 3 not** (§9).

**How numbers are judged.** 5-fold CV AUC, macro over the 11 shared drugs,
single seed. Test AUC is reported only because the papers publish it. Where the
prose says *significant* or *not significant*, that is §9's verdict: exact
Wilcoxon over the 55 fold differences, Holm over the paper's 14. Comparisons
outside that family are quoted with an unadjusted p.

| part | sections |
|---|---|
| **I** tasks, baselines and the linear control | §1–§2 |
| **II** `locusfusion`, and `locusfusion_v2` against it | §3–§4 |
| **III** what moves accuracy — loci, genome, modalities, parameters | §5–§8 |
| **IV** significance and caveats | §9–§10 |
| **Supplement** per-drug baselines, SHAP, the variant-set aggregators, coverage | S1–S6 |

> Generated — edit `scripts/build_results.py`, not the `.ipynb`.
""")

code('''
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from IPython.display import display, Image as IPImage
from scipy import stats

_here = Path.cwd()
PROJECT = next(p for p in [_here, *_here.parents] if (p / "bigtb_ref.py").exists())
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from results_frame import load_frame, TASKS, STD_MODSETS
from bigtb_baselines import SD_BASE_CV, MD_BASE_CV, ALL_DRUGS
from results_helpers import (SURFACE, INK, INK2, MUTED, AXIS, BLUE, ORANGE, RUST,
                             RUST_LIGHT, AQUA, PALE, TINTS, AQUA_RAMP, FAM_TINT,
                             NEMENYI_Q05, WilcoxonComparisons, fold_matrix,
                             PAPER_GROUPS, paper_family, plot_sig_by_claim,
                             label_points, style_table, tidy, tint)

frame, meta = load_frame()
# locusfusion has two single-drug runs -- `newmodels_full` (old tokenizer) and
# `locusfusion_fixed`. Averaging them would pool two different models into one
# cell, so only the fixed run is this notebook's locusfusion. Its joint cells
# exist on the old tokenizer only and stay. Dropped here, not in results_frame,
# because scripts/bigtb_paired_test.py reads the old run as its own arm.
_old_lf = ((frame.model == "locusfusion") & (frame.run == "newmodels_full")
           & (frame.task != "multi-drug"))
# kept apart, under its own name, for §9's tokenizer comparison only
frame_lf_old = frame[_old_lf].assign(model="locusfusion_old").reset_index(drop=True)
frame = frame[~_old_lf].reset_index(drop=True)
meta["n_rows"] = len(frame)

V1, V2 = "locusfusion", "locusfusion_v2"
AGGREGATORS = ["catalogue", "additive", "noisyor", "gatedpool", "deepsets", "fm"]
EXPERIMENTAL = set(AGGREGATORS) | {V1, V2}               # the variant-token family
BASELINE_MODELS = {"sparse_baseline", "sparse_baseline_trainvocab",
                   "lasso_onehot", "lasso_delta", "lasso_genome"}
ESTABLISHED = set(meta["MODELS"]) - EXPERIMENTAL - BASELINE_MODELS
EST3 = ["mdcnn", "cisfusion", "late_fusion"]
PD, L19, WG, MD = ("single-drug", "single-drug, all loci",
                   "single-drug, whole genome", "multi-drug")
A = "all_modalities"
MODS5 = STD_MODSETS
NET_TASKS = [t for t in TASKS if t != WG]               # the tasks networks ran


def family(model):
    if model in BASELINE_MODELS:
        return "baseline"
    return "experimental" if model in EXPERIMENTAL else "established"


def hue(model):                                          # colour follows identity
    if model in BASELINE_MODELS:
        return AQUA
    if model == V1:
        return RUST
    if model == V2:
        return RUST_LIGHT
    return ORANGE if model in EXPERIMENTAL else BLUE


frame["family"] = frame.model.map(family)

# One row per (task, modset, model) that finished all 11 drugs. Every figure
# slices this, so an incomplete cell is never averaged against a complete one.
best = (frame.groupby(["task", "modset", "model", "family"])
        .agg(cv=("cv", "mean"), n=("drug", "nunique"),
             params=("n_params", "mean"), loci=("n_loci", "max"))
        .reset_index().query("n == 11").sort_values("cv", ascending=False))


def top_of(fam, task=None, exclude=()):
    s = best[(best.family == fam) & ~best.model.isin(exclude)]
    if task:
        s = s[s.task == task]
    return None if s.empty else s.iloc[0]


def drug_cv(model, task, modset=None):
    """Per-drug CV AUC for one cell."""
    s = frame[(frame.model == model) & (frame.task == task)]
    if modset:
        s = s[s.modset == modset]
    return s.groupby("drug").cv.mean()


def macro_cell(model, task, modset=None):
    """Macro CV AUC for one cell -- NaN unless all 11 drugs finished."""
    v = drug_cv(model, task, modset)
    return v.mean() if len(v) == 11 else float("nan")


def best_cell(task, fam="established", modset=None):
    s = frame[(frame.family == fam) & (frame.task == task)]
    if modset:
        s = s[s.modset == modset]
    g = (s.groupby(["model", "modset"])
         .agg(cv=("cv", "mean"), n=("drug", "nunique")).query("n == 11")
         .sort_values("cv", ascending=False))
    return g.index[0], g.cv.iloc[0]


_pfmt = lambda v: "—" if pd.isna(v) else ("<0.0001" if v < 1e-4 else f"{v:.4f}")


def show_sig(sig, caption, adjusted=True):
    """A WilcoxonComparisons table, as §4 and §9 display it. `adjusted=False`
    shows the raw p and no verdict, for rows outside the paper's Holm family."""
    t = pd.DataFrame({
        "claim": sig.group, "comparison": sig.comparison,
        "macro Δ AUC": sig.delta.map("{:+.4f}".format),
        "HL Δ": sig.hl.map("{:+.4f}".format),
        "HL 95% CI": [f"[{a:+.4f}, {b:+.4f}]" for a, b in zip(sig.lo, sig.hi)],
        "drugs a > b": sig.wins,
    })
    if adjusted:
        t["Wilcoxon p Holm"], t["verdict"] = sig.p_holm, sig.verdict
        s = (t.style.format({"Wilcoxon p Holm": _pfmt})
             .map(lambda v: "font-weight:600" if v == "significant" else "font-style:italic",
                  subset=["verdict"]))
    else:
        t["Wilcoxon p (unadjusted)"] = sig.p
        s = t.style.format({"Wilcoxon p (unadjusted)": _pfmt})
    display(style_table(s, caption=caption))


SD_BASE, MD_BASE = SD_BASE_CV.mean(), MD_BASE_CV.mean()
FOLD_SD = frame[frame.task != MD].cv_sd.mean()
print(f"{meta['n_rows']:,} drug-runs · {meta['n_cells']} cells · "
      f"{len(meta['MODELS'])} models")
print(f"baselines: SD-CNN {SD_BASE:.4f}  MD-CNN {MD_BASE:.4f}   "
      f"mean per-drug fold SD {FOLD_SD:.4f}")
_v2 = frame[frame.model == V2].groupby(["task", "modset"]).drug.nunique()
print(f"locusfusion_v2: {int((_v2 == 11).sum())} of 15 task x modality cells "
      f"complete, {int(_v2.sum())} of 165 drug-runs")
''')

# ================================================================== headline
md("""
---
## Where the project stands

**SD-CNN** (one model per drug) and **MD-CNN** (13 drugs, one trunk) are the
published reference points; we reproduced MD-CNN at 0.9212 against its 0.9222.
Tiles are each family's best complete cell, which is a max over five modality
sets and so favours whichever model varies most — §3 and §9 pair them properly.
""")

code('''
est, exp, bas = top_of("established"), top_of("experimental"), top_of("baseline")
tiles = [
    ("Best ESTABLISHED cell", f"{est.cv:.4f}",
     f"{est.model} · {est.modset}" + chr(10) + f"{est.params/1e6:.1f}M params", BLUE),
    ("Best EXPERIMENTAL cell", f"{exp.cv:.4f}",
     f"{exp.model} · {exp.modset}" + chr(10) + f"{exp.params/1e6:.2f}M params", RUST),
    ("Best LINEAR cell", f"{bas.cv:.4f}",
     f"{bas.model} · L1-logistic" + chr(10) + "no GPU, no network", AQUA),
    ("Est. / exp. vs MD-CNN",
     f"{est.cv - MD_BASE:+.4f} / {exp.cv - MD_BASE:+.4f}",
     f"published baseline {MD_BASE:.4f}" + chr(10) + "all at 19 loci", AXIS),
]

fig, axes = plt.subplots(1, 4, figsize=(11.6, 2.0))
for ax, (label, value, sub, colour) in zip(axes, tiles):
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=PALE, edgecolor="none", zorder=0))
    ax.add_patch(plt.Rectangle((0, 0), 1, 0.035, transform=ax.transAxes,
                               facecolor=colour, edgecolor="none", zorder=1))
    ax.text(0.5, 0.82, label, ha="center", va="center", transform=ax.transAxes,
            fontsize=8.5, color=INK2)
    ax.text(0.5, 0.52, value, ha="center", va="center", transform=ax.transAxes,
            fontsize=20 if len(value) <= 6 else 13, color=INK)
    ax.text(0.5, 0.18, sub, ha="center", va="center", transform=ax.transAxes,
            fontsize=7, color=INK2, linespacing=1.6)
fig.suptitle("Macro CV AUC over the 11 shared drugs", y=1.10, fontsize=11,
             color=INK, fontweight="bold")
plt.tight_layout(); plt.show()

print(f"{exp.model} is {est.params / exp.params:.0f}x smaller than {est.model} "
      f"for {exp.cv - est.cv:+.4f} of AUC")
''')

# ==================================================================
# ============================ PART I ==============================
# ==================================================================
md("""
---
# Part I — Tasks, baselines and the linear control

## 1. The three tasks

One model per drug on its own 2–3 genes, one per drug on all 19 curated loci,
and one model with 11 outputs. Bars are each established model's best modality
set, plus the best variant-token cell for reference; the rule is that task's
published baseline.
""")

code('''
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharex=True)
_est = best[best.family == "established"]
# same row height in every panel, so a 4-model panel does not draw bars three
# times as thick as a 13-model one
_rows = max(_est[_est.task == t].model.nunique() for t in NET_TASKS) + 1
for ax, task in zip(axes, NET_TASKS):
    s = (_est[_est.task == task].sort_values("cv", ascending=False)
         .groupby("model", as_index=False).first())
    ref = top_of("experimental", task)
    if ref is not None:
        s = pd.concat([s, ref.to_frame().T], ignore_index=True)
    s["label"] = s.model + "   ·" + s.modset
    s = s.sort_values("cv")
    base = SD_BASE if task == PD else MD_BASE
    name = "SD-CNN" if task == PD else "MD-CNN"
    y = np.arange(len(s))
    ax.barh(y, s.cv, height=0.68, color=[hue(m) for m in s.model],
            edgecolor="none", zorder=2)
    for yi, v in zip(y, s.cv):                       # value labels outside the bar
        ax.text(v + 0.008, yi, f"{v:.4f}", va="center", ha="left",
                fontsize=7, color=INK)
    ax.axvline(base, color=INK2, lw=1.2, zorder=3)
    ax.text(base, -1.15, f"{name} {base:.4f} ", color=INK2, fontsize=7.5,
            va="center", ha="right")
    ax.set_yticks(y); ax.set_yticklabels(s.label, fontsize=7.5)
    for tick, m in zip(ax.get_yticklabels(), s.model):
        if m not in ESTABLISHED:
            tick.set_color(RUST)
    ax.set_xlim(0.5, 1.06); ax.set_ylim(-1.6, _rows - 0.4)
    ax.set_title(task)
    tidy(ax)
axes[0].set_xlabel("macro CV AUC")
fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, RUST)],
           ["established architectures", "variant-token models"],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Macro CV AUC by task and architecture, complete cells only (11/11 drugs)",
             y=1.00, fontsize=11.5, color=INK, fontweight="bold")
plt.tight_layout(); plt.show()
''')

md("""
- **Locus set, not multi-task sharing.** "Joint models win" was confounded:
  joint runs saw 19 loci, single-drug runs 2–3. Matched at 19, joint is +0.001
  (5 of 11 drugs), not significant (Wilcoxon over drugs, p = 0.83).
- **The `+tf` trunks lose everywhere** — −0.059 for `mdcnn+tf`, 0 of 11 drugs,
  p < 0.0001 unadjusted (S2 has the reason).
- Per drug, the best established cell is ahead of SD-CNN on all 11 (S1).
""")

# ---------------------------------------------------------------- lasso
md("""
---
## 2. A lasso on the same alignment

Every comparison in this literature is between networks. The control none of
them ran: **L1-logistic regression**, no GPU, seconds to minutes per drug.
Protocol is `run_modal_cv`'s step for step — same drugs, same 80/20 split at
seed 42, same `StratifiedKFold(5, random_state=42)`, `class_weight="balanced"`
for the networks' inverse-frequency alpha, C swept over eight values. Two runs:
`lasso_wholeseq_20260901` on the same one-hot alignment the networks read, and
`lasso_wholegenome_20260903` on every variant called in the 4.41 Mb genome (§6).
""")

code('''
(_npd_dna_m, _npd_dna), (_npd_m, _npd) = best_cell(PD, modset="dna"), best_cell(PD)
(_n19_dna_m, _n19_dna), (_n19_m, _n19) = best_cell(L19, modset="dna"), best_cell(L19)
(_x19_m, _x19) = best_cell(L19, fam="experimental")

ladder = [
    ("lasso, delta encoding, per-drug loci", macro_cell("lasso_delta", PD), AQUA),
    ("lasso, onehot encoding, per-drug loci", macro_cell("lasso_onehot", PD), AQUA),
    ("L1 on variant tokens, per-drug loci", macro_cell("sparse_baseline", PD), AQUA),
    (f"{_npd_dna_m[0]}, dna, per-drug loci", _npd_dna, BLUE),
    (f"{_npd_m[0]}, {_npd_m[1]}, per-drug loci", _npd, BLUE),
    ("lasso, delta encoding, 19 loci", macro_cell("lasso_delta", L19), AQUA),
    (f"{_n19_dna_m[0]}, dna, 19 loci", _n19_dna, BLUE),
    ("lasso, onehot encoding, 19 loci", macro_cell("lasso_onehot", L19), AQUA),
    (f"{_n19_m[0]}, {_n19_m[1]}, 19 loci", _n19, BLUE),
    (f"{_x19_m[0]}, {_x19_m[1]}, 19 loci", _x19, RUST),
    ("lasso, delta encoding, whole genome", macro_cell("lasso_genome", WG), AQUA),
]
ladder = sorted([r for r in ladder if r[1] == r[1]], key=lambda r: r[1])

fig, ax = plt.subplots(figsize=(9.8, 5.0))
y = np.arange(len(ladder))
ax.barh(y, [r[1] for r in ladder], height=0.66, zorder=2,
        color=[r[2] for r in ladder], edgecolor="none")
for yi, (_, v, _c) in zip(y, ladder):
    ax.text(v + 0.0020, yi, f"{v:.4f}", va="center", ha="left", fontsize=7.4,
            color=INK)
for base, name in ((SD_BASE, "SD-CNN"), (MD_BASE, "MD-CNN")):
    ax.axvline(base, color=INK2, lw=1.1, zorder=3)
    ax.text(base, len(ladder) - 0.35, f" {name} {base:.4f}", color=INK2,
            fontsize=7.5, va="center", ha="left")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in ladder], fontsize=8)
for tick, r in zip(ax.get_yticklabels(), ladder):
    tick.set_color(INK if r[2] == AQUA else INK2)
ax.set_xlim(0.85, 0.945); ax.set_ylim(-0.7, len(ladder) - 0.2)
ax.set_xlabel("macro CV AUC")
ax.set_title("Macro CV AUC by model, encoding and locus universe, 11 shared drugs",
             loc="left")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (AQUA, BLUE, RUST)],
          ["L1-logistic", "established architectures", "variant-token models"],
          loc="lower right")
tidy(ax); plt.tight_layout(); plt.show()
''')

md("""
**The lasso lands on the best established cell**: 0.9241 against `mdcnn`'s
0.9246, **not significant** (−0.0005, 5 of 11 drugs, Holm p = 0.95) — no detected
difference, not demonstrated parity. One is a 6.7M-parameter CNN over four
modalities; the other is 228 nonzero coefficients out of 22,384 columns, DNA
only. Against the published MD-CNN it is +0.002 (6 of 11 drugs). It does not
reach the best model overall — `locusfusion` is +0.010 over it on 10 of 11
drugs, **significant** (§3).

- **Networks earn their parameters on narrow inputs.** At per-drug loci the
  lasso loses by −0.034 (0 of 11, **significant**); at 19 loci it draws. The
  per-drug cells run 94–5,723 columns against 22,384.
- **The locus universe is worth more to the lasso**: +0.049 (`onehot`) and
  +0.054 (`delta`), against +0.016 for the networks (§5).
""")

code('''
_oh19, _dl19 = drug_cv("lasso_onehot", L19), drug_cv("lasso_delta", L19)
_ohpd, _dlpd = drug_cv("lasso_onehot", PD), drug_cv("lasso_delta", PD)
(_nm, _ns), _ = best_cell(L19)
_net19 = drug_cv(_nm, L19, _ns)
_n = frame[frame.task == PD].groupby("drug").n_valid.max()

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0),
                         gridspec_kw={"width_ratios": [1.08, 1]})

# -- left: per drug, lasso against the best network cell, both at 19 loci
ax = axes[0]
d = (_oh19 - _net19).sort_values()
y = np.arange(len(d))
for yi, drug in zip(y, d.index):
    ax.plot([_net19[drug], _oh19[drug]], [yi, yi], color=AXIS, lw=1.5, zorder=1,
            solid_capstyle="round")
ax.scatter(_net19[d.index], y, s=56, color=BLUE, zorder=3, edgecolor=SURFACE,
           linewidth=1.4, label=f"{_nm}, {_ns}")
ax.scatter(_oh19[d.index], y, s=56, color=AQUA, zorder=3, edgecolor=SURFACE,
           linewidth=1.4, label="lasso, onehot encoding")
for yi, drug in zip(y, d.index):
    ax.text(max(_oh19[drug], _net19[drug]) + 0.004, yi, f"{d[drug]:+.4f}",
            va="center", ha="left", fontsize=7, color=INK2)
ax.set_yticks(y)
ax.set_yticklabels([f"{drug}   n={int(_n[drug]):,}" for drug in d.index], fontsize=7.8)
ax.set_xlim(0.78, 1.02); ax.set_xlabel("CV AUC")
ax.set_title("Per drug at 19 loci: lasso against the best established cell", loc="left")
ax.legend(loc="lower left")
tidy(ax)

# -- right: what the encoding is worth, at both locus universes
ax = axes[1]
enc = pd.DataFrame({"per-drug loci": _ohpd - _dlpd,
                    "19 loci": _oh19 - _dl19}).sort_values("19 loci")
y = np.arange(len(enc)); w = 0.36
for i, (col, colour) in enumerate((("per-drug loci", AQUA_RAMP[2]),
                                   ("19 loci", AQUA_RAMP[0]))):
    ax.barh(y + (i - 0.5) * (w + 0.04), enc[col], height=w, color=colour,
            edgecolor="none", zorder=2, label=col)
ax.axvline(0, color=AXIS, lw=1.0, zorder=3)
ax.set_yticks(y); ax.set_yticklabels(enc.index, fontsize=7.8)
ax.set_xlabel("CV AUC, onehot encoding − delta encoding")
ax.set_title("Encoding effect by drug, onehot minus delta", loc="left")
ax.legend(loc="lower right")
tidy(ax)
plt.tight_layout(); plt.show()

for name, aa, bb in (("19 loci ", _oh19, _dl19), ("per-drug", _ohpd, _dlpd)):
    print(f"onehot − delta, {name}: {(aa - bb).mean():+.4f}   "
          f"{(aa > bb).sum()}/11 drugs")
print(f"lasso − {_nm}, 19 loci : {(_oh19 - _net19).mean():+.4f}   "
      f"{(_oh19 > _net19).sum()}/11 drugs")
''')

md("""
**Left.** All eleven per-drug margins are inside ±0.01 (widest: KANAMYCIN
−0.0096, STREPTOMYCIN +0.0095), so the near-zero macro is not two offsetting
errors. ETHIONAMIDE, the project's worst drug, reaches 0.8118 — level with
`mdcnn`'s best cell (0.8123, which needs the biophysical modality). Its largest
`fabG1` coefficients are the promoter positions §7 attributes ETHIONAMIDE's
modality gain to, reached from the coding alignment with no regulatory modality
loaded.

**Right: encoding matters.** `onehot` and `delta` span the same column space and
differ only in parametrisation, but **L1 is not invariant to that**: keeping the
reference level as its own penalised column is worth +0.009 at 19 loci and
+0.014 at per-drug loci, 11 of 11 drugs both times (**significant** at 19 loci;
p < 0.0001 unadjusted at per-drug loci). Every variant-token model is built on
`--delta`, so that choice is not free for a sparsely-penalised model; it does
not transfer automatically to the networks, which are not L1-penalised.
""")

# ==================================================================
# ============================ PART II =============================
# ==================================================================
md("""
---
# Part II — `locusfusion`

## 3. The one architecture that separates

SHAP (S2) says a model reads a handful of columns, and the input is nearly
constant: the median isolate differs from H37Rv at 14 columns across all 19
loci. `locusfusion` is built on that. It runs on reference-difference input
(`--delta`), emits one token per deviating column, and **fuses at the gene, then
across genes**: all of *rpoB*'s evidence (CDS one-hot, translation, biophysical
profile, promoter window) becomes one locus representation, and only those 19
talk to each other. A susceptible locus is a learned per-locus null token alone;
exact coordinates survive through a sinusoidal encoding. The aggregator
experiments behind these choices are S3–S5.
""")

code('''
# read off a real LocusFusionNet: `python scripts/locusfusion_diagram.py --check`
# asserts its parameter total against the 19-locus run
_diagram = PROJECT / "results" / "figures" / "locusfusion_shapes.png"
if _diagram.is_file():
    display(IPImage(filename=str(_diagram)))
else:
    print("not built — run: python scripts/locusfusion_diagram.py --check")
''')

code('''
lf = best[best.model == V1]
piv = (lf.pivot_table(index="modset", columns="task", values="cv")
       .reindex(index=[m for m in STD_MODSETS], columns=[t for t in TASKS
                                                         if t in lf.task.values]))

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4),
                         gridspec_kw={"width_ratios": [1.25, 1]})

# -- left: every task x modality cell it has run
ax = axes[0]
x = np.arange(len(piv.index)); w = 0.24
for i, task in enumerate(piv.columns):
    ax.bar(x + (i - (len(piv.columns) - 1) / 2) * (w + 0.045), piv[task],
           width=w, color=TINTS[i], edgecolor="none", label=task, zorder=2)
ax.set_xticks(x); ax.set_xticklabels(piv.index, fontsize=8)
ax.set_ylim(0.85, 0.945); ax.set_ylabel("macro CV AUC")
ax.axhline(MD_BASE, color=INK2, lw=1.1, zorder=3)
ax.text(-0.45, MD_BASE, f"MD-CNN {MD_BASE:.4f}", color=INK2, fontsize=7.5,
        va="bottom", ha="left")
ax.set_title("locusfusion: macro CV AUC by modality set and task", loc="left")
ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.30))
tidy(ax, xgrid=False, ygrid=True)

# -- right: matched cell by cell against the best established model. A max over
# five modality sets rewards variance (locusfusion spreads 0.003, mdcnn 0.008),
# so pairing on the modality set is the comparison that means something.
ax = axes[1]
_rival = top_of("established", L19).model
_paired = (frame[(frame.task == L19) & frame.model.isin([V1, _rival])]
           .groupby(["modset", "model"]).cv.mean().unstack().dropna()
           .reindex(STD_MODSETS))
yy = np.arange(len(_paired))
for yi, (ms, r) in zip(yy, _paired.iterrows()):
    ax.plot([r[_rival], r[V1]], [yi, yi], lw=2, zorder=2,
            color=RUST if r[V1] > r[_rival] else AXIS, solid_capstyle="round")
    ax.scatter([r[_rival]], [yi], s=42, color=BLUE, zorder=3, edgecolor=SURFACE,
               linewidth=1.4)
    ax.scatter([r[V1]], [yi], s=48, color=RUST, zorder=4,
               edgecolor=SURFACE, linewidth=1.4)
    ax.text(max(r[V1], r[_rival]) + 0.0006, yi,
            f"{r[V1] - r[_rival]:+.4f}", va="center", ha="left",
            fontsize=7.5, color=INK)
ax.axvline(MD_BASE, color=INK2, lw=1.0, zorder=1)
ax.text(MD_BASE, -0.95, f"MD-CNN {MD_BASE:.4f} ", color=INK2, fontsize=7,
        va="center", ha="right")
ax.set_yticks(yy); ax.set_yticklabels(_paired.index, fontsize=8)
ax.set_ylim(-1.4, len(_paired) - 0.4)
ax.set_xlim(min(_paired.min().min(), MD_BASE) - 0.0015, _paired.max().max() + 0.0035)
ax.set_xlabel("macro CV AUC — single-drug, all 19 loci")
ax.set_title(f"locusfusion vs. {_rival}, paired by modality set", loc="left")
ax.legend([Line2D([0], [0], marker="o", lw=0, markersize=7, color=c)
           for c in (RUST, BLUE)], ["locusfusion", _rival], loc="upper left")
tidy(ax)
plt.tight_layout(); plt.show()

_all = (frame[(frame.task == L19) & frame.modset.isin(STD_MODSETS)]
        .groupby(["model", "modset"]).cv.mean().unstack()
        .reindex([V1, *EST3]).dropna(how="all"))
print(f"{'model':<14}{'mean of 5':>11}{'max of 5':>10}{'spread':>9}   vs locusfusion")
for mdl, row in _all.iterrows():
    wins = ("" if mdl == V1 else
            f"   ahead on {int((_all.loc[V1] > row).sum())}"
            f"/{int(row.notna().sum())} cells")
    print(f"{mdl:<14}{row.mean():>11.4f}{row.max():>10.4f}"
          f"{row.max() - row.min():>9.4f}{wins}")
print(f"{chr(10)}locus set: {piv[PD].max():.4f} at per-drug loci -> "
      f"{piv[L19].max():.4f} at 19")
''')

md("""
1. **Ahead on all five matched cells**, at a tenth of `mdcnn`'s parameters:
   **significant** against the lasso (+0.010, 10/11) and `mdcnn` (+0.012,
   10/11), and ahead of `cisfusion` (+0.016) and `late_fusion` (+0.028), 11/11
   each, p < 0.0001 unadjusted.
2. **DNA alone gets nearly all of it.** Spread across the five modality sets is
   0.003; `all_modalities` − `dna` is +0.003 (10/11, p = 0.0002 unadjusted) —
   detectable, but a fifth of what the modalities give the established models.
   The token already carries the substituted residue and its biophysical
   consequence — S4's featurisation result as an architecture property.
3. **The locus count is what moves it**: +0.029 over the matched cells, 11 of 11
   drugs, **significant** — nearly twice what it gives the established models (§5).

At per-drug loci it is +0.004 over `mdcnn` (8/11, p = 0.0015 unadjusted). Test
AUC is 0.9191 against `mdcnn`'s 0.9109. Joint `locusfusion` cells are the old
tokenizer and do not describe this model.
""")

# ---------------------------------------------------------------- v2
md("""
---
## 4. `locusfusion_v2` against `locusfusion`

Same two-stage architecture and parameter count (0.645M vs 0.645M at per-drug
loci, 0.658M vs 0.659M at 19), second tokenizer. What changes is what a token
*is* (`models/README.md`, `notebooks/tokenization_v2.ipynb`):

| | `locusfusion` (v1) | `locusfusion_v2` |
|---|---|---|
| what makes a token | a deviating column in any stream | a deviating DNA column only; protein and biophysical ride on it |
| token budget per locus | 16 per stream (up to 1 + 48) | 16 per locus (1 + 16) |
| promoter and CDS | separate streams | one axis, start codon at 0, promoter negative |
| codon of a column | approximated | counted from H37Rv bases, genome-anchored |
| translation | isolate frame | reference frame, STOP as a 21st residue |
| per-locus summary token | `[WT]` | `[NULL]`, same mechanism |
| flags | `is_nt`/`is_aa`/`is_reg`, `is_gap`, `uncovered` | dropped |

Paired on the drugs both have finished; the grid is still running, so the
19-locus and joint numbers below are interim until every count reads 11/11.
""")

code('''
_lv = frame[frame.model.isin([V1, V2]) & frame.modset.isin(STD_MODSETS)
            & frame.task.isin([PD, L19])]
_rows = []
for task in [PD, L19]:
    for ms in STD_MODSETS:
        p = (_lv[(_lv.task == task) & (_lv.modset == ms)]
             .pivot_table(index="drug", columns="model", values="cv"))
        if V1 not in p or V2 not in p:
            continue
        p = p.dropna()
        _rows.append(dict(task=task, modset=ms, drugs=len(p), v1=p[V1].mean(),
                          v2=p[V2].mean(), ahead=int((p[V2] > p[V1]).sum())))
vv = pd.DataFrame(_rows)
vv["delta"] = vv.v2 - vv.v1

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0),
                         gridspec_kw={"width_ratios": [1, 1]})

# -- left: every cell v2 has started, v1 on the same drugs
ax = axes[0]
_f = vv.iloc[::-1].reset_index(drop=True)
y = np.arange(len(_f))
for yi, r in _f.iterrows():
    ax.plot([r.v1, r.v2], [yi, yi], color=AXIS, lw=1.8, zorder=1,
            solid_capstyle="round")
    ax.text(max(r.v1, r.v2) + 0.0015, yi, f"{r.delta:+.4f}", va="center",
            ha="left", fontsize=7.2, color=INK)
ax.scatter(_f.v1, y, s=50, color=RUST, zorder=3, edgecolor=SURFACE, linewidth=1.4,
           label="locusfusion")
ax.scatter(_f.v2, y, s=50, color=RUST_LIGHT, zorder=3, edgecolor=SURFACE,
           linewidth=1.4, label="locusfusion_v2")
ax.set_yticks(y)
ax.set_yticklabels([f"{'19 loci' if r.task == L19 else 'per-drug loci'} · "
                    f"{r.modset}   ({r.drugs}/11)" for r in _f.itertuples()],
                   fontsize=7.8)
ax.set_xlim(min(_f.v1.min(), _f.v2.min()) - 0.004, max(_f.v1.max(), _f.v2.max()) + 0.009)
ax.set_xlabel("macro CV AUC over the drugs both finished")
ax.set_title("locusfusion vs. locusfusion_v2 by task and modality set", loc="left")
ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.45, -0.11))
tidy(ax)

# -- right: per drug, v2 − v1, averaged over the modality sets v2 has finished
ax = axes[1]
_p = (_lv.pivot_table(index=["task", "drug", "modset"], columns="model", values="cv")
      .dropna())
_dd = (_p[V2] - _p[V1]).groupby(level=["task", "drug"]).agg(["mean", "count"])
_pd = _dd.loc[PD]
order = _pd["mean"].sort_values().index
y = np.arange(len(order))
ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
ax.scatter(_pd["mean"][order], y, s=50, color=RUST_LIGHT, zorder=3,
           edgecolor=SURFACE, linewidth=1.4, label="per-drug loci, 5 modality sets")
if L19 in _dd.index.get_level_values(0):
    _l = _dd.loc[L19].reindex(order)
    full = (_l["count"] == 5).values
    ax.scatter(_l["mean"][full], y[full], s=48, marker="D", color=RUST_LIGHT,
               zorder=4, edgecolor=SURFACE, linewidth=1.4,
               label="19 loci, 5 modality sets")
    ax.scatter(_l["mean"][~full], y[~full], s=44, marker="D", color=SURFACE,
               zorder=4, edgecolor=RUST_LIGHT, linewidth=1.6,
               label="19 loci, fewer modality sets")
ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8)
ax.set_xlabel("CV AUC, locusfusion_v2 − locusfusion")
ax.set_title("locusfusion_v2 − locusfusion by drug", loc="left")
ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.45, -0.11))
tidy(ax)
plt.tight_layout(); plt.show()

print(vv.assign(v1=vv.v1.round(4), v2=vv.v2.round(4), delta=vv.delta.round(4))
        [["task", "modset", "drugs", "v1", "v2", "delta", "ahead"]].to_string(index=False))
''')

md("""
Tested as in §9, one row per modality set, as **its own Holm family** — which
tokenizer to carry forward is a development question, not one of the paper's
comparisons. A row appears once both sides have all 11 drugs.
""")

code('''
v2cmp, _pending = WilcoxonComparisons(frame, ALL_DRUGS), []
# one row per modality set, against v1 and mdcnn, wherever both sides have 11 drugs
for task, where in [(PD, "per-drug loci"), (L19, "19 loci")]:
    for ms in MODS5:
        for ref in (V1, "mdcnn"):
            label = f"locusfusion_v2 − {ref} · {where} · {ms}"
            try:
                v2cmp.folds("v2 tokenizer", label, (task, V2, ms), (task, ref, ms))
            except ValueError:             # a side is still short of 11 drugs
                _pending.append(label)
if v2cmp.rows:
    sig_v2 = v2cmp.table()
    show_sig(sig_v2, caption=f"{len(sig_v2)} v2 comparisons · Wilcoxon signed-rank "
                             "over 55 fold differences · Holm within this family only")
print("pending (grid still running): " + ("; ".join(_pending) or "none"))
''')

md("""
**v2 ties v1 on DNA and protein, and trails it wherever `regulatory` is
loaded.** At per-drug loci `dna` is +0.0005 (7/11) and `dna_protein` +0.0000
(4/11), both not significant; `dna_regulatory` is −0.017 and `all_modalities`
−0.017, 0/11 each, significant. The first v2 tokenizer (archived, pre-fix)
scored 0.8920 and 0.8926 in those two regulatory cells; the retokenized v2
scores 0.8918 and 0.8924, so the fix recovered the DNA and protein cells and left
the regulatory ones untouched.

Not measured: why. Promoter columns now share a locus's 16-token cap with its
coding columns, which is one candidate. `dna_biophysical` has 8 of 11 drugs, the
19-locus grid only `dna` on 5, and no v2 job is running.
""")

# ==================================================================
# ============================ PART III ============================
# ==================================================================
md("""
---
# Part III — What moves accuracy

## 5. The locus set

Same models, modalities and schedule; only the gene set changes. Paired on
(drug, modality set, model), established models.
""")

code('''
_e = frame[frame.family == "established"]
m = (_e[_e.task == PD].merge(_e[_e.task == L19], on=["drug", "modset", "model"],
                             suffixes=("_a", "_b")))
gp = m.groupby("drug")
t = pd.DataFrame({"per-drug loci": gp.cv_a.mean(), "all 19 loci": gp.cv_b.mean()})
t["delta"] = t["all 19 loci"] - t["per-drug loci"]
t["n_lo"] = gp.n_loci_a.max()
t = t.sort_values("delta")

fig, ax = plt.subplots(figsize=(8.6, 5.4))
y = np.arange(len(t))
call = {"KANAMYCIN", "LEVOFLOXACIN"}
for yi, (d, r) in zip(y, t.iterrows()):
    c = ORANGE if d in call else BLUE
    ax.plot([r["per-drug loci"], r["all 19 loci"]], [yi, yi], color=c, lw=2,
            solid_capstyle="round", zorder=2)
    ax.scatter([r["per-drug loci"]], [yi], s=26, color=SURFACE, edgecolor=c,
               linewidth=1.6, zorder=3)
    ax.scatter([r["all 19 loci"]], [yi], s=42, color=c, zorder=3,
               edgecolor=SURFACE, linewidth=1.2)
    ax.text(max(r["per-drug loci"], r["all 19 loci"]) + 0.006, yi,
            f"{r['delta']:+.3f}", va="center", ha="left", fontsize=7.5, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([f"{d}   ({int(t.n_lo[d])} → 19)" for d in t.index], fontsize=8)
ax.set_xlim(0.66, 1.02); ax.set_xlabel("CV AUC")
ax.set_title("CV AUC at per-drug gene map vs. all 19 loci, established models",
             loc="left")
ax.legend([Line2D([0], [0], color=BLUE, lw=2, marker="o",
                  markerfacecolor=SURFACE, markersize=5),
           Line2D([0], [0], color=ORANGE, lw=2)],
          ["hollow = per-drug loci, filled = all 19 loci", "discussed below"],
          loc="lower center", bbox_to_anchor=(0.5, -0.19), ncol=2)
tidy(ax); plt.tight_layout(); plt.show()

print(f"macro over the paired drugs: {t.delta.mean():+.4f}")
for mdl, task_ms in (("lasso_onehot", "dna"), (V1, A)):
    a, b = drug_cv(mdl, L19, task_ms), drug_cv(mdl, PD, task_ms)
    print(f"{mdl:<13} 19 − per-drug loci: {(a - b).mean():+.4f}   {(a > b).sum()}/11 drugs")
''')

md("""
**The largest effect at 19 loci.** +0.016 for the established models (10/11),
+0.029 for `locusfusion` (11/11) and +0.049 for the lasso (11/11) — all three
**significant**. Per architecture (unadjusted): `mdcnn` +0.022, `cisfusion`
+0.019, and `late_fusion` +0.006 (9/11, p = 0.0002).

Two rows mislead. **KANAMYCIN gains most** because its per-drug map is the
single gene *rrs*, so it was also missing the *eis* promoter, its best-known
mechanism. **LEVOFLOXACIN's loss is noise**: 269 isolates, fold SD ~0.037.
""")

# ---------------------------------------------------------------- genome
md("""
---
## 6. Is the locus gain causal? The whole genome

The 19 curated loci are 0.9% of the genome and every one is a resistance gene
for some drug — so from inside that input neither "is curation leaving signal
outside" nor "does landing on resistance genes mean anything" is answerable.
Genome-wide, both are: the same L1-logistic on every `(position, ref, alt)` key
called in the cohort, 28k columns for LEVOFLOXACIN up to 899k for RIFAMPICIN.
Genome-wide can only be a `delta` arm, so its matched control is the 19-locus
`delta` cell (0.9152), not §2's `onehot` headline.
""")

code('''
_gen = drug_cv("lasso_genome", WG)
_who_f = PROJECT / "results/experiments/lasso_wholegenome_20260903/who_mass.csv"

fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))

# -- left: whole genome against its matched 19-locus control, same encoding
ax = axes[0]
d = (_gen - _dl19).sort_values()
y = np.arange(len(d))
for yi, drug in zip(y, d.index):
    ax.plot([_dl19[drug], _gen[drug]], [yi, yi], color=AXIS, lw=1.5, zorder=1,
            solid_capstyle="round")
ax.scatter(_dl19[d.index], y, s=54, color=AQUA_RAMP[2], zorder=3,
           edgecolor=SURFACE, linewidth=1.4, label="19 curated loci, 39,646 nt")
ax.scatter(_gen[d.index], y, s=54, color=AQUA_RAMP[0], zorder=3,
           edgecolor=SURFACE, linewidth=1.4, label="whole genome, 4.41 Mb")
for yi, drug in zip(y, d.index):
    ax.text(max(_gen[drug], _dl19[drug]) + 0.004, yi, f"{d[drug]:+.4f}",
            va="center", ha="left", fontsize=7, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(d.index, fontsize=7.8)
ax.set_xlim(0.78, 1.02); ax.set_xlabel("CV AUC")
ax.set_title("Whole genome against the 19 curated loci, delta encoding", loc="left")
ax.legend(loc="lower left")
tidy(ax)

# -- right: where the selected model's coefficient mass sits (WHO Table 21),
# as one hue most-specific -> least rather than three unrelated fills
ax = axes[1]
if _who_f.is_file():
    w = (pd.read_csv(_who_f).set_index("drug")
         .rename(columns={"this drug %": "this drug", "other drug %": "other drug",
                          "outside %": "outside"}))
    w = w[["this drug", "other drug", "outside"]].sort_values("outside")
    y = np.arange(len(w)); left = np.zeros(len(w))
    for col, colour in zip(w.columns, AQUA_RAMP):
        ax.barh(y, w[col], left=left, height=0.66, color=colour, zorder=2,
                edgecolor=SURFACE, linewidth=1.0, label=col)
        left += w[col].values
    ax.set_yticks(y); ax.set_yticklabels(w.index, fontsize=7.8)
    ax.set_xlim(0, 100); ax.set_xlabel("share of |coefficient| mass (%)")
    ax.set_title("Coefficient mass by WHO Table-21 membership", loc="left")
    ax.legend(loc="lower right", ncol=3, bbox_to_anchor=(1.0, -0.28))
    tidy(ax)
else:
    ax.axis("off")
    ax.text(0.5, 0.5, "who_mass.csv not built", ha="center", va="center",
            fontsize=8, color=MUTED)
plt.tight_layout(); plt.show()

print(f"whole genome − 19 loci (delta) : {(_gen - _dl19).mean():+.4f}   "
      f"{(_gen > _dl19).sum()}/11 drugs")
print(f"whole genome − 19 loci (onehot): {(_gen - _oh19).mean():+.4f}   "
      f"{(_gen > _oh19).sum()}/11 drugs")
''')

md("""
**1. The other 99.1% of the genome buys nothing detectable.** +0.0022 macro, 6 of
11 drugs, **not significant** (Holm p = 0.71; HL CI −0.004 to +0.007) — no
detected gain, not a demonstrated zero. Returns to more sequence ended well
before the genome did: 2–3 loci → 19 was +0.05 for this model, 19 loci → 4.41 Mb
is +0.002. The macro is a cancellation: AMIKACIN +0.040 (five folds, which cannot
reach significance), LEVOFLOXACIN −0.030 with 22 nonzero coefficients on 269
isolates.

**2. Most of the weight is outside the WHO catalogue.** Against Table 21 (tier
1 + 2, flanking intergenic regions counted as inside): **62.4% of |coefficient|
mass sits outside**, 27.1% on the drug's own genes, 10.6% on another drug's.
The model is not ignoring biology — ETHIONAMIDE's single largest coefficient out
of 127,697 is `1673425:C>T`, the *fabG1–inhA* promoter, found with no curation —
but the rest lands on *PE_PGRS* / *PPE* / *esx* repeats, the standard lineage
markers. At 19 loci the lasso's top ISONIAZID coefficients include `rpoB:1418`,
and its PYRAZINAMIDE picks `embB:319` and `embA:403` (alignment columns, not
codons).

**What that licenses.** First-line resistance predicts second-line resistance,
and lineage correlates with both, so extra loci import correlated signal: §5's
gain is **probably not causal**. A coefficient-mass split is not a variance
decomposition, and population structure and calling artifacts (PE/PPE/esx is
where short-read calling is least reliable) are both open. It is visible here
only because a lasso names its columns; every network in this notebook reads the
same lineage-correlated loci.
""")

# ---------------------------------------------------------------- modalities
md("""
---
## 7. The modalities

Each bar is a drug's best single-added-modality cell minus its own DNA-only
cell, at matched loci and architecture (established models, per-drug loci).
""")

code('''
sd = frame[(frame.task == PD) & frame.model.isin(EST3)]
dna = sd[sd.modset == "dna"].groupby(["drug", "model"]).cv.mean()
gain, which = {}, {}
for d in ALL_DRUGS:
    rows = sd[(sd.drug == d) & (sd.modset != "dna")]
    per = rows.groupby(["modset", "model"]).cv.mean()
    deltas = {}
    for (ms, mo), v in per.items():
        # single-added sets only, so each label names one modality
        if (d, mo) in dna.index and ms in ("dna_protein", "dna_biophysical",
                                           "dna_regulatory"):
            deltas.setdefault(ms, []).append(v - dna[(d, mo)])
    if deltas:
        ms, vals = max(deltas.items(), key=lambda kv: np.mean(kv[1]))
        gain[d], which[d] = np.mean(vals), ms.replace("dna_", "")

g = pd.Series(gain).sort_values()
y = np.arange(len(g))
promoter = {"ETHIONAMIDE", "ISONIAZID"}

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.barh(y, g.values, height=0.66, edgecolor="none", zorder=2,
        color=[ORANGE if d in promoter else BLUE for d in g.index])
ax.axvline(0, color=AXIS, lw=1)
for yi, (d, v) in zip(y, g.items()):
    ax.text(v + (0.0022 if v >= 0 else -0.0022), yi, f"{v:+.3f}   {which[d]}",
            va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color=INK)
ax.set_yticks(y); ax.set_yticklabels(g.index, fontsize=8.5)
ax.set_xlim(min(g.min() * 1.6, -0.01), g.max() * 1.55)
ax.set_xlabel("best single added modality minus that drug's DNA-only cell (CV AUC)")
ax.set_title("CV AUC gain from one added modality, by drug", loc="left")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=ORANGE)], ["promoter-mechanism drugs"],
          loc="lower center", bbox_to_anchor=(0.5, -0.20))
tidy(ax); plt.tight_layout(); plt.show()

print(f"macro {g.mean():+.4f}   median {g.median():+.4f}   "
      f"five drugs at or below {g.nsmallest(5).max():+.4f}")
''')

md("""
**At the drug's own loci, as large as the locus effect; concentrated in a few
drugs, where the mechanism predicts.** ETHIONAMIDE and ISONIAZID gain from
`regulatory` — both are driven by the *fabG1–inhA* promoter (`c-15t`), invisible
to a CDS one-hot. PYRAZINAMIDE gains from `protein`/`biophysical` (*pncA* loss of
function). KANAMYCIN and AMIKACIN gain nothing, and should not: *rrs* is 16S rRNA,
with no protein product.

The macro all-modalities gain over DNA, +0.019 on 11 of 11, is **significant**
— against +0.016 for adding loci to the same models. Per drug, five folds cannot
reach significance under this test (smallest possible p = 0.0625), so
ETHIONAMIDE +0.089, ISONIAZID +0.031 and PYRAZINAMIDE +0.020 are sizes, not
verdicts. At 19 loci the gain shrinks; for `locusfusion` it is +0.003 (§3).
""")

# ---------------------------------------------------------------- parameters
md("""
---
## 8. Parameters

Every network model's best complete cell at 19 loci — same input, schedule and
seed — across three orders of magnitude of parameter count.
""")

code('''
_net = best[(best.task == L19) & (best.family != "baseline")
            & ~best.model.isin(AGGREGATORS) & best.params.notna()]
s19 = (_net.sort_values("cv", ascending=False).groupby("model", as_index=False)
       .first().sort_values("params"))
fig, ax = plt.subplots(figsize=(9.4, 4.8))
ax.scatter(s19.params, s19.cv, s=90, color=[hue(m) for m in s19.model],
           edgecolor=SURFACE, linewidth=1.6, zorder=3)
ax.set_xscale("log"); ax.set_xlabel("parameters (log scale)")
ax.set_ylabel("macro CV AUC")
ax.set_ylim(s19.cv.min() - 0.012, s19.cv.max() + 0.010)
ax.set_xlim(s19.params.min() / 3.0, s19.params.max() * 3.0)
label_points(ax, s19.params, s19.cv,                 # after scale and limits
             [f"{r.model}" + chr(10) + f"{r.params/1e6:.2f}M  {r.cv:.4f}"
              for r in s19.itertuples()], fontsize=7.3)
ax.set_title("Parameter count vs. macro CV AUC, network models, all 19 loci",
             loc="left")
ax.legend([Line2D([0], [0], marker="o", lw=0, markersize=8, color=c)
           for c in (BLUE, RUST, RUST_LIGHT)],
          ["established architectures", "locusfusion", "locusfusion_v2"],
          loc="lower right")
tidy(ax, ygrid=True); plt.tight_layout(); plt.show()

_cells = best[(best.task == L19) & (best.family != "baseline") & best.params.notna()
              & (best.model != V2)]
rho, p = stats.spearmanr(_cells.params, _cells.cv)
print(f"Spearman(params, CV) over {len(_cells)} complete network cells at 19 loci "
      f"(aggregators included, v2 not): rho = {rho:+.2f}, p = {p:.2f}   "
      f"span {_cells.params.min()/1e6:.3f}M-{_cells.params.max()/1e6:.1f}M")
''')

md("""
**Parameter count is uncorrelated with accuracy.** Among the established models
the two best cells are 9× apart in parameters and 0.0015 apart in AUC. Matched
over five modality sets, `mdcnn` (5.1M median) − `cisfusion` (47.9M) is +0.004,
**not significant** (Holm p = 0.056). The two large models are not alike:
`cisfusion` is +0.012 over `late_fusion` (45.9M) at near-equal size, and `mdcnn`
+0.017 over it, both **significant** — so among the established models fusion
design moves AUC and size does not. `late_fusion` has ~90% of its parameters in
one fully-connected layer. The model that does separate is 0.66M. Two capacity
sweeps agree (`joint_capacity`, `setfusion_scaling`: 62 arms, nothing closed).
""")

# ==================================================================
# ============================ PART IV =============================
# ==================================================================
md("""
---
# Part IV — Significance and caveats

## 9. Significance tests

The paper's comparisons, fixed before any p was computed, one or more per claim.
Cells are matched, never a "best cell". Method in `SIGNIFICANCE_TESTS.md`.

- **Test.** Two-sided exact Wilcoxon signed-rank over the 55 fold-paired
  differences (11 drugs × 5 folds), BIG-TB's method. Every single-drug run uses
  `StratifiedKFold(5, random_state=42)` on the same cohort, so fold *i* is the
  same held-out isolates for every model.
- **Interval.** Hodges–Lehmann shift (the median of the Walsh averages) and its
  exact 95% CI, which inverts the same test. The macro Δ is printed beside it.
- **Multiple testing.** Holm over the 14 paper comparisons. Verdict:
  **significant** if Holm p < 0.05, otherwise **not significant**, which is not
  a claim of equivalence.
- Everything else the notebook compares is in the second table, unadjusted and
  without a verdict.
""")

code('''
frame_sig = pd.concat([frame, frame_lf_old], ignore_index=True)
G1, G2, G3, G4, G5 = PAPER_GROUPS
sig = paper_family(frame_sig, ALL_DRUGS, MODS5)          # results_helpers: one definition
show_sig(sig, caption=f"{len(sig)} paper comparisons · exact Wilcoxon signed-rank over "
                      "55 fold differences · Holm across the family")
print(sig.verdict.value_counts().to_string())
''')

code('''
plot_sig_by_claim(sig); plt.show()
''')

md("""
**11 of 14 significant.** Not significant: LASSO − `mdcnn` at 19 loci (−0.0005,
5/11), whole genome − 19 loci (+0.0022, 6/11), and `mdcnn` − `cisfusion`
(+0.0042, 8/11, Holm p = 0.056). These are "no detected difference", not
demonstrated parity. The figure omits three tabled rows: the tokenizer fix,
`onehot` − `delta`, and whole genome.

- **LASSO vs. CNN.** Level at 19 loci; at per-drug loci the LASSO is behind
  (−0.034, 0/11).
- **LocusFusion.** Ahead of the LASSO (+0.0103) and of `mdcnn` (+0.0117), and the
  tokenizer fix alone is +0.0095, 11/11 — about the size of both gaps.
- **Locus set.** Positive for every model family; the genome beyond the 19 loci
  adds nothing detected.
- **Modalities.** +0.0185 macro at per-drug loci, 11/11.
- **Model size.** `late_fusion` is behind both others; `mdcnn` and `cisfusion`,
  9× apart, are not distinguished.

HL Δ is a median shift over the 55 fold differences and runs below the macro
(mean) Δ where a few drugs carry the effect — modalities is +0.0185 macro against
+0.0086 HL, because ETHIONAMIDE and ISONIAZID carry most of it.
""")

md("""
**Outside the family.** Every other comparison the notebook cites: same test and
interval, raw p, no verdict.
""")

code('''
ext = WilcoxonComparisons(frame_sig, ALL_DRUGS)
for g, label, a, b, *dr in [
    (G1, "lasso: onehot − delta · per-drug loci",
     (PD, "lasso_onehot", "dna"), (PD, "lasso_delta", "dna")),
    (G2, "locusfusion − cisfusion · 19 loci · 5 modsets", (L19, V1, MODS5), (L19, "cisfusion", MODS5)),
    (G2, "locusfusion − late_fusion · 19 loci · 5 modsets", (L19, V1, MODS5), (L19, "late_fusion", MODS5)),
    (G2, "locusfusion − mdcnn · 19 loci · all_modalities", (L19, V1, A), (L19, "mdcnn", A)),
    (G2, "locusfusion − mdcnn · 19 loci · dna", (L19, V1, "dna"), (L19, "mdcnn", "dna")),
    (G2, "locusfusion − mdcnn · per-drug loci · 5 modsets", (PD, V1, MODS5), (PD, "mdcnn", MODS5)),
    (G2, "locusfusion: fixed − old tokenizer · per-drug loci · 5 modsets",
     (PD, V1, MODS5), (PD, "locusfusion_old", MODS5)),
    (G2, "locusfusion: all_modalities − dna · 19 loci", (L19, V1, A), (L19, V1, "dna")),
    (G2, "locusfusion − late_fusion · multi-drug · 5 modsets", (MD, V1, MODS5), (MD, "late_fusion", MODS5)),
    (G2, "locusfusion − mdcnn · multi-drug · 5 modsets", (MD, V1, MODS5), (MD, "mdcnn", MODS5)),
    *[(G3, f"{m}: 19 − per-drug loci · 5 modsets", (L19, m, MODS5), (PD, m, MODS5)) for m in EST3],
    (G3, "locusfusion: 19 − per-drug loci · all_modalities", (L19, V1, A), (PD, V1, A)),
    (G4, "ETHIONAMIDE: dna_regulatory − dna · established",
     (PD, EST3, "dna_regulatory"), (PD, EST3, "dna"), ["ETHIONAMIDE"]),
    (G4, "ISONIAZID: dna_regulatory − dna · established",
     (PD, EST3, "dna_regulatory"), (PD, EST3, "dna"), ["ISONIAZID"]),
    (G4, "PYRAZINAMIDE: dna_protein − dna · established",
     (PD, EST3, "dna_protein"), (PD, EST3, "dna"), ["PYRAZINAMIDE"]),
    ("Other", "mdcnn+tf − mdcnn · per-drug loci · 5 modsets", (PD, "mdcnn+tf", MODS5), (PD, "mdcnn", MODS5)),
]:
    ext.folds(g, label, a, b, drugs=dr[0] if dr else None)
ext.folds(G3, "AMIKACIN: lasso_delta whole genome − 19 loci",
          (WG, "lasso_genome", "dna"), (L19, "lasso_delta", "dna"), drugs=["AMIKACIN"])
ext.drugs_only("Other", "established: multi-drug − single-drug · 19 loci · 5 modsets",
               (MD, EST3, MODS5), (L19, EST3, MODS5))
sig_ext = ext.table()
show_sig(sig_ext, caption=f"{len(sig_ext)} comparisons outside the paper family · "
                          "unadjusted, no verdict", adjusted=False)
''')

md("""
Single-drug rows have 5 fold differences, and an exact Wilcoxon on 5 cannot go
below p = 0.0625. So no single drug's effect — ETHIONAMIDE's +0.089 included — can
be significant under this test. Their interval is the range, at 93.75% coverage.
""")

md("""
### Ranking the paper's models at 19 loci

Friedman over the 11 drugs with the Nemenyi critical difference for all pairs
(Demšar 2006). Networks at `all_modalities`, lassos at `dna`. Non-overlapping
intervals differ at α = 0.05.
""")

code('''
_rank_cells = ([(m, A) for m in ["mdcnn", "cisfusion", "late_fusion", V1]]
               + [("lasso_onehot", "dna"), ("lasso_delta", "dna")])
_R = pd.DataFrame({m: drug_cv(m, L19, ms) for m, ms in _rank_cells}) \\
       .reindex(ALL_DRUGS).dropna(axis=1)              # complete columns only
ranks = _R.rank(axis=1, ascending=False).mean().sort_values()
k, N = _R.shape[1], _R.shape[0]
fr = stats.friedmanchisquare(*[_R[c] for c in _R.columns])
CD = NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6 * N))

fig, ax = plt.subplots(figsize=(9.0, 0.34 * k + 1.2))
yy = np.arange(k)[::-1]
for yi, (m, rk) in zip(yy, ranks.items()):
    ax.plot([max(rk - CD / 2, 0), min(rk + CD / 2, k + 1)], [yi, yi], color=AXIS,
            lw=2.2, zorder=1, solid_capstyle="round")
    ax.scatter([rk], [yi], s=58, color=hue(m), zorder=3, edgecolor=SURFACE,
               linewidth=1.4)
    ax.text(min(rk + CD / 2, k + 1) + 0.15, yi, f"{rk:.2f}", va="center",
            ha="left", fontsize=7.4, color=INK)
ax.set_yticks(yy); ax.set_yticklabels(ranks.index, fontsize=8.5)
ax.set_xlim(0, k + 1.8)
ax.set_xlabel(f"mean rank over {N} drugs (1 = best), interval = rank ± CD/2")
ax.set_title("Friedman mean rank with Nemenyi CD, single-drug, 19 loci", loc="left")
ax.annotate(f"Friedman χ² = {fr.statistic:.1f}, df = {k - 1}, p = {fr.pvalue:.1e}"
            f"   ·   CD = {CD:.2f}   ·   N = {N}",
            xy=(0.0, -0.13 - 1.0 / k), xycoords="axes fraction", fontsize=8, color=INK)
tidy(ax); plt.tight_layout(); plt.show()

_sep = [m for m, rk in ranks.items() if rk - ranks.iloc[0] > CD]
print(f"best mean rank {ranks.index[0]} ({ranks.iloc[0]:.2f}), first on "
      f"{int((_R.rank(axis=1, ascending=False)[ranks.index[0]] == 1).sum())}/11 drugs")
print(f"separated from it by more than CD: {', '.join(_sep) or 'none'}")
''')

md("""
`locusfusion` has the best mean rank (1.27) and is first on 9 of 11 drugs. With
6 models and 11 drugs the CD is 2.27 ranks, which separates it from `late_fusion`
and `lasso_delta` only; the paired tests above also resolve `cisfusion` and
`lasso_onehot`, which this all-pairs correction cannot.
""")

# ---------------------------------------------------------------- caveats
md("""
---
## 10. How to argue with these numbers

**On the tests**

- **The pooled Wilcoxon treats the 55 fold differences as independent.** They
  are not: 5-fold training sets share 75% of their isolates, and most isolates
  carry labels for several drugs. Its p-values are optimistic; this is the
  convention being matched (BIG-TB's), not a property of the data.
- **No single drug can be significant.** Five fold differences cannot reach
  p < 0.0625, so every per-drug effect is a size, not a verdict.
- **Seed noise is small; partition noise is unmeasured.** A 5-seed rerun of
  `locusfusion` and `mdcnn` at 19 loci (`seeds_19loci`) put the SD across seeds
  at 0.0006 macro against 0.004–0.008 across folds, but its `locusfusion` arm
  ran the pre-fix tokenizer. Every test is on `StratifiedKFold` seed 42;
  `repartition_19loci` has `mdcnn` on four more partitions and no `locusfusion`
  arm on the fixed tokenizer.
- **Holm depends on the family.** §9's is 14 rows (`SIGNIFICANCE_TESTS.md`); the
  first 12 were fixed before their p-values, the two extra model-size rows after
  `mdcnn` − `late_fusion` had been seen unadjusted.
- **The §9 list was written after the prose it tests** — matched cells fixed
  before any p was computed, but not pre-registered before the runs.
- **A "best cell" is optimistic, unevenly.** It selects on the CV it reports, and
  the bias scales with how much a model varies across cells: `cisfusion` spreads
  0.0129 across modality sets, `locusfusion` 0.0031. Head-to-heads are paired.
- **Held-out test AUC is favourably selected** (best of 5 folds by validation
  AUC, where BIG-TB uses fold 4 unconditionally). Judge on CV.

**On the data and the models**

- **Lineage confounding is not controlled anywhere.** Lineages carry many neutral
  variants, so a model that learns lineage looks like one that learns resistance
  on this split — see §6, where the lasso's weight lands on *PE_PGRS*/*PPE*. A
  lineage-held-out split is the highest-value follow-up.
- **Dense-sequence and variant-token models are not the same experiment**: the
  established models read every column, `locusfusion` and the aggregators
  reference-difference input. They are never pooled into one ranking.
- **The variant-token family cannot see anything constant across the cohort**, by
  construction. That carries no discriminative signal, but it is a real
  restriction.
- **`single-drug, all loci` cannot be quoted against SD-CNN**, which sees the
  per-drug gene map. It is input-matched to MD-CNN only.
- **Joint `locusfusion` is the old tokenizer** — no fixed-tokenizer joint run
  exists.
- **`locusfusion_v2` is incomplete**: `dna_biophysical` 8 of 11 drugs at
  per-drug loci, `dna` 5 of 11 at 19 loci, nothing else at 19 loci or joint, and
  no job running.
- **Whole genome and 19-locus lassos are not representation matched** (`delta`
  only genome-wide), and SHAP attribution share mis-ranks predictive value (S2).
- **`setfusion` and the 150-epoch runs are excluded everywhere**: near-collinear
  tokens at init, and an epoch cap that bound on 40% of folds.
""")

# ==================================================================
# =========================== SUPPLEMENT ===========================
# ==================================================================
md("""
---
# Supplement

Results the paper does not lean on: the per-drug baselines, the SHAP analysis
that motivated the variant tokenizer, the six variant-set aggregators it was
developed through, and the coverage of the grid. None of the aggregators are in
§9's tests; their effects are sizes and drug counts.
""")

# ---------------------------------------------------------------- S1
md("""
## S1. Per drug, against both baselines

Best established single-drug cell per drug, against SD-CNN and the published
MD-CNN.
""")

code('''
_sd = frame[(frame.family == "established") & (frame.task != MD)]
bestpd = _sd.sort_values("cv", ascending=False).groupby("drug").first()
order = (bestpd.cv - SD_BASE_CV).sort_values().index
y = np.arange(len(order))

fig, ax = plt.subplots(figsize=(9.5, 5.6))
for yi, d in zip(y, order):
    ax.plot([SD_BASE_CV[d], bestpd.cv[d]], [yi, yi], color=AXIS, lw=1.6,
            zorder=1, solid_capstyle="round")
ax.scatter(SD_BASE_CV[order], y, s=46, color=MUTED, zorder=3,
           edgecolor=SURFACE, linewidth=1.6, label="SD-CNN")
ax.scatter(MD_BASE_CV[order], y, s=46, color=AQUA, zorder=3, marker="D",
           edgecolor=SURFACE, linewidth=1.6, label="MD-CNN (published)")
ax.scatter(bestpd.cv[order], y, s=52, color=BLUE, zorder=4,
           edgecolor=SURFACE, linewidth=1.6, label="best established cell")
for yi, d in zip(y, order):
    ax.text(bestpd.cv[d] + 0.006, yi, f"{bestpd.cv[d]:.3f}", va="center",
            ha="left", fontsize=7, color=INK)
ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8.5)
ax.set_xlim(0.60, 1.03); ax.set_xlabel("CV AUC")
ax.set_title("Best established single-drug cell vs. published baselines, by drug",
             loc="left")
ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.20))
tidy(ax); plt.tight_layout(); plt.show()

print(f"ahead of SD-CNN on {int((bestpd.cv[order] > SD_BASE_CV[order]).sum())} "
      f"of {len(order)} drugs")
''')

md("""
Widest where the baseline had headroom (ETHIONAMIDE starts at 0.69), narrowest
on RIFAMPICIN (0.976), where the *rpoB* RRDR is a tight positional signal a
one-hot already captures.
""")

# ---------------------------------------------------------------- S2
md("""
---
## S2. Where the SHAP attribution lands

`scripts/shap_attribution.py` runs GradientSHAP over the best fold of each
`full_run_v2` `mdcnn` cell. Per drug: how many input columns carry 90% of total
|SHAP|, against how many columns the model was given.
""")

code('''
SHAP_DIR = PROJECT / "results" / "analysis" / "shap" / "full_run_v2" / "mdcnn"
if not (SHAP_DIR / "columns_all.csv").is_file():
    print(f"no SHAP tables under {SHAP_DIR} -- section skipped")
else:
    cols = pd.read_csv(SHAP_DIR / "columns_all.csv")
    blks = pd.read_csv(SHAP_DIR / "blocks_all.csv")
    # columns_all.csv keeps only each block's top columns, so a column's share of
    # the whole input is its share of its block times that block's share. Rows
    # whose captured mass stops short of 90% are lower bounds, drawn hollow.
    cm = cols.merge(blks[["drug", "cell", "block", "share"]],
                    on=["drug", "cell", "block"])
    cm["total_share"] = cm.share_of_block * cm.share
    width = blks.groupby(["drug", "cell"]).length.sum()

    rows = []
    for (d, cell), gg in cm.groupby(["drug", "cell"]):
        gg = gg.sort_values("total_share", ascending=False)
        captured = float(gg.total_share.sum())
        rows.append(dict(drug=d, cell=cell, n_cols=int(width[(d, cell)]),
                         n90=int((gg.total_share.cumsum() < 0.90).sum() + 1),
                         top1=float(gg.total_share.iloc[0]),
                         censored=captured < 0.90))
    conc = (pd.DataFrame(rows).query("cell == 'all_modalities'")
            .set_index("drug").sort_values("n_cols"))

    y = np.arange(len(conc))
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    for yi, (d, r) in zip(y, conc.iterrows()):
        ax.plot([r.n90, r.n_cols], [yi, yi], color=AXIS, lw=1.6, zorder=1,
                solid_capstyle="round")
        ax.scatter([r.n_cols], [yi], s=44, color=MUTED, zorder=3,
                   edgecolor=SURFACE, linewidth=1.4)
        ax.scatter([r.n90], [yi], s=54, zorder=4, edgecolor=BLUE, linewidth=1.8,
                   color=SURFACE if r.censored else BLUE)
        ax.text(r.n90 * 0.82, yi, f"{'≥' if r.censored else ''}{r.n90}",
                va="center", ha="right", fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(conc.index, fontsize=8.5)
    ax.set_xlim(0.5, conc.n_cols.max() * 2.4)
    ax.set_xlabel("input columns (log scale)")
    ax.set_title("Input columns carrying 90% of |SHAP|, mdcnn, all modalities",
                 loc="left")
    ax.legend([Line2D([0], [0], marker="o", lw=0, markerfacecolor=c,
                      markeredgecolor=e, markersize=7)
               for c, e in ((BLUE, BLUE), (SURFACE, BLUE), (MUTED, MUTED))],
              ["columns carrying 90% of |SHAP|",
               "hollow = lower bound (table truncated per block)",
               "columns in the input"],
              loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.21))
    tidy(ax); plt.tight_layout(); plt.show()

    ok = conc[~conc.censored]
    print(f"resolved for {len(ok)}/{len(conc)} drugs; median columns for 90% of "
          f"attribution: {ok.n90.median():.0f} of {conc.n_cols.min():,}-"
          f"{conc.n_cols.max():,}; top column takes {ok.top1.median():.0%}")
''')

md("""
**Ninety percent of what a model reads is a handful of columns**, and the input
is nearly constant — censused against each alignment's `MT_H37Rv` row, the
median isolate differs at 0–3 columns of a 2.5 kb gene and 14 across all 19
loci. A dense-sequence encoder spends almost all its capacity restating
constants; `token_signal` measured exactly that on `setfusion` (0.14% of an
encoded token varied with genotype, attention flat at 1/8), which is why the
`+tf` arms lose.

**The caveat, which is also a result.** SHAP gave the DNA input 1.0% of the
budget, so `shap_followup` retrained without it:
""")

code('''
_ab = frame[(frame.task == PD) & (frame.model == "mdcnn")].groupby("modset").cv.mean()
lab = {"all_modalities": "dna+protein+biophys+reg", "dna": "dna",
       "no_regulatory": "dna+protein+biophys", "no_dna": "protein+biophys+reg",
       "regulatory_only": "regulatory"}
for k in [m for m in lab if m in _ab.index]:
    print(f"{lab[k]:>26s}  {_ab[k]:.4f}  " + "█" * int(round((_ab[k] - 0.5) * 60)))
print(f"{chr(10)}dropping DNA costs {_ab['no_dna'] - _ab['all_modalities']:+.4f}"
      f" -- from an input SHAP scored at 1% of the budget")
''')

md("""
Dropping the input SHAP called 1% costs ~0.10 AUC: **attribution share mis-ranks
predictive value.** What survives is the *positional* claim, which the variant
tokenizer is built on: one token per deviating column, a susceptible isolate as
the empty set against a learned null token, 19 loci × 4,066 columns collapsed to
a median ~14 tokens.
""")

# ---------------------------------------------------------------- S3
md("""
---
## S3. The six variant-set aggregators

Six aggregators (`models/experimental_models.py`) over one shared tokenizer, run
as measurements, not as a leaderboard. All consume an identical *flat* token set
(no locus hierarchy), so the only thing varying is the aggregation operator.

| `--arch` | aggregator | the question |
|---|---|---|
| `catalogue` | learned scalar per exact variant id, summed | how far does memorisation get? (a learned WHO-style catalogue) |
| `additive` | `Σ w(features_v)` | does featurising the variant generalise to unseen substitutions? |
| `noisyor` | `1 − Π(1 − p_v)` | "susceptible unless something confers resistance", as an architecture |
| `gatedpool` | sigmoid gate, no softmax | was *normalisation* what broke attention? |
| `deepsets` | `ρ([Σφ, max φ, log1p(count)])` | does attention buy anything over plain additivity? |
| `fm` | factorization machine, rank 8 | is epistasis worth `O(T·k)`? |

**Why not softmax.** Softmax normalises, so weights must sum to 1: with one
informative token and thirteen neutral ones the operator cannot say "this token,
absolutely" — the mechanism behind `token_signal`'s flat 1/8 attention. A needle
detector wants an absolute, monotone aggregator. `catalogue` and `additive` are a
**matched pair**: one change, whether a weight comes from a variant's *identity*
or its *features*.
""")

code('''
fam = best[best.model.isin(AGGREGATORS)].pivot_table(index="model", columns="task",
                                                     values="cv")
fpar = best[best.model.isin(AGGREGATORS)].pivot_table(index="model", columns="task",
                                                      values="params")
_TT = [t for t in [PD, L19] if t in fam.columns]
fam = fam.reindex(fam[_TT[-1]].sort_values().index)

y = np.arange(len(fam)); w = 0.36
fig, ax = plt.subplots(figsize=(9.6, 4.8))
for i, task in enumerate(_TT):
    off = (i - (len(_TT) - 1) / 2) * (w + 0.04)
    ax.barh(y + off, fam[task], height=w, edgecolor="none", zorder=2,
            color=TINTS[0] if i == 0 else ORANGE, label=task)
    for yi, v in zip(y, fam[task]):
        if not np.isnan(v):
            p = fpar.loc[fam.index[yi], task]
            ax.text(v + 0.0025, yi + off, f"{v:.4f}   ({p/1e3:.0f}k)",
                    va="center", ha="left", fontsize=7, color=INK)
# the two rules sit 0.016 apart, so their labels go to opposite sides
for i, task in enumerate(_TT):
    r = top_of("established", task)
    ax.axvline(r.cv, color=INK2, lw=1.1, ls="-" if i else "--", zorder=3)
    ax.text(r.cv, -1.05, (f"{r.model} {r.cv:.4f}, {task} " if i == 0
                          else f" {r.model} {r.cv:.4f}, {task}"),
            color=INK2, fontsize=7, va="center", ha="right" if i == 0 else "left")
ax.set_yticks(y); ax.set_yticklabels(fam.index, fontsize=9)
ax.set_ylim(-1.5, len(fam) - 0.4); ax.set_xlim(0.84, 0.95)
ax.set_xlabel("macro CV AUC, all modalities (parameter count in parentheses)")
ax.set_title("Variant-set aggregators by locus setting", loc="left")
ax.legend(loc="lower left")
tidy(ax); plt.tight_layout(); plt.show()

for task in _TT:
    r = top_of("established", task)
    print(f"{task:24s}  best aggregator {fam[task].max():.4f}   "
          f"best established {r.cv:.4f} ({r.model})   "
          f"gap {fam[task].max() - r.cv:+.4f}")
''')

md("""
**None of the six wins, and none was built to.** The best, `gatedpool`, is
−0.008 behind `mdcnn` at 19 loci. What they are for is S4 and S5.
""")

# ---------------------------------------------------------------- S4
md("""
---
## S4. `catalogue` → `additive`: what featurising a variant is worth

Identity versus features, one change. The stated hypothesis was *pncA* /
PYRAZINAMIDE — the mechanism §7 attributes the biophysical gain to.
""")

code('''
def cat_vs_add(task):
    s = frame[(frame.task == task) & (frame.modset == A)]
    c = s[s.model == "catalogue"].set_index("drug").cv
    a = s[s.model == "additive"].set_index("drug")
    if c.empty or a.empty:
        return None
    d = (a.cv - c).dropna()
    return pd.DataFrame({"n": a.n_valid.reindex(d.index), "gain": d}).dropna()


cva, cva_pd = cat_vs_add(L19), cat_vs_add(PD)
rho, pval = stats.spearmanr(cva.n, cva.gain)

# Eleven points, four within a factor of 1.4 in n -- as a scatter the labels
# collide on any ladder, so the drugs are ordered BY cohort size and the rank
# correlation is stated rather than eyeballed.
cva = cva.sort_values("n", ascending=False)
y = np.arange(len(cva))
fig, ax = plt.subplots(figsize=(9.2, 5.4))
ax.axvline(0, color=AXIS, lw=1, zorder=1)
for yi, r in zip(y, cva.itertuples()):
    ax.plot([0, r.gain], [yi, yi], color=ORANGE, lw=2, zorder=2,
            solid_capstyle="round")
    ax.scatter([r.gain], [yi], s=52, color=ORANGE, zorder=4, edgecolor=SURFACE,
               linewidth=1.4)
    ax.text(r.gain + 0.0018, yi, f"{r.gain:+.4f}", va="center", ha="left",
            fontsize=7.5, color=INK)
    if cva_pd is not None and r.Index in cva_pd.index:        # does it replicate?
        ax.scatter([cva_pd.gain[r.Index]], [yi], s=34, color=SURFACE, zorder=3,
                   edgecolor=ORANGE, linewidth=1.4)
ax.set_yticks(y)
ax.set_yticklabels([f"{d}   n = {int(n):,}" for d, n in cva.n.items()], fontsize=8)
ax.set_xlim(-0.006, cva.gain.max() * 1.28)
ax.set_xlabel("additive − catalogue   (CV AUC)")
ax.set_title("additive − catalogue CV AUC, by drug cohort size", loc="left")
ax.legend([Line2D([0], [0], color=ORANGE, lw=2, marker="o", markersize=6,
                  markeredgecolor=SURFACE),
           Line2D([0], [0], color=SURFACE, lw=0, marker="o", markersize=6,
                  markerfacecolor=SURFACE, markeredgecolor=ORANGE)],
          ["all 19 loci", "same measurement at per-drug loci"], loc="lower right")
ax.annotate(f"Spearman(n, gain) = {rho:+.3f},  p = {pval:.4f}",
            xy=(0.5, -0.135), xycoords="axes fraction", ha="center",
            fontsize=8.5, color=INK)
tidy(ax); plt.tight_layout(); plt.show()

print(f"macro {cva.gain.mean():+.4f} at 19 loci, {cva_pd.gain.mean():+.4f} at "
      f"per-drug loci;  LEVOFLOXACIN (n=269) {cva.gain['LEVOFLOXACIN']:+.4f}, "
      f"RIFAMPICIN (n=17,582) {cva.gain['RIFAMPICIN']:+.4f}")
''')

md("""
**The gain is not about *pncA*** — PYRAZINAMIDE gains +0.005. It is a
**small-cohort effect**: above ~8,000 isolates a learned catalogue has seen
enough, below it features matter, and at n = 269 they are worth 0.07 AUC. Macro
+0.024 and +0.022 at the two locus settings, 10 of 11 drugs each — a result about
sample efficiency, which is what transfers to a drug or region where labels are
scarce.
""")

# ---------------------------------------------------------------- S5
md("""
---
## S5. The gate, and the price

`gatedpool` is the attention query with softmax deleted; `deepsets` deletes
gating entirely. The rule, written before the run: *if `gatedpool` closes a gap
`deepsets` does not, normalisation was the problem; if `deepsets` matches it,
gating was never doing anything.*
""")

code('''
pair = (best[best.model.isin(["gatedpool", "deepsets"])]
        .pivot_table(index="model", columns="task", values="cv"))
print(f"{'':12}" + "".join(f"{t:>24}" for t in pair.columns))
for m in ["gatedpool", "deepsets"]:
    print(f"{m:<12}" + "".join(f"{pair.loc[m, t]:>24.4f}" for t in pair.columns))
print(f"{'difference':<12}" + "".join(
    f"{pair.loc['gatedpool', t] - pair.loc['deepsets', t]:>+24.4f}"
    for t in pair.columns))
''')

md("""
**Deleting the gating machinery costs nothing detected** — Δ +0.0009, 5/11
drugs, HL CI −0.0005 to +0.0022, Wilcoxon p = 0.26 unadjusted (outside §9's
family). The contribution is the **representation**, not the attention. The
price, every complete cell at 19 loci, both families:
""")

code('''
p19 = (best[(best.task == L19) & best.params.notna()]
       .sort_values("cv", ascending=False).groupby("model", as_index=False).first())
fig, ax = plt.subplots(figsize=(9.6, 5.0))
for f_, mk in (("established", "o"), ("experimental", "s")):
    s = p19[p19.family == f_]
    ax.scatter(s.params, s.cv, s=88, marker=mk, zorder=3,
               color=[hue(m) for m in s.model], edgecolor=SURFACE, linewidth=1.6,
               label=f_)
ax.set_xscale("log"); ax.set_xlabel("parameters (log scale)")
ax.set_ylabel("macro CV AUC")
ax.set_xlim(p19.params.min() / 5.0, p19.params.max() * 5.0)
ax.set_ylim(p19.cv.min() - 0.022, p19.cv.max() + 0.020)
label_points(ax, p19.params, p19.cv,                 # after scale and limits
             [f"{r.model}" + chr(10) + f"{r.params/1e6:.3f}M  {r.cv:.4f}"
              for r in p19.itertuples()], fontsize=7.1)
ax.set_title("Parameter count vs. macro CV AUC, both families, all 19 loci",
             loc="left")
ax.legend(loc="lower right")
tidy(ax, ygrid=True); plt.tight_layout(); plt.show()

a, l = p19[p19.model == "additive"].iloc[0], p19[p19.model == "late_fusion"].iloc[0]
print(f"additive {a.cv:.4f} at {a.params:,.0f} params vs late_fusion {l.cv:.4f} at "
      f"{l.params:,.0f}: {a.cv - l.cv:+.4f} AUC at {l.params / a.params:.0f}x fewer")
''')

md("""
`additive` — a sum of per-variant weights — lands level with `late_fusion` at a
thousandth of the parameters. `fm` is 0.007 behind it: its rank-8 interaction
term costs more in variance than epistasis recovers. **The signal is sparse and
mostly additive.**

Aggregator-specific limits: `noisyor` cannot learn a protective variant (every
factor in (0,1)) and has no per-drug-loci number (a CUDA fault in the tokenizer's
`topk`); no aggregator has a joint arm; `residual_catalogue` was never run.
""")

# ---------------------------------------------------------------- S6
md("""
---
## S6. Coverage and the full table

Tinted = complete (11 drugs), amber = short, blank = not run.
""")

code('''
piv = (frame.groupby(["task", "modset", "model"]).drug.nunique().rename("drugs")
       .reset_index().pivot_table(index=["task", "modset"], columns="model",
                                  values="drugs"))
_order = ([m for m in sorted(ESTABLISHED) if m in piv.columns]
          + [m for m in AGGREGATORS if m in piv.columns]
          + [m for m in [V1, V2] if m in piv.columns]
          + [m for m in sorted(BASELINE_MODELS) if m in piv.columns])
piv = piv.reindex(columns=_order)
piv = piv.reindex(index=[(t, m) for t in TASKS for m in meta["MODSETS"]
                         if (t, m) in piv.index])

# the family grouping goes on the column headers, which needs a table-style
# selector: a <th> painted by Styler.apply is overwritten by the cell styler
_head = [{"selector": f"th.col_heading.level0.col{i}",
          "props": tint(FAM_TINT[family(c)]) + "; font-weight:600; font-size:11px; "
                   "text-align:center"}
         for i, c in enumerate(piv.columns)]
display(style_table(
    piv.style.format("{:.0f}", na_rep="")
    .map(lambda v: "" if pd.isna(v) else tint(PALE if v == 11 else "#fdf0d0"))
    .set_table_styles(_head, overwrite=False),
    caption="drugs completed per cell — header tint: blue = established, "
            "orange = experimental, green = baseline",
    hide_index=False))
''')

code('''
tab = (best[["family", "task", "model", "modset", "loci", "cv", "params"]]
       .sort_values(["family", "task", "cv"], ascending=[True, True, False]).copy())
tab["vs baseline"] = tab.apply(
    lambda r: r.cv - (SD_BASE if r.task == PD else MD_BASE), axis=1)
tab["params"] = tab.params / 1e6
display(style_table(
    tab.style.format({"cv": "{:.4f}", "vs baseline": "{:+.4f}",
                      "params": "{:.3f}M", "loci": "{:.0f}"}, na_rep="—")
    .map(lambda v: tint(FAM_TINT[v]), subset=["family"]),
    caption="every complete cell (11/11 drugs), by family then task, best first"))
''')

# ================================================================== write
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(cells)} cells)")

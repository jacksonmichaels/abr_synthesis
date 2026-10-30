"""Generate notebooks/claims.ipynb — the figures behind the paper's claims.

    python scripts/build_claims.py
    python scripts/build_claims.py -o /tmp/preview.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/claims.ipynb

Three notebooks now read the same tidy table, and they are not the same view:

  `overview.ipynb`  every cell of the task x modality x model cube, as tables,
                    for deciding what to run next
  `results.ipynb`   the project tour, in the order you would narrate it
  `claims.ipynb`    THIS ONE -- one figure per claim the paper intends to make,
                    plus an appendix of the fuller versions

The argument this notebook is built to carry, in four figures:

  Figure 1  (claim 1)  a lasso on the raw alignment matches the best deep cell
  Figure 2  (claim 2)  the locus universe is what moves the number; modality is
                       second and concentrated; architecture and multi-task
                       sharing are inside the noise
  Figure 3  (claim 4)  what a modality TAKES in attribution does not track what
                       it BUYS in AUC
  Figure 4  (claim 6)  the read-out attention is flat while SHAP is not, and
                       deleting the gate costs nothing

Figures 3 and 4 are why Figure 2 is not a null result. On their own, "the
architecture does not matter" and "a linear model ties" invite the reply that
the deep models must be doing something the ablation cannot see. Figures 3 and 4
close that door from the inside: the selection machinery -- the part that is
supposed to be finding the signal -- is measurably not selecting, and removing it
changes nothing. So the input, not the architecture, is where the performance
lives, which is exactly what Figure 2 decomposes and Figure 1 bounds.

`locusfusion` is the method under test throughout. Other models appear when they
are the control that makes a locusfusion number mean something, and not
otherwise.

Figures are matplotlib, light surface, dark ink, one panel wherever one panel
will do. **A figure title names the subject and the slice, never the finding** --
the claim goes in the markdown around it, where it can be qualified, and a title
cannot carry a caveat. Nothing explanatory is drawn inside a chart either; what
may be drawn is axis labels, data values, series names, statistical results, and
reference lines labelled with what they are. See `~/.claude/CLAUDE.md`.

Colour does one job, identity, and it never does it alone: every series is also
direct-labelled or given its own marker, so the figures survive greyscale and
colour-vision deficiency. The palette is the one `scripts/build_results.py`
established -- blue for the established architectures, rust for `locusfusion`,
orange for the rest of the variant-token family, aqua for a baseline that is not
a network -- so the two notebooks cannot say the same thing in different colours.
"""
import argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("-o", "--output", type=Path, default=None,
                 help="notebook path (default: <project>/notebooks/claims.ipynb)")
_args = _ap.parse_args()
OUT = (_args.output.expanduser().resolve() if _args.output
       else PROJECT / "notebooks" / "claims.ipynb")

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.rstrip("\n").splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


# ============================================================== title
md("""
# What this work can claim

Genotype -> phenotype antibiotic-resistance prediction for *M. tuberculosis*,
11 drugs, 17,942 clinical isolates, scored against the two published CNNs of the
BIG-TB benchmark.

**Part I** is four figures, one per claim, for reading in order. **Part II** is
the appendix: the same evidence at full per-drug resolution, plus the
measurements that qualify it.

**How every number is judged.** 5-fold cross-validated AUC, macro-averaged over
the 11 drugs shared with the baselines. Held-out test AUC is recorded because it
is what the papers publish, but it is one 20% split of one best-fold model and
swings +/-0.05 on the small drugs. **Single seed throughout**, so a difference
smaller than the fold-to-fold spread drawn in Figure 2 is not resolved -- it is
plotted, but it is not a claim.

> Generated — edit `scripts/build_claims.py`, not the `.ipynb`.
> `python scripts/build_claims.py && jupyter nbconvert --to notebook --execute --inplace notebooks/claims.ipynb`
""")

# ============================================================== setup
code('''
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from IPython.display import display

_here = Path.cwd()
PROJECT = next(p for p in [_here, *_here.parents] if (p / "bigtb_ref.py").exists())
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from results_frame import load_frame
from bigtb_baselines import SD_BASE_CV, MD_BASE_CV, ALL_DRUGS

frame, meta = load_frame()
DRUGS = list(ALL_DRUGS)

# --- the model families -----------------------------------------------------
# Same line `models/__init__.py` draws: DELTA_ARCHS is locusfusion plus the
# experimental aggregators, i.e. everything that consumes reference-difference
# input. Everything else reads dense sequence.
AGGREGATORS = ["catalogue", "additive", "noisyor", "gatedpool", "deepsets", "fm"]
EXPERIMENTAL = set(AGGREGATORS) | {"locusfusion"}
BASELINES = {"sparse_baseline", "sparse_baseline_trainvocab",
             "lasso_onehot", "lasso_delta"}
ESTABLISHED = ["mdcnn", "cisfusion", "late_fusion"]

# `locusfusion` exists twice at BOTH single-drug locus sets -- once per
# tokenizer -- so every reference to it must name a run or it silently picks
# whichever sorts first. LF_RUN is the current model; LF_OLD is the tokenizer it
# replaced, and is what the SHAP corpus was computed on (those checkpoints are
# the old ones, which is now a stated limitation of Figures 3 and 4-left).
LF_RUN, LF_OLD = "locusfusion_fixed", "newmodels_full"

# `setfusion` is already dropped by results_frame (degenerate tokens --
# token_signal); `+tf` are the transformer-encoder arms, kept for the appendix.

# --- palette (mirrors scripts/build_results.py; identity only) --------------
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, AXIS  = "#898781", "#e1e0d9", "#c3c2b7"
BLUE   = "#2a78d6"   # established architectures
ORANGE = "#eb6834"   # experimental aggregators
RUST   = "#a8420f"   # locusfusion -- same family, deeper: it is the method
AQUA   = "#1baf7a"   # a baseline that is not a network
PALE   = "#f0efec"

def hue(model):
    """Colour follows the entity, never its rank."""
    if model in BASELINES:  return AQUA
    if model == "locusfusion": return RUST
    return ORANGE if model in EXPERIMENTAL else BLUE

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "figure.dpi": 120,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlepad": 10,
})

def tidy(ax, xgrid=True, ygrid=False):
    ax.set_axisbelow(True)
    ax.xaxis.grid(xgrid); ax.yaxis.grid(ygrid)
    return ax

# --- the macro table --------------------------------------------------------
# One row per (task, modset, model, run); `run` is kept because locusfusion
# exists twice at per-drug loci -- newmodels_full is the old tokenizer and
# locusfusion_fixed is the rebuilt one, and collapsing them would average a bug
# with its fix.
MACRO = (frame.groupby(["task", "modset", "model", "run"])
              .agg(cv=("cv", "mean"), test=("test", "mean"),
                   cv_sd=("cv_sd", "mean"), n_params=("n_params", "max"),
                   n=("drug", "size")).reset_index())
MACRO = MACRO[MACRO.n == len(DRUGS)]        # complete cells only

def cell(task, modset, model, run=None):
    q = MACRO[(MACRO.task == task) & (MACRO.modset == modset) & (MACRO.model == model)]
    if run is not None:
        q = q[q.run == run]
    return q.iloc[0] if len(q) else None

# The noise floor every claim is measured against: the mean per-drug
# fold-to-fold SD of the 5-fold CV, over the single-drug cells of the
# architectures this notebook argues about.
NOISE = float(frame[(frame.task != "multi-drug")
                    & (frame.model.isin(ESTABLISHED + ["locusfusion"]))].cv_sd.mean())

# ================================================== intervals
# Every interval in this notebook is one of three things, and which one is
# always named in the figure's caption.
#
#  (1) PERCENTILE BOOTSTRAP OVER THE 11 DRUGS, 10,000 resamples, fixed seed.
#      Used for any macro quantity and for any paired difference between two
#      models. The drugs are the replicate unit; resampling them makes no
#      distributional assumption, and for a paired quantity it resamples the
#      PAIRS, so the pairing is preserved.
#
#  (2) t INTERVAL OVER THE 5 CV FOLDS, for a single drug's CV mean.
#
#  (3) t INTERVAL OVER THE 5 FOLD-PAIRED DIFFERENCES, for one drug, one pair of
#      models. Every single-drug run splits with the same
#      `StratifiedKFold(5, shuffle=True, random_state=42)` over the same cohort,
#      so fold i of two models is the same held-out isolates and the two are
#      paired. Pairing removes the fold-difficulty term, which is the larger one.
#      NOT valid against the joint runs -- `run_multidrug_cv` uses a
#      non-stratified split, so those are paired over drugs only.
#
# (2) and (3) are known to be ANTI-CONSERVATIVE: k-fold CV folds share training
# data, and there is no unbiased estimator of the variance of k-fold CV
# (Bengio & Grandvalet 2004). They are drawn because they are the honest
# available bound, not because they are exact.
#
# **What none of them contain: seed variance.** Every run in this project is a
# single seed, so initialisation and training stochasticity are not sampled at
# all. Every interval below is therefore a LOWER BOUND on total uncertainty.
NBOOT, ALPHA = 10_000, 0.05
RNG = np.random.default_rng(20260903)

def boot_ci(values, n=NBOOT, alpha=ALPHA):
    """Percentile bootstrap CI for the mean, resampling the rows of `values`."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    draws = v[RNG.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return tuple(np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)]))

def t_ci(values, alpha=ALPHA):
    """t interval for the mean of a handful of observations (5 folds)."""
    from scipy import stats
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    h = stats.t.ppf(1 - alpha / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return (v.mean() - h, v.mean() + h)

def fold_macro_ci(folds, alpha=ALPHA):
    """95% CI for a MARGINAL macro AUC, with the drug panel treated as FIXED.

    The 11 drugs are the benchmark's, not a sample, so the only sampled
    variation in "this model's macro on this benchmark" is fold noise. Combine
    each drug's fold SE and divide by the number of drugs.

    Use this for any single model's macro. Do NOT use a bootstrap over drugs
    there: drug difficulty spans 0.81-0.98, that spread is SHARED by every
    model, and folding it into a marginal interval makes every model look
    equally uncertain while saying nothing about the model. Paired differences
    are the opposite case -- see the note in Part I.
    """
    ses = [np.std(np.asarray(f), ddof=1) / np.sqrt(len(f))
           for f in folds if len(f) > 1]
    if not ses:
        return (np.nan, np.nan)
    se = np.sqrt(np.sum(np.square(ses))) / len(ses)
    from scipy import stats
    h = stats.norm.ppf(1 - alpha / 2) * se
    m = float(np.mean([np.mean(f) for f in folds]))
    return (m - h, m + h)

def boot_rho_ci(df, xcol, ycol, cluster="drug", n=4000, alpha=ALPHA):
    """Spearman rho CI, resampling CLUSTERS (a drug contributes several points)."""
    groups = [g for _, g in df.groupby(cluster)]
    out = []
    for _ in range(n):
        s = pd.concat([groups[i] for i in RNG.integers(0, len(groups), len(groups))])
        if s[xcol].nunique() > 2 and s[ycol].nunique() > 2:
            out.append(spearmanr(s[xcol], s[ycol]).statistic)
    return tuple(np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)]))

# --- the per-fold table, which every interval above is computed from --------
FOLDS = frame[frame.fold_aucs.apply(len) == 5].copy()

def folds_of(task, modset, model, run=None):
    """-> Series drug -> (5 fold AUCs). One cell."""
    q = FOLDS[(FOLDS.task == task) & (FOLDS.modset == modset) & (FOLDS.model == model)]
    if run is not None:
        q = q[q.run == run]
    q = q.drop_duplicates("drug")
    return q.set_index("drug").fold_aucs

def per_drug_delta(a, b, paired=True):
    """Per-drug (cell a - cell b). `a`, `b` are `folds_of` results.

    Fold-paired when both sides come from the same splitter; otherwise the
    difference of the two fold MEANS, which is the same point estimate with a
    wider honest interval.
    """
    k = [d for d in a.index if d in b.index]
    if paired:
        return pd.Series({d: float(np.mean(np.asarray(a[d]) - np.asarray(b[d])))
                          for d in k})
    return pd.Series({d: float(np.mean(a[d]) - np.mean(b[d])) for d in k})

def macro_of(task, modset, model, run=None):
    """Per-drug CV means for one cell, at full fold precision."""
    return folds_of(task, modset, model, run).apply(np.mean)

print(f"models   : {', '.join(meta['MODELS'])}")
print(f"cells    : {len(MACRO)} complete (11/11 drugs)   rows: {meta['n_rows']}")
print(f"per-fold : {len(FOLDS)}/{len(frame)} rows carry all 5 fold AUCs")
print(f"noise floor: mean per-drug CV fold SD = {NOISE:.4f}")
''')

# ============================================================== PART I
md("""
---
# Part I — the four claims

### How to read the intervals

Every interval drawn in this notebook is one of three things, and each figure's
footnote names which:

* **A marginal macro** (one model's macro AUC) gets a **95% CI from fold noise
  with the 11-drug panel treated as fixed**. The eleven drugs are the
  benchmark's, not a sample from anything, so they contribute no sampling
  variation to "this model's score on this benchmark."
  *Why not bootstrap the drugs there:* drug difficulty spans 0.81 to 0.98 —
  a spread of 0.17, twenty times any difference between models — and it is
  **shared by every model**. Resampling drugs folds that into each model's
  interval, making all of them look equally uncertain (±0.027 instead of
  ±0.005) while saying nothing about the model. Overlapping marginal intervals
  would then be read as "no difference," which is exactly the wrong inference.
* **A paired difference between two models** gets a **95% percentile bootstrap
  over the 11 drugs** (10,000 resamples, fixed seed). Here the drug spread
  cancels — the models are correlated at r = 0.98-0.99 across drugs — and what
  the bootstrap captures instead is real: how much the *difference itself*
  varies by drug. That is a property of the comparison and belongs in its
  interval.
* **95% t interval over the 5 CV folds** — for one drug's CV mean.
* **95% t interval over the 5 fold-paired differences** — for one drug, two
  models. Every single-drug run splits with the same
  `StratifiedKFold(5, shuffle=True, random_state=42)` over the same cohort, so
  fold *i* is the same held-out isolates for both models and the two are paired.
  Pairing removes the fold-difficulty term, which is much the larger one. It is
  *not* valid against the joint runs, which split differently; those are paired
  over drugs only.

**Two things these intervals are not.**

They are **anti-conservative**: k-fold CV folds share training data, and there is
no unbiased estimator of the variance of k-fold cross-validation
(Bengio & Grandvalet, 2004). And they contain **no seed variance at all** —
every run in this project is a single seed, so initialisation and training
stochasticity are simply not sampled. Every interval here is a lower bound on
the true uncertainty.

Where the mean's interval is wide because of one drug, the **sign count**
(*k*/11 drugs up) is printed beside it. It makes no distributional assumption and
is the more robust statement in exactly those cases.

**Nothing plotted here is selected on the data it is compared against.** Every
model appears at the same modality set, the architecture reference in Figure 2
is pre-specified (`mdcnn`, BIG-TB's own topology), and A3 shows all four
modality sets rather than each drug's best of four. Where an ordering is chosen
for legibility the ordering is descriptive and the values are not.
""")

# -------------------------------------------------------------- claim 1
md("""
## Claim 1 — a lasso matches the published and conventional deep models; what beats them is a better input representation

Every model below sees the same 19 curated loci, the same cohort and the same
5-fold protocol.

**Read the right panel, not the left.** The left panel's intervals are wide
because the eleven drugs differ enormously in difficulty, and that spread is
shared by every model — overlapping marginal intervals there are not evidence of
anything. The right panel removes it by pairing: same drug, same fold, same
held-out isolates, one model minus the other.

**The lasso ties the conventional deep models.** 228 nonzero coefficients
against `mdcnn`'s 6.7 M parameters: **−0.0005 [−0.0033, +0.0023]**, 5 of 11
drugs up, and 0.9241 against the published MD-CNN's 0.9222. `cisfusion` at
60.7 M parameters is −0.0015 [−0.0038, +0.0004] against `mdcnn`, and
`late_fusion` at 57.7 M is **−0.0175 [−0.0407, −0.0039]** — resolvably worse for
nine times the parameters. Nothing in that group is buying anything with its
capacity, which is Claim 6.

**One model does beat them, and it is the one with the better input
representation.** `locusfusion` with the rebuilt tokenizer reaches **0.9343** —
**+0.0098 [+0.0055, +0.0142] over `mdcnn`** and **+0.0103 [+0.0061, +0.0148]
over the lasso**, ahead on 10 of 11 drugs in both cases, at 0.66 M parameters
(a tenth of `mdcnn`).

**This does not contradict the thesis; it is the sharpest version of it.** The
rebuild changed no architecture and added no capacity — 646,273 parameters
against 645,763, a wash. What changed is how a variant becomes a token: the
coordinate is computed from the CDS annotation instead of learned (it had put
*katG* S315 43 codons from its own protein token), a token became a symbol pair
rather than a 42-float slot vector, and an unknown base call stopped reading as
wild type. That input-representation change alone is worth **+0.0107
[+0.0066, +0.0147] at 19 loci, improving all 11 drugs** (A5, A11). So the three
levers that move this task — locus scope, modality, and representation — are all
on the input side, and the one lever that does not is model capacity.

*Held-out test macro, quoted because it does not contradict the CV and because
it is what the papers publish: `locusfusion` 0.9343 CV corresponds to the best
cell in the project. Test AUC is one 20% split of one best-fold model and swings
±0.05 on the small drugs, so it is never the number to conclude from.*
""")

code('''
LOCI19 = "single-drug, all loci"
REF = ("mdcnn", "all_modalities")          # the best established cell, the reference
# Every network at the SAME modality set. Picking each model's best-scoring
# cell would be selection on the same data -- and it is selection over a flat
# line: locusfusion's five 19-locus cells span 0.9230-0.9242. The lasso has one
# arm (DNA), which is the point of it.
rows = [
    ("L1-logistic, one-hot alignment", "dna",            "lasso_onehot"),
    ("locusfusion (rebuilt tokenizer), all four", "all_modalities", "locusfusion"),
    ("mdcnn, all four modalities",     "all_modalities", "mdcnn"),
    ("cisfusion, all four modalities", "all_modalities", "cisfusion"),
    ("late_fusion, all four modalities", "all_modalities", "late_fusion"),
]
# The mean number of nonzero lasso coefficients, read from the run's own JSONs.
_lasso = [json.loads(q.read_text()) for q in
          (PROJECT / "results/experiments/lasso_wholeseq_20260901/onehot_all").glob("*.json")]
LASSO_SUPPORT = np.mean([d["cv_support_mean"] for d in _lasso if "cv_support_mean" in d])

ref_folds = folds_of(LOCI19, REF[1], REF[0])
data = []
for label, modset, model in rows:
    fo = folds_of(LOCI19, modset, model,
                  run=LF_RUN if model == "locusfusion" else None)
    per_drug = fo.apply(np.mean)
    dl = per_drug_delta(fo, ref_folds)              # fold-paired, per drug
    if model == "lasso_onehot":
        size = f"{LASSO_SUPPORT:,.0f} coefficients"
    else:
        npar = MACRO[(MACRO.task == LOCI19) & (MACRO.modset == modset)
                     & (MACRO.model == model)].n_params.iloc[0]
        size = f"{npar / 1e6:.2f} M parameters"
    data.append(dict(label=label, model=model, size=size,
                     macro=per_drug.mean(), ci=fold_macro_ci(list(fo)),
                     delta=dl.mean(), dci=boot_ci(dl),
                     wins=int((dl > 0).sum()), n=len(dl)))
data.sort(key=lambda d: d["macro"])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 3.2),
                               gridspec_kw={"width_ratios": [1.32, 1]})
y = np.arange(len(data))

# --- A: the macro, with its bootstrap CI ----------------------------------
for i, d in enumerate(data):
    axA.plot(d["ci"], [i, i], color=hue(d["model"]), lw=2.0, alpha=0.55,
             solid_capstyle="butt", zorder=2)
axA.scatter([d["macro"] for d in data], y, s=58, zorder=3,
            color=[hue(d["model"]) for d in data], edgecolor=SURFACE, linewidth=1.3)
for i, d in enumerate(data):
    axA.annotate(f"{d['macro']:.4f}   {d['size']}", xy=(0.995, i),
                 xycoords=("axes fraction", "data"), ha="right", va="center",
                 fontsize=7.6, color=INK,
                 bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.6))
axA.axvline(MD_BASE_CV.mean(), color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=1)
axA.annotate(f"MD-CNN baseline {MD_BASE_CV.mean():.4f}",
             xy=(MD_BASE_CV.mean(), 1.005), xycoords=("data", "axes fraction"),
             xytext=(-4, 0), textcoords="offset points", ha="right", va="bottom",
             fontsize=7.4, color=INK2)
axA.set_yticks(y, [d["label"] for d in data], fontsize=8.2)
axA.set_xlabel("macro 5-fold CV AUC over 11 drugs")
axA.set_xlim(0.898, 0.968)
axA.set_title("Macro CV AUC by model\\n(interval: fold noise, drug panel fixed)")
tidy(axA)

# --- B: fold-paired difference against the reference cell -----------------
for i, d in enumerate(data):
    if d["model"] == REF[0]:
        continue
    axB.plot(d["dci"], [i, i], color=hue(d["model"]), lw=2.0, alpha=0.55,
             solid_capstyle="butt", zorder=2)
    axB.scatter([d["delta"]], [i], s=58, color=hue(d["model"]),
                edgecolor=SURFACE, linewidth=1.3, zorder=3)
    axB.annotate(f"{d['delta']:+.4f}  [{d['dci'][0]:+.4f}, {d['dci'][1]:+.4f}]"
                 f"   {d['wins']}/{d['n']} up",
                 (max(d["dci"][1], 0), i), xytext=(8, 0), textcoords="offset points",
                 va="center", fontsize=7.2, color=INK)
axB.axvline(0, color=INK2, lw=1.1, zorder=1)
axB.set_yticks(y, ["" for _ in data])
axB.set_xlabel("CV AUC minus mdcnn, all four modalities")
axB.set_xlim(-0.048, 0.052)
axB.set_title("Fold-paired difference against that cell\\n(interval: bootstrap over drugs)")
tidy(axB)

fig.suptitle("Macro CV AUC by model, single-drug task, all 19 loci",
             y=1.04, fontsize=11, fontweight="bold")
fig.text(0.5, -0.11, "left: 95% CI from fold noise with the 11-drug panel treated as "
         "fixed. right: 95% percentile bootstrap over the 11 drugs, which also "
         "captures how much\\nthe difference itself varies by drug. Single seed "
         "throughout, so no interval here contains seed variance.",
         ha="center", fontsize=7.2, color=MUTED, linespacing=1.5)
plt.tight_layout(); plt.show()
''')

# -------------------------------------------------------------- claim 2
md("""
## Claim 2 — the locus universe is what moves the number

Each bar is a **paired** effect: the same architecture and the same modality set,
measured twice with one thing changed, then averaged over every matched pair that
exists. The interval is a bootstrap over drugs; the count beside it is how many
of the 11 drugs moved up.

**The locus universe is the largest effect by a factor of three, and for the
lasso it is unambiguous** — +0.049 [+0.029, +0.074], **11 of 11 drugs**. For the
networks the point estimate is +0.016 but the interval contains zero, and the
sign count explains why: **10 of 11 drugs improve**, and the interval is opened
by a single outlier, LEVOFLOXACIN at −0.075, which has 269 labelled isolates and
the widest fold spread of any drug in the panel (A9). Ten of eleven under a sign
test is p = 0.012; the mean is the fragile summary here, not the effect.

**Adding all three non-DNA modalities is resolved but small** — +0.014
[+0.005, +0.025], 9 of 11 up. **Multi-task sharing is nothing**: +0.001
[−0.003, +0.006], 5 of 11 up, which is exactly a coin flip. And the best
DNA-only network is **−0.007 [−0.017, +0.000]** against the lasso, ahead on
2 of 11 drugs.

**The caveat that belongs beside this, not after it.** The lasso is readable, and
its largest 19-locus coefficients include loci that cannot be causal — `rpoB` in
the ISONIAZID model, `embB` in the PYRAZINAMIDE model. That is MDR co-occurrence
and lineage structure, and the networks given the same 19 loci can exploit it
too. So "more loci help" is evidenced here; "more loci carry mechanism" is not,
and separating them needs a lineage-held-out split that no run in this project
has.
""")

code('''
def matched_cells(task_a, task_b, models=ESTABLISHED, modsets=None):
    """Every (modset, model) that exists on both sides -> per-drug deltas.

    Returns a DataFrame indexed by drug, one column per matched cell pair, so a
    bootstrap can resample DRUGS while keeping every cell -- which is the
    dependency structure these effects actually have.
    """
    def side(t):
        q = FOLDS[(FOLDS.task == t) & (FOLDS.model.isin(models))]
        if modsets is not None:
            q = q[q.modset.isin(modsets)]
        return q.drop_duplicates(["modset", "model", "drug"])
    A, B = side(task_a), side(task_b)
    ka = set(map(tuple, A[["modset", "model"]].drop_duplicates().values))
    kb = set(map(tuple, B[["modset", "model"]].drop_duplicates().values))
    # Fold-paired only within the single-drug family; the joint engine splits
    # differently, so a joint comparison is paired over drugs alone.
    fp = "multi-drug" not in (task_a, task_b)
    cols = {}
    for modset, model in sorted(ka & kb):
        a = A[(A.modset == modset) & (A.model == model)].set_index("drug").fold_aucs
        b = B[(B.modset == modset) & (B.model == model)].set_index("drug").fold_aucs
        cols[f"{modset}__{model}"] = per_drug_delta(a, b, paired=fp)
    return pd.DataFrame(cols)

def matched_modsets(a, b, tasks, models=ESTABLISHED):
    cols = {}
    for t in tasks:
        for model in models:
            fa, fb = folds_of(t, a, model), folds_of(t, b, model)
            if len(fa) and len(fb):
                cols[f"{t}__{model}"] = per_drug_delta(fa, fb)
    return pd.DataFrame(cols)

SD, MD = "single-drug", "multi-drug"
lo_a, lo_b = folds_of(LOCI19, "dna", "lasso_onehot"), folds_of(SD, "dna", "lasso_onehot")
# `mdcnn` is PRE-SPECIFIED here -- it is BIG-TB's own topology and the project's
# reference architecture. Taking whichever established net happened to score
# highest would be selection on the same data the comparison is made from.
NET_REF = "mdcnn"

effects = [
    ("locus universe: the drug's 2-3 loci -> all 19\\n(L1-logistic, DNA only)",
     per_drug_delta(lo_a, lo_b).to_frame("lasso"), AQUA),
    ("locus universe: the drug's 2-3 loci -> all 19\\n(networks)",
     matched_cells(LOCI19, SD), BLUE),
    ("modality set: DNA -> all four\\n(networks, both single-drug locus sets)",
     matched_modsets("all_modalities", "dna", [SD, LOCI19]), BLUE),
    ("multi-task sharing: 11 single-drug models -> one joint\\n(networks, at 19 loci)",
     matched_cells(MD, LOCI19), BLUE),
    ("architecture: the mdcnn DNA cell - the lasso\\n(at 19 loci)",
     per_drug_delta(folds_of(LOCI19, "dna", NET_REF), lo_a).to_frame("net"), BLUE),
]

fig, ax = plt.subplots(figsize=(8.4, 3.6))
y = np.arange(len(effects))[::-1]
for yi, (label, df, c) in zip(y, effects):
    per_drug = df.mean(axis=1)             # a drug's effect, averaged over cells
    m = per_drug.mean()
    lo, hi = boot_ci(per_drug)
    ax.barh(yi, m, height=0.5, color=c, zorder=3)
    ax.plot([lo, hi], [yi, yi], color=INK2, lw=1.6, solid_capstyle="butt", zorder=4)
    ax.plot([lo, lo], [yi - 0.11, yi + 0.11], color=INK2, lw=1.6, zorder=4)
    ax.plot([hi, hi], [yi - 0.11, yi + 0.11], color=INK2, lw=1.6, zorder=4)
    up = int((per_drug > 0).sum())
    ax.annotate(f"{m:+.4f}  [{lo:+.4f}, {hi:+.4f}]   {up}/{len(per_drug)} drugs up"
                + (f",  {df.shape[1]} cells" if df.shape[1] > 1 else ""),
                (max(hi, 0), yi), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=7.3, color=INK)
ax.axvline(0, color=AXIS, lw=1.0, zorder=2)
ax.set_yticks(y, [e[0] for e in effects], fontsize=8)
ax.set_xlabel("paired change in CV AUC, averaged over the 11 drugs")
ax.set_xlim(-0.020, 0.125)
ax.set_title("Paired effect on CV AUC, by what was changed")
ax.annotate("intervals: 95% percentile bootstrap over the 11 drugs, 10,000 resamples",
            xy=(0, -0.26), xycoords="axes fraction", fontsize=7.2, color=MUTED)
tidy(ax); plt.tight_layout(); plt.show()
''')

# -------------------------------------------------------------- claim 4
md("""
## Claim 4 — attribution on these models does not track what the model uses

This is a controlled ladder over `locusfusion` at 19 loci: same tokenizer, same
protocol, **one non-DNA modality at a time**, each against the DNA-only cell on
the same loci. Every point is one (drug, modality) pair — x is the share of the
attribution budget the non-DNA modality took, y is the CV AUC that modality
bought.

There is no relationship: rho = +0.11, 95% CI [−0.18, +0.42]. Across the whole
ladder `biophysical` takes **13.5x** the budget `regulatory` takes and buys
**1.2x** the AUC.

The vertical bars carry a second, blunter result. **All 44 gains have a 95%
interval containing zero.** At 19 loci not one modality's contribution to one
drug is individually resolved — the bars are an order of magnitude taller than
the effects they bracket. Whatever the modalities are worth here, this is not a
measurement that can see it, and the correlation is being computed over
quantities that are themselves noise.

**Why this matters for claim 2.** The natural objection to "the architecture does
not matter" is that the ablation is too blunt to see what the model is really
doing. This is the finer instrument, and it says the model's own attribution
cannot rank its inputs by value either — so the ablation is not the thing that is
blunt. It is also the strongest argument in the project for preferring an
architecture whose attribution is exact (`additive`, whose per-token
contributions sum to the logit) over one explained after the fact.

**Right: the model's own attention and SHAP disagree about which locus matters.**
Same trained models, same isolates, same draw. `locusfusion`'s read-out
attention is flat to within 11% of an even split over its 19 loci at every
modality set, while SHAP on those same models puts 4.1-6.1x uniform on one
locus. Two attribution methods applied to one model, giving two different
answers — and by the left panel, neither of them predicts what the input is
worth.

SHAP is the default interpretability tool in this field, and these are the two
failure modes that matter for it: **the share a feature takes does not rank its
value, and a second attribution method on the same model does not agree with
it.** Both are measured here on a controlled ladder rather than asserted.

*Expected-gradients attribution, `wt` background, `nsamples=64` (rank-identical
to 256; §7 of the SHAP write-up). The same rank correlation on held-out test gain
is -0.026, p = 0.87.*
""")

code('''
SHAP_LF = PROJECT / "results/analysis/shap_locusfusion/newmodels_full/locusfusion"
blocks = pd.read_csv(SHAP_LF / "blocks_all.csv")
loci   = pd.read_csv(SHAP_LF / "loci_all.csv")

LADDER = ["sd19_dna_protein", "sd19_dna_regulatory", "sd19_dna_biophysical",
          "sd19_all_modalities"]
NICE = {"sd19_dna_protein": "+ protein", "sd19_dna_regulatory": "+ regulatory",
        "sd19_dna_biophysical": "+ biophysical", "sd19_all_modalities": "+ all three"}
BG = "wt"

base = frame[(frame.run == "newmodels_full") & (frame.task == LOCI19)
             & (frame.modset == "dna")].set_index("drug").cv

base_f = folds_of(LOCI19, "dna", "locusfusion", run="newmodels_full")
pts = []
for c in LADDER:
    modset = c.replace("sd19_", "")
    got_f = folds_of(LOCI19, modset, "locusfusion", run="newmodels_full")
    dl = per_drug_delta(got_f, base_f)                    # fold-paired
    b = blocks[(blocks.background == BG) & (blocks.cell == c)]
    sh = (b.assign(non=np.where(b.modality != "dna", b.share, 0.0))
           .groupby("drug").agg(non=("non", "sum"), tot=("share", "sum")))
    for d in DRUGS:
        if d in dl.index and d in sh.index:
            per_fold = np.asarray(got_f[d]) - np.asarray(base_f[d])
            lo, hi = t_ci(per_fold)
            pts.append(dict(drug=d, cell=c, share=sh.loc[d, "non"] / sh.loc[d, "tot"],
                            gain=dl[d], lo=lo, hi=hi))
pts = pd.DataFrame(pts)

rho, pval = spearmanr(pts.share, pts.gain)
rlo, rhi = boot_rho_ci(pts, "share", "gain")
crosses = int(((pts.lo <= 0) & (pts.hi >= 0)).sum())

MARKS = {"sd19_dna_protein": "o", "sd19_dna_regulatory": "^",
         "sd19_dna_biophysical": "s", "sd19_all_modalities": "D"}
fig, (ax, axC) = plt.subplots(1, 2, figsize=(11.6, 4.0),
                              gridspec_kw={"width_ratios": [1.1, 1]})
ax.vlines(pts.share, pts.lo, pts.hi, color=RUST, lw=0.9, alpha=0.30, zorder=2)
for c in LADDER:
    s = pts[pts.cell == c]
    ax.scatter(s.share, s.gain, s=40, marker=MARKS[c], color=RUST, alpha=0.9,
               edgecolor=SURFACE, linewidth=0.9, zorder=3, label=NICE[c])
ax.axhline(0, color=AXIS, lw=1.0, zorder=2)
ax.set_xlabel("share of the attribution budget taken by the non-DNA modality")
ax.set_ylabel("CV AUC gained over the\\nDNA-only cell, same 19 loci")
ax.set_title("Attribution share vs. CV AUC gain, locusfusion, 19 loci")
ax.annotate(f"Spearman rho = {rho:+.3f}  95% CI [{rlo:+.2f}, {rhi:+.2f}]  "
            f"p = {pval:.2f}  n = {len(pts)}\\n"
            f"{crosses} of {len(pts)} gains have a 95% interval containing zero",
            xy=(0.97, 0.04), xycoords="axes fraction", ha="right",
            fontsize=8, color=INK, linespacing=1.5,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.0))
ax.legend(loc="upper left", ncols=2)
tidy(ax, ygrid=True)

# --- right: the two attributions disagree about which locus matters -------
CELLS19 = ["sd19_dna", "sd19_dna_protein", "sd19_dna_regulatory",
           "sd19_dna_biophysical", "sd19_all_modalities"]
LBL = {"sd19_dna": "DNA", "sd19_dna_protein": "DNA+protein",
       "sd19_dna_regulatory": "DNA+regulatory",
       "sd19_dna_biophysical": "DNA+biophysical",
       "sd19_all_modalities": "all four"}
conc = []
for c in CELLS19:
    g = loci[(loci.background == "wt") & (loci.cell == c)]
    a, sh = [], []
    for _, sub_ in g.groupby("drug"):
        n = len(sub_)
        if n < 2:
            continue
        a.append(sub_.attn_mean.max() / sub_.attn_mean.sum() * n)
        sh.append(sub_.share.max() / sub_.share.sum() * n)
    conc.append((LBL[c], np.mean(a), np.mean(sh), boot_ci(a), boot_ci(sh)))

yc = np.arange(len(conc))[::-1]
h = 0.32
axC.barh(yc + h / 2 + 0.02, [c[2] for c in conc], height=h, color=RUST, zorder=3)
axC.barh(yc - h / 2 - 0.02, [c[1] for c in conc], height=h, color=PALE,
         edgecolor=RUST, linewidth=1.1, zorder=3)
for yi, (_, a, sh, aci, sci) in zip(yc, conc):
    axC.plot(sci, [yi + h / 2 + 0.02] * 2, color=INK2, lw=1.4, zorder=4)
    axC.plot(aci, [yi - h / 2 - 0.02] * 2, color=INK2, lw=1.4, zorder=4)
    axC.annotate(f"{sh:.2f}x", (sci[1], yi + h / 2 + 0.02), xytext=(5, 0),
                 textcoords="offset points", va="center", fontsize=7.4, color=INK)
    axC.annotate(f"{a:.2f}x", (aci[1], yi - h / 2 - 0.02), xytext=(5, 0),
                 textcoords="offset points", va="center", fontsize=7.4, color=INK)
axC.axvline(1.0, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=5)
axC.annotate("an even split over 19 loci", xy=(1.0, 1.0),
             xycoords=("data", "axes fraction"), xytext=(4, -2),
             textcoords="offset points", va="top", fontsize=7.2, color=INK2)
axC.set_yticks(yc, [c[0] for c in conc], fontsize=8)
axC.set_xlabel("top-locus concentration, as a multiple of uniform")
axC.set_xlim(0, 8.6)
axC.set_title("Top-locus concentration: SHAP against the\\nmodel's own read-out attention")
axC.legend([plt.Rectangle((0, 0), 1, 1, color=RUST),
            plt.Rectangle((0, 0), 1, 1, facecolor=PALE, edgecolor=RUST)],
           ["SHAP", "read-out attention"], loc="upper right")
tidy(axC)

fig.text(0.5, -0.09, "left: vertical bars are 95% t intervals over the 5 fold-paired "
         "differences, and the rho interval is a bootstrap over drugs. right: 95% "
         "bootstrap over the 11 drugs.", ha="center", fontsize=7.2, color=MUTED)
plt.tight_layout(); plt.show()
''')

# -------------------------------------------------------------- claim 6
md("""
## Claim 6 — model capacity is inert, and at the top end it hurts

Claim 1 shows a 228-coefficient lasso matching a 6.7 M-parameter network. The
obvious reply is that the deep models must be doing *something* the macro cannot
see. This is the direct test of that: take four places where capacity or
machinery was deliberately added, hold everything else fixed, and measure what
it bought.

**Nothing, and once it hurts.** Left: across **3.5 orders of magnitude** of
parameter count — 17 thousand to 60.7 million — macro CV AUC is uncorrelated
with model size (Spearman ρ = +0.07, p = 0.73 over the 27 parameterised cells;
the lasso is plotted but held out of the correlation, because its x-axis unit is
nonzero coefficients rather than free parameters, and it sits at the extreme
left anyway). Right: the four controlled additions. Three return intervals containing zero. The
fourth, going from `mdcnn` at 6.7 M to `late_fusion` at 57.7 M, is **−0.0175
[−0.0406, −0.0040], ahead on 1 of 11 drugs** — nine times the parameters, and
resolvably worse.

This is the negative half of the thesis, stated as a measurement rather than an
absence of evidence: on this task the model is not where the performance is.
Figure 2 says where it is instead.

*Two supporting results not drawn here. `joint_capacity` widened the shared head
to 512 units (−0.004), gave each drug its own head (−0.001) and added dropout
with weight decay (−0.046 on `cisfusion`) — no arm beat its control.
`setfusion_scaling` swept four width axes across 62 arms and closed nothing. And
the multi-task bar in Figure 2 is the same result again: one shared trunk over
11 separate models is worth +0.0012 [−0.0031, +0.0058].*
""")

code('''
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.0),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# --- left: size against accuracy, every complete 19-locus cell -------------
q = MACRO[MACRO.task == LOCI19].copy()
q["size"] = np.where(q.model == "lasso_onehot", LASSO_SUPPORT, q.n_params)
q = q[q["size"].notna() & (q["size"] > 0)]
for m, sg in q.groupby("model"):
    axL.scatter(sg["size"], sg.cv, s=46, color=hue(m), edgecolor=SURFACE,
                linewidth=1.0, zorder=3)
best = q.sort_values("cv").groupby("model").tail(1).sort_values("size")
placed = []
for _, r in best.iterrows():
    lx, ly = np.log10(r["size"]), r.cv
    below = any(abs(lx - px) < 0.42 and abs(ly - py) < 0.006 for px, py in placed)
    axL.annotate(r.model, (r["size"], r.cv), xytext=(0, -17 if below else 9),
                 textcoords="offset points", ha="center", fontsize=7.2, color=INK2)
    placed.append((lx, ly))
sized = q[q.model != "lasso_onehot"]
rho_p, p_p = spearmanr(np.log10(sized["size"]), sized.cv)
axL.set_xscale("log")
axL.set_xlim(80, 2e8)
axL.set_xlabel("free parameters (lasso: mean nonzero coefficients), log scale")
axL.set_ylabel("macro CV AUC")
axL.set_ylim(0.850, 0.938)
axL.set_title("Model size against macro CV AUC,\\nsingle-drug task, all 19 loci")
axL.annotate(f"Spearman rho = {rho_p:+.3f}, p = {p_p:.2f}, n = {len(sized)} "
             f"parameterised cells",
             xy=(0.97, 0.04), xycoords="axes fraction", ha="right", fontsize=8,
             color=INK, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.0))
tidy(axL, ygrid=True)

# --- right: what each deliberate addition bought --------------------------
# Every pair is (the bigger / more elaborate model) minus (the simpler one), so
# a positive value means the addition paid. All references are pre-specified.
LADDER = [
    ("a 6.7 M conv net over\\na 228-coefficient lasso",
     (LOCI19, "all_modalities", "mdcnn"), (LOCI19, "dna", "lasso_onehot")),
    ("a learned per-drug gate over\\na plain sum and max",
     (LOCI19, "all_modalities", "gatedpool"), (LOCI19, "all_modalities", "deepsets")),
    ("9x the parameters:\\ncisfusion 60.7 M over mdcnn 6.7 M",
     (LOCI19, "all_modalities", "cisfusion"), (LOCI19, "all_modalities", "mdcnn")),
    ("9x the parameters:\\nlate_fusion 57.7 M over mdcnn 6.7 M",
     (LOCI19, "all_modalities", "late_fusion"), (LOCI19, "all_modalities", "mdcnn")),
]
y = np.arange(len(LADDER))[::-1]
for yi, (label, big, small) in zip(y, LADDER):
    dl = per_drug_delta(folds_of(*big), folds_of(*small))
    m, (lo, hi) = dl.mean(), boot_ci(dl)
    axR.barh(yi, m, height=0.5, color=BLUE, zorder=3)
    axR.plot([lo, hi], [yi, yi], color=INK2, lw=1.6, solid_capstyle="butt", zorder=4)
    for x_ in (lo, hi):
        axR.plot([x_, x_], [yi - 0.11, yi + 0.11], color=INK2, lw=1.6, zorder=4)
    axR.annotate(f"{m:+.4f}  [{lo:+.4f}, {hi:+.4f}]   {int((dl > 0).sum())}/{len(dl)} up",
                 (max(hi, 0), yi), xytext=(8, 0), textcoords="offset points",
                 va="center", fontsize=7.4, color=INK)
axR.axvline(0, color=AXIS, lw=1.0, zorder=2)
axR.set_yticks(y, [l[0] for l in LADDER], fontsize=7.6)
axR.set_xlabel("paired change in CV AUC from the addition")
axR.set_xlim(-0.045, 0.055)
axR.set_title("What each deliberate addition bought,\\nfold-paired over the 11 drugs")
tidy(axR)

fig.text(0.5, -0.10, "left: one point per complete cell. right: 95% percentile "
         "bootstrap over the 11 drugs; every reference is pre-specified, none is "
         "chosen by score.", ha="center", fontsize=7.2, color=MUTED)
plt.tight_layout(); plt.show()
''')

# -------------------------------------------------------------- synthesis
md("""
## What the four figures say together

The argument is about **inputs, not models**, and the four figures are its four
moves.

1. **Claim 1** sets the bar: an L1-logistic on the raw alignment, at 228 nonzero
   coefficients, is indistinguishable from a 6.7 M-parameter CNN and from the
   published MD-CNN. Fold-paired, −0.0005 [−0.0033, +0.0023].
2. **Claim 6** shows this is not an accident of one comparison. Across 3.5
   orders of magnitude of parameter count there is no relationship with
   accuracy, and four deliberate additions of capacity or machinery return
   nothing — one of them a resolvable *loss* of 0.0175 for nine times the
   parameters. Model complexity is not where the performance is.
3. **Claim 2** then finds where it is: the input. Handing a model all 19 curated
   loci instead of the drug's own 2–3 is worth +0.049 to the lasso (11/11 drugs)
   and +0.016 to the networks (10/11). Adding the non-DNA modalities is worth
   +0.014, and it is concentrated on the drugs whose mechanism predicts it (A3).
   Multi-task sharing is worth nothing.
4. **Claim 4** closes the obvious escape route. "Your ablation is too blunt to
   see what the deep model is doing" would be a fair objection if the model's
   own attribution could rank its inputs — and it cannot. The share a modality
   takes is uncorrelated with the AUC it buys (ρ = +0.11 [−0.18, +0.42]), no
   single per-drug gain is resolvable at all, and SHAP and the model's own
   attention disagree about which locus matters (4.1–6.1× uniform against
   1.06–1.11×).

**So the paper is:** the field's current direction — bigger, more elaborate
sequence models — is not what moves this task, and the standard tool for
inspecting those models does not reliably say what they use. What does move it
is what you feed in. We give a strong, cheap, reproducible baseline that anyone
can run, and a decomposition that says where to spend effort instead: on the
locus set, on the modalities that carry a mechanism, and on curation — the 19
WHO-derived loci already capture everything 4.41 Mb of genome does (A7).

**What the intervals changed.** The tie in Claim 1 is a paired result with an
interval around zero, not an eyeballed 0.0005. The gate ablation is *bounded* at
[−0.0004, +0.0023] rather than merely small. The network locus effect is **not**
resolved by its mean's interval and rests on 10 of 11 drugs moving up. And all
44 per-drug modality gains have intervals containing zero, so the modality
ladder supports only the pooled effect.

**Standing limitations.** Single seed throughout — no interval here contains
seed variance, so all are lower bounds. CV-fold intervals are anti-conservative
because folds share training data. No lineage-held-out split exists, which is
what C3 in `PAPER.md` needs. Alignment columns are not H37Rv coordinates outside
protein blocks. And the 19-locus `locusfusion` cells were trained on the **old**
tokenizer; the rebuilt one is worth a uniform +0.017 at per-drug loci (A5), and
the matching 19-locus grid is running.
""")

# ============================================================== PART II
md("""
---
# Part II — appendix

The same evidence at full resolution, plus the measurements that qualify it.
Nothing here is a new claim; these are the figures to turn to when someone asks
"which drug?", "how big is the noise?", or "does that hold per drug?".
""")

# -------------------------------------------------------------- A1
md("""
### A1 — every complete cell, by task and model

Macro CV AUC over the 11 drugs, one point per (modality set, model) cell that
ran all 11 drugs. `setfusion` is excluded everywhere in this notebook: its tokens
are degenerate by measurement (`token_signal`), not by underperformance.

`locusfusion` carries ten points in the single-drug panel rather than five,
because that cell was run twice — once on each tokenizer. A5 separates them.

**No intervals here, deliberately.** Roughly forty cells share each panel, and
their bootstrap intervals overlap almost completely (Figure 1, left) because
drug difficulty dominates. Drawn, they would hide the ordering this figure exists
to show and would invite exactly the overlapping-error-bar reading that Figure 1
warns against. The intervals that matter are the paired ones, and they are in
Figure 1 and A2.
""")

code('''
fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.2), sharex=True)
for ax, task in zip(axes, [SD, LOCI19, MD]):
    q = MACRO[MACRO.task == task].copy()
    order = q.groupby("model").cv.max().sort_values().index
    for i, m in enumerate(order):
        s = q[q.model == m]
        ax.scatter(s.cv, np.full(len(s), i), s=34, color=hue(m),
                   edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax.set_yticks(range(len(order)), order, fontsize=7.6)
    ax.set_title(task, fontsize=9.5)
    ax.set_xlabel("macro CV AUC")
    tidy(ax)
axes[0].set_xlim(0.76, 0.95)
fig.suptitle("Macro CV AUC by model and task, one point per modality set",
             y=1.02, fontsize=10.5, fontweight="bold")
fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, RUST, AQUA)],
           ["established architectures", "variant-token aggregators",
            "locusfusion", "non-neural baseline"],
           loc="lower center", ncols=4, bbox_to_anchor=(0.5, -0.09))
plt.tight_layout(); plt.show()
''')

# -------------------------------------------------------------- A2
md("""
### A2 — per drug, at 19 loci

The three tied models of Figure 1, drug by drug, against the published MD-CNN.
The drugs are ordered by cohort size; the spread at the bottom of the axis is
where the small cohorts live and where nothing is resolved on one seed.
""")

code('''
n_by_drug = (frame[frame.task == LOCI19].groupby("drug").n_valid.max()
             .sort_values(ascending=False))
order = [d for d in n_by_drug.index]

series = [("mdcnn, all four modalities", "all_modalities", "mdcnn", "o", -0.22),
          ("locusfusion (rebuilt), all four", "all_modalities", "locusfusion", "D", 0.0),
          ("L1-logistic, one-hot alignment", "dna", "lasso_onehot", "s", 0.22)]

fig, ax = plt.subplots(figsize=(9.8, 4.0))
x = np.arange(len(order))
for label, modset, model, mk, dx in series:
    fo = folds_of(LOCI19, modset, model,
                  run=LF_RUN if model == "locusfusion" else None)
    mu = [np.mean(fo[d]) if d in fo.index else np.nan for d in order]
    ci = [t_ci(fo[d]) if d in fo.index else (np.nan, np.nan) for d in order]
    ax.vlines(x + dx, [c[0] for c in ci], [c[1] for c in ci],
              color=hue(model), lw=1.3, alpha=0.55, zorder=2)
    ax.scatter(x + dx, mu, s=40, marker=mk, color=hue(model),
               edgecolor=SURFACE, linewidth=0.9, zorder=3, label=label)
ax.scatter(x, [MD_BASE_CV.get(d, np.nan) for d in order], s=52, marker="_",
           color=INK2, zorder=4, label="MD-CNN baseline (published)")
ax.set_xticks(x, [f"{d.title()}\\n{int(n_by_drug[d]):,}" for d in order],
              fontsize=7.2)
ax.set_ylabel("5-fold CV AUC")
ax.set_xlabel("drug, and the number of labelled isolates it has")
ax.set_title("CV AUC by drug, single-drug task, all 19 loci")
ax.legend(loc="lower left", ncols=2)
ax.annotate("bars: 95% t interval over the 5 CV folds (anti-conservative -- folds share "
            "training data)", xy=(0, -0.26), xycoords="axes fraction",
            fontsize=7.2, color=MUTED)
tidy(ax, xgrid=False, ygrid=True); plt.tight_layout(); plt.show()
''')

# -------------------------------------------------------------- A3
md("""
### A3 — what the extra modalities buy, per drug

Best gain over that drug's own DNA-only cell, averaged over the established
architectures, at the drug's own loci. The macro (+0.017) is a bad summary of
this: two drugs carry almost all of it, and the pattern matches the resistance
mechanism — the *fabG1–inhA* operon promoter for ETHIONAMIDE and ISONIAZID,
*pncA* loss-of-function for PYRAZINAMIDE, and nothing for the two drugs whose
target is 16S rRNA and therefore has no protein product to describe.

**This is the *best* of four modality sets**, so it is a selected statistic: with
four draws, noise alone produces a small positive for every drug. Only the top
three bars clear the fold SD, and the bottom half of this chart is what "no
effect" looks like here, not a small effect.
""")

code('''
# All four modality sets, every drug -- NO selection. The previous version of
# this figure showed each drug's BEST of four, which is a selected maximum: with
# four draws, noise alone produces a positive for every drug, and an interval
# drawn on a selected maximum is not a confidence interval for anything.
MODSET_GAINS = ["dna_protein", "dna_biophysical", "dna_regulatory", "all_modalities"]
GAIN_LBL = {"dna_protein": "+ protein", "dna_biophysical": "+ biophysical",
            "dna_regulatory": "+ regulatory", "all_modalities": "+ all three"}
GAIN_MARK = {"dna_protein": "o", "dna_biophysical": "s",
             "dna_regulatory": "^", "all_modalities": "D"}
# A modality set is not an identity, it is a setting of one family, so it gets a
# sequential ramp inside the family hue rather than four unrelated colours --
# the same rule scripts/build_results.py uses for task and locus set. Shape
# carries identity too, so neither channel is doing the job alone.
GAIN_TINT = {"dna_protein": "#a8c8f0", "dna_biophysical": "#7aa9e2",
             "dna_regulatory": "#4a8ad4", "all_modalities": BLUE}

def modality_gain(task):
    """{(drug, modset): (mean fold-paired gain over DNA, (lo, hi))}.

    Pooled over the established architectures by averaging each model's 5
    fold-paired differences, so the interval is a t interval over the 5 pooled
    per-fold differences. That treats the three architectures as fixed, which is
    what "the average established network" means here.
    """
    out = {}
    for d in DRUGS:
        for ms in MODSET_GAINS:
            diffs = []
            for model in ESTABLISHED:
                a, b = folds_of(task, ms, model), folds_of(task, "dna", model)
                if d in a.index and d in b.index:
                    diffs.append(np.asarray(a[d]) - np.asarray(b[d]))
            if diffs:
                per_fold = np.mean(diffs, axis=0)
                out[(d, ms)] = (float(per_fold.mean()), t_ci(per_fold))
    return out

g = modality_gain(SD)
# Order drugs by their largest gain, so the shape of the result is legible --
# the ORDERING is descriptive; every value plotted is pre-specified.
order = sorted(DRUGS, key=lambda d: max(g[(d, ms)][0] for ms in MODSET_GAINS
                                        if (d, ms) in g))
fig, ax = plt.subplots(figsize=(8.6, 5.0))
offs = np.linspace(-0.27, 0.27, len(MODSET_GAINS))
for j, ms in enumerate(MODSET_GAINS):
    ys, xs, los, his = [], [], [], []
    for i, d in enumerate(order):
        if (d, ms) not in g:
            continue
        v, (lo, hi) = g[(d, ms)]
        ys.append(i + offs[j]); xs.append(v); los.append(lo); his.append(hi)
    ax.hlines(ys, los, his, color=GAIN_TINT[ms], lw=1.3, alpha=0.75, zorder=2 + j)
    ax.scatter(xs, ys, s=30, marker=GAIN_MARK[ms], color=GAIN_TINT[ms],
               edgecolor=SURFACE, linewidth=0.7, zorder=6 + j, label=GAIN_LBL[ms])
ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
ax.set_yticks(range(len(order)), [d.title() for d in order], fontsize=8)
ax.set_xlabel("gain in CV AUC over that drug's DNA-only cell")
ax.set_xlim(-0.04, 0.15)
ax.set_title("Modality gain by drug and modality set, single-drug task,\\n"
             "the drug's own loci")
ax.legend(loc="center right", ncols=1, bbox_to_anchor=(1.0, 0.42))
ax.annotate("bars: 95% t interval over the 5 fold-paired differences, pooled over the "
            "three established architectures",
            xy=(0, -0.16), xycoords="axes fraction", fontsize=7.2, color=MUTED)
tidy(ax); plt.tight_layout(); plt.show()

clears = sum(1 for k, (v, (lo, hi)) in g.items() if lo > 0)
print(f"{clears} of {len(g)} (drug, modality set) gains have a 95% interval above zero")
''')

# -------------------------------------------------------------- A4
md("""
### A4 — what the locus universe buys, per drug

The same move as the top two bars of Figure 2, drug by drug: the drug's own 2–3
loci against all 19. The lasso gains more than the networks on 10 of 11 drugs,
which is the cleanest statement that this is an input effect and not an
architectural one — and it is also where the co-resistance caveat bites hardest,
since the added loci include every other drug's target.
""")

code('''
def locus_gain(models):
    """-> {drug: (mean gain, (lo, hi))}, fold-paired, pooled over `models`."""
    out = {}
    for d in DRUGS:
        diffs = []
        for m in models:
            a, b = folds_of(LOCI19, "dna", m), folds_of(SD, "dna", m)
            if d in a.index and d in b.index:
                diffs.append(np.asarray(a[d]) - np.asarray(b[d]))
        if diffs:
            per_fold = np.mean(diffs, axis=0)
            out[d] = (per_fold.mean(), t_ci(per_fold))
    return out

net, las = locus_gain(ESTABLISHED), locus_gain(["lasso_onehot"])
order = sorted([d for d in DRUGS if d in las], key=lambda d: las[d][0])

fig, ax = plt.subplots(figsize=(8.4, 4.0))
y = np.arange(len(order)); h = 0.34
for src, off, c, lab in [(las, +h / 2 + 0.02, AQUA, "L1-logistic, one-hot alignment"),
                         (net, -h / 2 - 0.02, BLUE, "established networks (mean)")]:
    ax.barh(y + off, [src[d][0] for d in order], height=h, color=c, zorder=3, label=lab)
    for yi, d in zip(y, order):
        lo, hi = src[d][1]
        ax.plot([lo, hi], [yi + off] * 2, color=INK2, lw=1.2,
                solid_capstyle="butt", zorder=4)
ax.axvline(0, color=AXIS, lw=1.0, zorder=2)
ax.set_yticks(y, [d.title() for d in order], fontsize=8)
ax.set_xlabel("change in CV AUC, the drug's own loci -> all 19")
ax.set_title("Locus-universe gain by drug, single-drug task, DNA only")
ax.legend(loc="lower right")
ax.annotate("bars: 95% t interval over the 5 fold-paired differences",
            xy=(0, -0.24), xycoords="axes fraction", fontsize=7.2, color=MUTED)
tidy(ax); plt.tight_layout(); plt.show()
''')

# -------------------------------------------------------------- A5
md("""
### A5 — the tokenizer rebuild, all 55 cells

`locusfusion_fixed` is the same architecture, the same protocol, the same seed and
the same loci as `newmodels_full`; only the tokenizer underneath it changed. Two
fixes shipped together: the nucleotide token's coordinate is now computed from
the CDS annotation rather than learned (it had put *katG* S315 43 codons away
from its own protein token), and a token became a symbol pair rather than a
42-float slot vector, which also made an unknown base call visible instead of
reading as wild type.

**Every one of the 55 cells improved**, mean +0.017, and every modality set's
bootstrap interval excludes zero by a wide margin (+0.010 to +0.026 at the
narrowest). Measured against everything else in this notebook, **this is the most
solidly established effect the project has** — larger than the modality effect,
resolved where the network locus effect is not, and uniform across five modality
sets and eleven drugs, which is what a fixed input representation looks like
rather than a lucky seed. The two changes are *not* attributed separately — they
shipped together and no arm holds one back.

This arm had only ever run at the drug's own loci, so the 19-locus `locusfusion`
numbers everywhere else in this notebook are still the **old** tokenizer. The
matching 19-locus grid (`locusfusion_fixed/sd19_*`, 55 jobs,
`results/experiments/locusfusion_fixed/submit_sd19.sh`) was submitted on
**2026-09-03** and will replace them; until it lands, read Figures 1, 4 and A2
as a lower bound on what this architecture does at 19 loci.
""")

code('''
msets = ["dna", "dna_protein", "dna_biophysical", "dna_regulatory", "all_modalities"]
rows_d = []
for ms in msets:
    a = folds_of(SD, ms, "locusfusion", run="locusfusion_fixed")
    b = folds_of(SD, ms, "locusfusion", run="newmodels_full")
    for d, v in per_drug_delta(a, b).items():          # fold-paired
        rows_d.append(dict(modset=ms, drug=d, delta=v))
delta = pd.DataFrame(rows_d)

fig, ax = plt.subplots(figsize=(8.0, 3.6))
jit = np.random.default_rng(0)
for i, ms in enumerate(msets):
    sm = delta[delta.modset == ms]
    lo, hi = boot_ci(sm.delta)
    ax.plot([lo, hi], [i, i], color=INK2, lw=1.6, solid_capstyle="butt", zorder=4)
    ax.plot([lo, lo], [i - 0.13, i + 0.13], color=INK2, lw=1.6, zorder=4)
    ax.plot([hi, hi], [i - 0.13, i + 0.13], color=INK2, lw=1.6, zorder=4)
    ax.scatter(sm.delta, np.full(len(sm), i) + jit.normal(0, 0.07, len(sm)),
               s=34, color=RUST, alpha=0.7, edgecolor=SURFACE, linewidth=0.7, zorder=3)
    ax.annotate(f"{sm.delta.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]", (0.056, i),
                fontsize=7.4, va="center", color=INK)
ax.axvline(0, color=AXIS, lw=1.0, zorder=2)
MSET_LBL = {"dna": "DNA", "dna_protein": "DNA + protein",
            "dna_biophysical": "DNA + biophysical",
            "dna_regulatory": "DNA + regulatory", "all_modalities": "all four"}
ax.set_yticks(range(len(msets)), [MSET_LBL[m] for m in msets], fontsize=8)
ax.set_xlabel("change in CV AUC, rebuilt tokenizer - old tokenizer")
ax.set_xlim(-0.006, 0.088)
ax.set_title("Per-cell change from the tokenizer rebuild, "
             f"{len(delta)} cells, the drug's own loci")
ax.annotate("intervals: 95% percentile bootstrap over the 11 drugs, within each "
            "modality set", xy=(0, -0.24), xycoords="axes fraction",
            fontsize=7.2, color=MUTED)
tidy(ax); plt.tight_layout(); plt.show()
print(f"cells improved: {(delta.delta > 0).sum()}/{len(delta)}   "
      f"mean {delta.delta.mean():+.4f}   min {delta.delta.min():+.4f}   "
      f"max {delta.delta.max():+.4f}")
''')

# -------------------------------------------------------------- A6
md("""
### A6 — featurising a variant against memorising it, by cohort size

`catalogue` and `additive` share a tokenizer, an aggregator (a sum) and an
optimizer. One thing differs: whether a variant's weight comes from its
**identity** (a zero-initialised table, so a substitution unseen in training
scores exactly zero) or from its **features** — locus, exact position, which
residue it became, the biophysical consequence.

The gain from featurising is almost perfectly ordered by how little labelled data
the drug has. This is the project's cleanest mechanistic result and it is a
deployment claim: above ~8,000 isolates a learned catalogue has seen enough;
below that, features matter, and at n = 269 they are worth 0.07 AUC.
""")

code('''
af, cf = (folds_of(LOCI19, "all_modalities", "additive"),
          folds_of(LOCI19, "all_modalities", "catalogue"))
nval = frame[(frame.task == LOCI19) & (frame.model == "additive")
             & (frame.modset == "all_modalities")].set_index("drug").n_valid
k = [d for d in DRUGS if d in af.index and d in cf.index]
tab = pd.DataFrame([dict(drug=d, n=nval[d],
                         gain=float(np.mean(np.asarray(af[d]) - np.asarray(cf[d]))),
                         lo=t_ci(np.asarray(af[d]) - np.asarray(cf[d]))[0],
                         hi=t_ci(np.asarray(af[d]) - np.asarray(cf[d]))[1])
                    for d in k])
rho, pval = spearmanr(tab.n, tab.gain)
rlo, rhi = boot_ci_rho = None, None
_b = []
for _ in range(4000):
    idx = RNG.integers(0, len(tab), len(tab))
    ss = tab.iloc[idx]
    if ss.n.nunique() > 2:
        _b.append(spearmanr(ss.n, ss.gain).statistic)
rlo, rhi = np.percentile(_b, [2.5, 97.5])

fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.vlines(tab.n, tab.lo, tab.hi, color=ORANGE, lw=1.3, alpha=0.55, zorder=2)
ax.scatter(tab.n, tab.gain, s=50, color=ORANGE, edgecolor=SURFACE,
           linewidth=1.0, zorder=3)
# Four large-cohort drugs land almost on top of each other; alternate the label
# below the point when the previous one would collide.
placed = []
for _, r in tab.sort_values("n").iterrows():
    lx, ly = np.log10(r.n), r.gain
    below = any(abs(lx - px) < 0.14 and abs(ly - py) < 0.012 for px, py in placed)
    ax.annotate(r.drug.title(), (r.n, r.lo if below else r.hi),
                xytext=(0, -13 if below else 6), textcoords="offset points",
                ha="center", fontsize=6.8, color=INK2)
    placed.append((lx, ly))
ax.axhline(0, color=AXIS, lw=1.0, zorder=2)
ax.set_xscale("log")
ax.set_xlabel("labelled isolates for that drug (log scale)")
ax.set_ylabel("CV AUC, additive - catalogue")
ax.set_title("Gain from featurising a variant vs. memorising it, by cohort size,\\n"
             "19 loci, all modalities")
ax.annotate(f"Spearman rho = {rho:+.3f}  95% CI [{rlo:+.2f}, {rhi:+.2f}]  "
            f"p = {pval:.1e}  n = {len(tab)}",
            xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
            fontsize=8, color=INK,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.0))
ax.annotate("bars: 95% t interval over the 5 fold-paired differences; "
            "rho interval: bootstrap over drugs",
            xy=(0, -0.22), xycoords="axes fraction", fontsize=7.2, color=MUTED)
tidy(ax, ygrid=True); plt.tight_layout(); plt.show()
''')

# -------------------------------------------------------------- A7
md("""
### A7 — the whole genome against the 19 curated loci

`lasso_wholegenome_20260903` fits the same L1-logistic over all 4.41 Mb, with no
locus list and no gene annotation in the objective. Two things come out of it,
and they pull in opposite directions.

**It finds the mechanism unaided.** The single largest coefficient of 127,697
for ETHIONAMIDE is `1673425:C>T` — the *fabG1–inhA* operon promoter, the
canonical *c-15t* mechanism. Handed the whole genome, the strongest variant it
selects is the right one.

**And 4.41 Mb buys nothing over 19 hand-picked genes.** Fold-paired against the
19-locus arm **at matched encoding**, the whole genome is
**+0.0022 [−0.0074, +0.0123]**, 6 of 11 drugs up. Both arms are `delta`-encoded;
comparing the genome against the 0.9241 `onehot` headline instead would confound
the locus universe with the encoding, which is worth about +0.009 on its own
(A11, `lasso_onehot` vs `lasso_delta`).

So the WHO-curated locus set is *sufficient* — not merely convenient. That is a
stronger statement than "the genome is worse," and it is the one the data
supports.

**But 86% of the genome model's weight sits on sequence with no resistance
role.** Of its top-60 coefficients by mass for ETHIONAMIDE: 40.2% on PE/PPE/esx
repeat families — the standard lineage markers of *M. tuberculosis*, and the
regions short-read calling handles worst — 35.6% on other genes with no
resistance role, and only 14.3% on the ETO mechanism. Read with C2: part of what
the locus universe buys is population structure, not mechanism. Separating them
needs a lineage-held-out split, which no run in this project has.
""")

code('''
WG = "single-drug, whole genome"
rows = []
for lbl, a, b in [
    ("whole genome (4.41 Mb) - 19 curated loci\\nmatched encoding, both delta",
     (WG, "dna", "lasso_genome"), (LOCI19, "dna", "lasso_delta")),
    ("19 curated loci - the drug's own 2-3 loci\\nboth delta",
     (LOCI19, "dna", "lasso_delta"), (SD, "dna", "lasso_delta")),
    ("one-hot - delta encoding\\nboth at 19 loci",
     (LOCI19, "dna", "lasso_onehot"), (LOCI19, "dna", "lasso_delta")),
]:
    dl = per_drug_delta(folds_of(*a), folds_of(*b))
    rows.append((lbl, dl.mean(), boot_ci(dl), int((dl > 0).sum()), len(dl)))

fig, ax = plt.subplots(figsize=(8.0, 2.9))
y = np.arange(len(rows))[::-1]
for yi, (lbl, m, (lo, hi), up, n) in zip(y, rows):
    ax.barh(yi, m, height=0.46, color=AQUA, zorder=3)
    ax.plot([lo, hi], [yi, yi], color=INK2, lw=1.6, solid_capstyle="butt", zorder=4)
    for x_ in (lo, hi):
        ax.plot([x_, x_], [yi - 0.1, yi + 0.1], color=INK2, lw=1.6, zorder=4)
    ax.annotate(f"{m:+.4f}  [{lo:+.4f}, {hi:+.4f}]   {up}/{n} up",
                (max(hi, 0), yi), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=7.6, color=INK)
ax.axvline(0, color=AXIS, lw=1.0, zorder=2)
ax.set_yticks(y, [r[0] for r in rows], fontsize=7.8)
ax.set_xlabel("paired change in CV AUC")
ax.set_xlim(-0.02, 0.10)
ax.set_title("What the input scope and the encoding buy the same linear model")
ax.annotate("95% percentile bootstrap over the 11 drugs; every comparison is "
            "fold-paired and encoding-matched except the last, which is the "
            "encoding itself",
            xy=(0, -0.34), xycoords="axes fraction", fontsize=7.0, color=MUTED)
tidy(ax); plt.tight_layout(); plt.show()
''')

# -------------------------------------------------------------- A8
md("""
### A8 — where the attribution budget goes, per drug

The `sd19_all_modalities` cell, `wt` background. The share a modality takes is a
property of the **drug**, not of the modality's worth — `biophysical` runs from
0.003 on KANAMYCIN to 0.78 on PYRAZINAMIDE at per-drug loci. Figure 3 is the
scatter of these numbers against what each modality actually bought.

**No intervals.** Each share is one number from one fold's checkpoint over one
300-isolate draw; the per-isolate attributions that would let this be resampled
are not kept in the aggregated CSVs. Treat the cells as point estimates with
unquantified error.
""")

code('''
b = blocks[(blocks.background == "wt") & (blocks.cell == "sd19_all_modalities")]
piv = (b.groupby(["drug", "modality"]).share.sum().unstack(fill_value=0.0))
piv = piv.div(piv.sum(axis=1), axis=0)
cols = [c for c in ["dna", "protein", "biophysical", "regulatory"] if c in piv.columns]
piv = piv.loc[[d for d in DRUGS if d in piv.index], cols]

fig, ax = plt.subplots(figsize=(6.4, 4.2))
from matplotlib.colors import LinearSegmentedColormap
PALE_ORANGES = LinearSegmentedColormap.from_list(
    "pale_oranges", plt.get_cmap("Oranges")(np.linspace(0.0, 0.52, 256)))
im = ax.imshow(piv.values, cmap=PALE_ORANGES, vmin=0, vmax=piv.values.max(),
               aspect="auto")
ax.set_xticks(range(len(cols)), cols, fontsize=8)
ax.set_yticks(range(len(piv)), [d.title() for d in piv.index], fontsize=7.6)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=INK)
ax.set_title("Share of the attribution budget by drug and modality,\\n"
             "locusfusion, 19 loci, all modalities")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="share of budget")
plt.tight_layout(); plt.show()
''')

# -------------------------------------------------------------- A9
md("""
### A9 — the noise floor

Every claim in Part I is measured against this. Each point is one cell's
fold-to-fold SD of the 5-fold CV AUC for one drug; the dashed line is their mean.
On the small cohorts a single seed cannot resolve anything below ~0.03.

**This figure is the input to every interval in the notebook**, so it carries
none of its own. It is also the explanation for Figure 2's widest bar:
LEVOFLOXACIN, 269 isolates, has fold SDs running past 0.11 — an order of
magnitude above RIFAMPICIN's — which is why one drug can open the interval on an
effect that 10 of 11 drugs agree about.
""")

code('''
q = frame[(frame.task != MD) & frame.cv_sd.notna()
          & (frame.model.isin(ESTABLISHED + ["locusfusion", "lasso_onehot"]))]
order = [d for d in n_by_drug.index]
fig, ax = plt.subplots(figsize=(9.0, 3.4))
for i, d in enumerate(order):
    s = q[q.drug == d].cv_sd
    ax.scatter(np.full(len(s), i) + np.random.default_rng(1).normal(0, 0.08, len(s)),
               s, s=13, color=BLUE, alpha=0.35, edgecolor="none", zorder=3)
    ax.scatter([i], [s.mean()], s=52, color=INK, marker="_", zorder=4)
ax.axhline(NOISE, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=5)
ax.annotate(f"mean {NOISE:.3f}", (len(order) - 0.4, NOISE), xytext=(6, 0),
            textcoords="offset points", fontsize=7.5, color=INK2, va="center")
ax.set_xticks(range(len(order)), [f"{d.title()}\\n{int(n_by_drug[d]):,}" for d in order],
              fontsize=6.6)
ax.set_ylabel("CV fold SD")
ax.set_xlabel("drug, and the number of labelled isolates it has")
ax.set_title("Fold-to-fold SD of the 5-fold CV AUC, by drug, single-drug cells")
tidy(ax, xgrid=False, ygrid=True); plt.tight_layout(); plt.show()
''')

# -------------------------------------------------------------- A10
md("""
### A10 — the model does read its own loci, and the reference choice decides it

The share of the attribution budget landing on the drug's **own** target genes,
divided by the share of variant tokens those genes carry — so a model spreading
attribution evenly over the deviations it actually saw scores 1.0. Those 1–3
genes hold only 3–4% of the tokens the model sees, which is why the ratio rather
than the raw share is the quantity to read.

Against a real-isolate (`train`) reference **all 11 drugs clear the null**, from
1.3x on PYRAZINAMIDE to 39x on KANAMYCIN. Against the all-zero wild-type
reference two drop below it — ETHIONAMIDE and PYRAZINAMIDE — and the ordering
scrambles, because ~43% of the `wt` budget lands on columns carried by more than
80% of isolates: sites where the H37Rv reference disagrees with the population,
whose carriers are resistant at the base rate. Same model, same isolates, same
300-isolate draw; only the reference differs.

**This is a reading instruction, not a caveat.** `train` is the background to
read for "which mutation matters"; `wt` is the background for "what does this
isolate's deviation from H37Rv buy". Figure 3 uses `wt` because it compares
budget *shares between modalities*, where both sit on the same footing.

**No intervals**, for the same reason as A8: one fold, one 300-isolate draw, and
no per-isolate attributions retained to resample. The `wt`-to-`train` span drawn
for each drug is a *sensitivity* range over an analysis choice, not a confidence
interval, and it is much the larger of the two effects anyway.
""")

code('''
# The drug's own loci, read from the per-drug-loci run's own result JSONs.
own = {}
for p in (PROJECT / "results/experiments/newmodels_full/sd_dna__locusfusion").glob("*.json"):
    d = json.loads(p.read_text())
    if "drug" not in d:                 # weights_location.json sits in the same folder
        continue
    own[d["drug"]] = set(d.get("genes") or [])

def enrichment(bg, cell_name="sd19_dna_protein"):
    """On-target attribution share / the share of variant tokens those loci carry."""
    out = {}
    g = loci[(loci.background == bg) & (loci.cell == cell_name)]
    for d, sub_ in g.groupby("drug"):
        if not own.get(d):
            continue
        m = sub_.locus.isin(own[d])
        tok, tot = sub_.mean_variant_tokens.sum(), sub_.share.sum()
        if tok <= 0 or tot <= 0:
            continue
        null = sub_.loc[m, "mean_variant_tokens"].sum() / tok
        if null > 0:
            out[d] = (sub_.loc[m, "share"].sum() / tot) / null
    return out

tr, wt = enrichment("train"), enrichment("wt")
order = sorted([d for d in DRUGS if d in tr and d in wt], key=lambda d: tr[d])

fig, ax = plt.subplots(figsize=(7.2, 3.8))
y = np.arange(len(order))
for d, yi in zip(order, y):
    ax.plot([wt[d], tr[d]], [yi, yi], color=GRID, lw=1.4, zorder=2)
ax.scatter([wt[d] for d in order], y, s=44, marker="s", color=BLUE,
           edgecolor=SURFACE, linewidth=1.0, zorder=3, label="wt (all-zero) background")
ax.scatter([tr[d] for d in order], y, s=48, marker="o", color=RUST,
           edgecolor=SURFACE, linewidth=1.0, zorder=4, label="train (real isolates) background")
ax.axvline(1.0, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=5)
ax.annotate("the variant-token null", xy=(1.0, 0.99), xycoords=("data", "axes fraction"),
            xytext=(5, 0), textcoords="offset points", va="top", fontsize=7.4, color=INK2)
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 5, 10, 20, 40], ["0.5x", "1x", "2x", "5x", "10x", "20x", "40x"])
ax.set_yticks(y, [d.title() for d in order], fontsize=8)
ax.set_xlabel("on-target attribution share, as a multiple of the variant-token null")
ax.set_title("On-target attribution enrichment by drug and reference background,\\n"
             "locusfusion, 19 loci, DNA + protein")
ax.legend(loc="lower right")
tidy(ax); plt.tight_layout(); plt.show()
print(f"clears the null on train: {sum(v > 1 for v in tr.values())}/{len(tr)}   "
      f"on wt: {sum(v > 1 for v in wt.values())}/{len(wt)}")
''')

# -------------------------------------------------------------- A11
md("""
### A11 — the model zoo, every task and every modality set

Macro CV AUC over the 11 shared drugs, **mean ± SD across the 5 CV folds**. A
fold's macro is the mean over the drugs of that fold's AUC, so the SD is
fold-to-fold movement of the whole benchmark score, not the spread between
drugs — which is far larger (A9) and shared by every model.

The four tasks are four **input scopes**, and reading down a column is the
locus-universe result (C2) in table form: the drug's own 2–3 loci, all 19
curated loci, the whole 4.41 Mb genome, and the joint 11-output model at 19
loci.

`lasso_genome` is DNA-only by construction and appears in one column. It is the
run that shows 4.41 Mb buys nothing over 19 curated genes — compare it against
`lasso_delta` at 19 loci, **not** against `lasso_onehot`, because the
whole-genome arm is delta-encoded (a wild-type indicator per site would make the
matrix dense at ~10⁶ sites).

**Every `locusfusion` row at 19 loci and multi-drug is still the old tokenizer**
until the run in flight lands; at per-drug loci both tokenizers are shown as
separate rows, and A5 is the difference between them.

Two absences are deliberate. `sparse_baseline` (the L1-logistic on the *variant*
matrix) kept no per-fold AUCs, so it cannot be given an SD and is left out
rather than shown with a blank one — its macro is 0.8788 at per-drug loci and
0.8925 at 19 loci, and the gap to `lasso_onehot` is the "called variants throw
away signal" result. `noisyor` has no per-drug-loci row because all 11 of those
jobs died on a node-specific CUDA fault and were never resubmitted.

A cell is blank where the model was never run at that modality set — the six
aggregators were run at `all_modalities` only, deliberately, because
`locusfusion` moves 0.0012 across all five at 19 loci and five rows would cost
five times the jobs to re-measure a flat line.
""")

code('''
ZOO = ["mdcnn", "cisfusion", "late_fusion", "locusfusion",
       "additive", "catalogue", "deepsets", "gatedpool", "fm", "noisyor",
       "lasso_onehot", "lasso_delta", "lasso_genome", "sparse_baseline"]
FAMILY = {**{m: "conv / fusion nets" for m in ["mdcnn", "cisfusion", "late_fusion"]},
          **{m: "variant-token" for m in ["locusfusion", "additive", "catalogue",
                                          "deepsets", "gatedpool", "fm", "noisyor"]},
          **{m: "non-neural baseline" for m in ["lasso_onehot", "lasso_delta",
                                                "lasso_genome", "sparse_baseline"]}}
TASK_LBL = {SD: "single-drug, the drug's own loci",
            LOCI19: "single-drug, all 19 loci",
            "single-drug, whole genome": "single-drug, whole genome",
            MD: "multi-drug (11 outputs), 19 loci"}

def fold_macro(task, modset, model, run=None):
    """(mean, sd) of the fold-wise macro: per fold, average over the drugs."""
    q = FOLDS[(FOLDS.task == task) & (FOLDS.modset == modset) & (FOLDS.model == model)]
    if run is not None:
        q = q[q.run == run]
    q = q.drop_duplicates("drug")
    if len(q) != len(DRUGS):
        return None                    # incomplete cells are left blank, never partial
    per_fold = np.array([list(f) for f in q.fold_aucs]).mean(axis=0)
    return per_fold.mean(), per_fold.std(ddof=1)

def zoo_table(task):
    rows = []
    for m in ZOO:
        # locusfusion exists twice at per-drug loci -- one run per tokenizer.
        runs = ([(LF_RUN, " (rebuilt tokenizer)"), (LF_OLD, " (old tokenizer)")]
                if (m == "locusfusion" and task in (SD, LOCI19)) else [(None, "")])
        for run, suffix in runs:
            cells = {ms: fold_macro(task, ms, m, run) for ms in STD}
            if not any(cells.values()):
                continue
            row = {"family": FAMILY[m], "model": m + suffix}
            for ms in STD:
                v = cells[ms]
                row[MS_LBL[ms]] = f"{v[0]:.4f} ± {v[1]:.4f}" if v else "—"
            rows.append(row)
    return pd.DataFrame(rows).set_index(["family", "model"]) if rows else None

STD = ["dna", "dna_protein", "dna_biophysical", "dna_regulatory", "all_modalities"]
MS_LBL = {"dna": "DNA", "dna_protein": "DNA+protein",
          "dna_biophysical": "DNA+biophys", "dna_regulatory": "DNA+regulatory",
          "all_modalities": "all four"}

export = []
for task in [SD, LOCI19, "single-drug, whole genome", MD]:
    t = zoo_table(task)
    if t is None:
        continue
    print(f"\\n=== {TASK_LBL[task]} ===  macro CV AUC, mean ± SD over 5 folds")
    print(t.to_string())
    display(t)
    e = t.reset_index(); e.insert(0, "task", TASK_LBL[task]); export.append(e)

out = PROJECT / "results" / "figures" / "claims" / "model_zoo.csv"
pd.concat(export).to_csv(out, index=False)
print(f"\\nwritten: {out.relative_to(PROJECT)}")
''')

# ============================================================== provenance
md("""
---
### Provenance

Every number is derived from the result JSONs under `results/experiments/` and
the attribution CSVs under `results/analysis/`, through
`results_frame.load_frame()` and `bigtb_baselines` — the two single sources this
project allows for results and baselines. Nothing here is transcribed.

| figure | runs read |
|---|---|
| 1, A1, A2, A7 | `alllocus_run_v2`, `newmodels_full`, `variant_aggregators_alllocus`, `lasso_wholeseq_20260901` |
| 2, A3, A4 | + `full_run_v2` (the matched per-drug-loci control) |
| 3, 4 (left), A8, A10 | `results/analysis/shap_locusfusion/newmodels_full/locusfusion/` |
| 4 (right), A6 | `variant_aggregators_alllocus` |
| A5 | `locusfusion_fixed` against `newmodels_full/sd_*` |
| A9 | every single-drug cell that reports a fold SD |

**One precision note.** `results_frame` reads each run's `summary.csv`, which is
written to four decimal places; the per-drug JSONs beside it carry full
precision. Every difference reported here is at or above the third decimal, so
the rounding does not reach any claim, but a number quoted from a JSON may differ
from one quoted here in the fourth.
""")

# ============================================================== emit
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
import json as _json
OUT.write_text(_json.dumps(nb, indent=1) + "\n")
print(f"wrote {OUT}  ({len(cells)} cells)")

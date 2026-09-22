"""Generate notebooks/lasso.ipynb — the three lasso runs, in detail, per drug.

    python scripts/build_lasso.py
    python scripts/build_lasso.py -o /tmp/preview.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/lasso.ipynb

`results.ipynb` §13–§14 carries the two lasso findings that bear on the project's
argument, at macro resolution, and deliberately leaves the windows run out. This
notebook is the other half: **every arm of all three runs, every drug, both
metrics**, with the cost and sparsity columns that say what each number was
bought with.

The three runs, and what each one changes against the one before it:

  lasso_wholeseq_20260901    the aligned loci -- the SAME one-hot input the
                             convolutional models read. 2 encodings x 2 locus
                             universes x 11 drugs, plus an ETHIONAMIDE locus
                             ablation.
  lasso_wholegenome_20260903 every variant in the 4.41 Mb genome, delta only.
                             Plus where the coefficients land against the WHO
                             catalogue.
  lasso_windows_20260903     200 bp windows around the genome model's own top
                             50 coefficients. 3 arms, and its CV column is
                             selection-optimistic by construction.

Baselines on every per-drug table, in the order they are quoted:

  SD-CNN / MD-CNN     the two published CNNs (leak-corrected SD-CNN)
  best overall        the best complete cell in the project, any family
  best DNA-only       the best complete cell reading DNA and nothing else

The last two are derived from `results_frame`, never hard-coded, so they follow
the grid rather than a number typed here.

FIGURES. Same house style as `scripts/build_results.py`: light surface, dark ink,
colour used for identity only, no text on a saturated fill, and **a title names
the subject and the slice, never the finding** (see ~/.claude/CLAUDE.md). Where a
chart shows one model at different settings, colour is a single-hue ordinal ramp
inside that family rather than unrelated hues -- AQUA_RAMP, which passes the
lightness-monotonicity, step-gap and light-end-contrast checks against this
surface. The categorical trio (AQUA / BLUE / RUST) passes adjacent-pair CVD
separation; AQUA's 2.74:1 fill contrast sits under 3:1, so every chart using it
carries direct value labels or a table beside it.
"""
import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("-o", "--output", type=Path, default=None,
                 help="notebook path (default: <project>/notebooks/lasso.ipynb)")
_args = _ap.parse_args()
OUT = (_args.output.expanduser().resolve() if _args.output
       else PROJECT / "notebooks" / "lasso.ipynb")

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": src.rstrip("\n").splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


# ============================================================== title
md("""
# The lasso runs, in detail

L1-logistic regression on *M. tuberculosis* sequence, 11 drugs, against the same
5-fold protocol as every network in this project. Three runs, nine arms, and the
per-drug numbers behind each one.

| run | input | arms |
|---|---|---|
| `lasso_wholeseq_20260901` | the **aligned curated loci** — the same one-hot the convolutional models read | `onehot` / `delta` × per-drug loci / all 19, plus an ETHIONAMIDE locus ablation |
| `lasso_wholegenome_20260903` | every variant called in the **4.41 Mb genome** | `delta` only — a wild-type indicator per site is dense at ~10⁶ sites |
| `lasso_windows_20260903` | **200 bp windows** around the genome model's own top 50 coefficients | the 50 keys alone / every key in their windows / those keys plus a wild-type indicator per site |

**Why a lasso at all.** It is the control the networks should be quoted against.
SHAP on the CNNs said the signal is sparse; the aggregator family said it is
mostly additive; a lasso is "sparse and additive" written down as a model. It is
also *readable* — a coefficient names a locus and a column — which is how the
co-resistance and lineage results below became visible at all.

**How every number is judged.** 5-fold CV AUC, macro over the 11 drugs shared
with the baselines. Held-out test AUC is one 20% split at seed 42, scored once
with the best CV fold's model. Single seed throughout, mean per-drug fold SD
≈0.016, so **differences under ~0.01 are not resolved**. Two exceptions to
"read CV": the windows run, whose CV is selection-optimistic by construction
(§4), and any comparison against the published baselines, which are CV.

> Generated — edit `scripts/build_lasso.py`, not the `.ipynb`.
> `python scripts/build_lasso.py && jupyter nbconvert --to notebook --execute --inplace notebooks/lasso.ipynb`
>
> `results.ipynb` §13–§14 is the macro-level version of this, in the project's
> argument. `wholegenome_lasso.ipynb` is the genome run on its own.
""")

# ============================================================== setup
code('''
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

_here = Path.cwd()
PROJECT = next(p for p in [_here, *_here.parents] if (p / "bigtb_ref.py").exists())
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
EXP = PROJECT / "results" / "experiments"

from results_frame import load_frame
from bigtb_baselines import SD_BASE_CV, MD_BASE_CV, ALL_DRUGS

# ---------------------------------------------------------------- palette
# Identical to scripts/build_results.py -- the two notebooks sit beside each
# other and a second look would read as a second project.
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; AXIS = "#c3c2b7"; PALE = "#f0efec"
BLUE = "#2a78d6"     # the established convolutional architectures
RUST = "#a8420f"     # locusfusion / the variant-token family
AQUA = "#1baf7a"     # L1-logistic -- this notebook's subject
# One model at different settings is not several identities, so it gets a
# single-hue ordinal ramp rather than unrelated hues. Validated light->dark:
# monotone L, adjacent dL >= 0.06, light end 2.24:1 against SURFACE.
AQUA_RAMP = ["#0d5c43", "#1e9370", "#63bb9d"]
# Table fills: text is INK on every one, so both ramps stop well short of
# saturated. Sequential for magnitude, diverging (two hues, neutral midpoint)
# for signed differences.
CM_SEQ = LinearSegmentedColormap.from_list("seq", [SURFACE, "#b7e3d1"])
CM_DIV = LinearSegmentedColormap.from_list("div", ["#f3c9b6", SURFACE, "#b7e3d1"])

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
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlepad": 10,
})


def tidy(ax, xgrid=True, ygrid=False):
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=xgrid)
    ax.grid(axis="y", visible=ygrid)
    return ax


# Direct labels with hairline leaders, placed so none overlaps another -- the
# same helper build_results.py uses. Boxes are estimated from the text rather
# than measured, which keeps it usable before the figure is drawn; points are
# labelled in x order and every placed box is an obstacle for the next, so the
# result is deterministic across rebuilds.
CANDIDATES = [(0, 15), (0, -25), (0, 31), (0, -41), (52, 5), (-52, 5),
              (52, -13), (-52, -13), (0, 47), (0, -57)]


def label_points(ax, xs, ys, texts, fontsize=7.0, pad=3.0):
    xs, ys, texts = list(xs), list(ys), list(texts)
    trans = ax.transData
    placed = []
    for x, y in zip(xs, ys):                       # markers are obstacles too
        px, py = trans.transform((x, y))
        placed.append((px - 7, py - 7, px + 7, py + 7))

    def overlaps(box):
        l, b, r, t = box
        return any(not (r < L or l > R or t < B or b > T) for L, B, R, T in placed)

    for i in np.argsort(np.asarray(xs, dtype=float)):
        w = len(texts[i]) * fontsize * 0.60
        h = fontsize * 1.45
        px, py = trans.transform((xs[i], ys[i]))
        for dx, dy in CANDIDATES:
            cy = py + dy + (h / 2 if dy >= 0 else -h / 2)
            cx = px + dx + (0 if dx == 0 else (w / 2 if dx > 0 else -w / 2))
            box = (cx - w / 2 - pad, cy - h / 2 - pad,
                   cx + w / 2 + pad, cy + h / 2 + pad)
            if not overlaps(box):
                break
        placed.append(box)
        ax.annotate(texts[i], (xs[i], ys[i]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=fontsize, color=INK2,
                    ha="center" if dx == 0 else ("right" if dx < 0 else "left"),
                    va="bottom" if dy >= 0 else "top", zorder=5,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6,
                                    shrinkA=0, shrinkB=5))


TSTYLE = [
    {"selector": "caption", "props": f"caption-side:top; font-size:11px; "
     f"color:{INK2}; text-align:left; padding-bottom:6px"},
    {"selector": "th", "props": f"background-color:{PALE}; color:{INK}; "
     "font-weight:600; font-size:11px; text-align:left; padding:3px 8px"},
    {"selector": "td", "props": f"color:{INK}; font-size:11px; padding:3px 8px"},
]


def show(styler, caption):
    display(styler.set_caption(caption).set_table_styles(TSTYLE))


# ---------------------------------------------------------------- the arms
# (run folder, cell folder, row filter, label, encoding, universe). The windows
# run puts three arms in ONE folder tagged by `modalities`, so the filter is
# part of the arm's identity rather than the folder.
ARMS = [
    ("lasso_wholeseq_20260901", "delta_perdrug",  None, "delta · per-drug loci", "delta", "2-3 loci"),
    ("lasso_wholeseq_20260901", "onehot_perdrug", None, "onehot · per-drug loci", "onehot", "2-3 loci"),
    ("lasso_wholeseq_20260901", "delta_all",      None, "delta · 19 loci", "delta", "19 loci"),
    ("lasso_wholeseq_20260901", "onehot_all",     None, "onehot · 19 loci", "onehot", "19 loci"),
    ("lasso_wholegenome_20260903", "all_drugs",   None, "delta · whole genome", "delta", "4.41 Mb"),
    ("lasso_windows_20260903", "all_drugs", "delta-top50",   "delta · top-50 keys", "delta", "50 keys"),
    ("lasso_windows_20260903", "all_drugs", "delta-window",  "delta · 200bp windows", "delta", "windows"),
    ("lasso_windows_20260903", "all_drugs", "onehot-window", "onehot · 200bp windows", "onehot", "windows"),
]
RUN_LABEL = {"lasso_wholeseq_20260901": "aligned loci",
             "lasso_wholegenome_20260903": "whole genome",
             "lasso_windows_20260903": "windows"}
ARM_ORDER = [a[3] for a in ARMS]


def _load_arms():
    """One tidy row per (arm, drug), straight off each run's summary.csv."""
    rows = []
    for run, cell, tag, label, enc, uni in ARMS:
        f = EXP / run / cell / "summary.csv"
        if not f.is_file():
            print(f"missing: {f}")
            continue
        s = pd.read_csv(f)
        if tag is not None:
            s = s[s["modalities"] == tag]
        for _, r in s.iterrows():
            if r["drug"] not in set(ALL_DRUGS):
                continue
            rows.append(dict(run=RUN_LABEL[run], arm=label, encoding=enc,
                             universe=uni, drug=r["drug"],
                             n=r["n_valid"], n_R=r["n_R"],
                             cv=r["cv_auc_mean"], cv_sd=r["cv_auc_std"],
                             test=r["test_auc"], test_pr=r["test_auc_pr"],
                             sens=r["test_sens"], spec=r["test_spec"],
                             C=r["best_C"], cols=r["n_features"],
                             support=r["cv_support_mean"], secs=r["seconds"]))
    return pd.DataFrame(rows)


L = _load_arms()

# ---------------------------------------------------------------- baselines
# Two are published; two are this project's own best cells, taken from the grid
# rather than typed in, so they track whatever has been run.
frame, meta = load_frame()
AGGREGATORS = {"catalogue", "additive", "noisyor", "gatedpool", "deepsets", "fm"}
NOT_A_NETWORK = {"sparse_baseline", "sparse_baseline_trainvocab",
                 "lasso_onehot", "lasso_delta", "lasso_genome"}
nets = frame[~frame.model.isin(NOT_A_NETWORK)]


def _pick(df, label):
    """Best COMPLETE (11/11) cell in a slice -> (label, per-drug cv, per-drug test)."""
    g = (df.groupby(["task", "model", "modset"])
         .agg(cv=("cv", "mean"), n=("drug", "nunique"))
         .query("n == 11").sort_values("cv", ascending=False))
    if g.empty:
        return None
    task, model, modset = g.index[0]
    s = df[(df.task == task) & (df.model == model) & (df.modset == modset)]
    name = f"{label} ({model}, {modset}, {int(s.n_loci.max())} loci)"
    # A cell can exist in more than one run folder (locusfusion is in both
    # `newmodels_full` and `locusfusion_fixed`), so average over runs rather than
    # indexing by drug -- that is also how `build_results.best` aggregates.
    return name, s.groupby("drug").cv.mean(), s.groupby("drug").test.mean()


_best_all = _pick(nets, "best overall")
_best_dna = _pick(nets[nets.modset == "dna"], "best DNA-only")

BASE_CV = {"SD-CNN": SD_BASE_CV.reindex(ALL_DRUGS),
           "MD-CNN": MD_BASE_CV.reindex(ALL_DRUGS),
           _best_all[0]: _best_all[1].reindex(ALL_DRUGS),
           _best_dna[0]: _best_dna[1].reindex(ALL_DRUGS)}
BASE_TEST = {_best_all[0]: _best_all[2].reindex(ALL_DRUGS),
             _best_dna[0]: _best_dna[2].reindex(ALL_DRUGS)}
BASE_SHORT = {k: k.split(" (")[0] for k in BASE_CV}

# Drug order: cohort size, smallest first. Half the results in this notebook are
# a function of n, so sorting by it stops the reader from having to hold the
# eleven cohort sizes in their head.
NVAL = L.groupby("drug").n.max().sort_values()
DRUGS = list(NVAL.index)


def per_drug(metric="cv", arms=None, baselines=True):
    """drug x arm table of one metric, with the baselines appended as columns."""
    s = L[L.arm.isin(arms)] if arms else L
    t = s.pivot_table(index="drug", columns="arm", values=metric)
    t = t.reindex(index=DRUGS, columns=[a for a in ARM_ORDER if a in t.columns])
    if baselines:
        src = BASE_CV if metric == "cv" else BASE_TEST
        for name, v in src.items():
            t[BASE_SHORT[name]] = v.reindex(DRUGS)
    t.insert(0, "n", NVAL.reindex(DRUGS).astype(int))
    return t


def macro(t):
    """Column means over the 11 drugs, as a MACRO row."""
    return t.drop(columns=["n"]).mean()


print(f"{len(L)} (arm, drug) rows over {L.arm.nunique()} arms and {L.run.nunique()} runs")
for k, v in BASE_CV.items():
    print(f"baseline  {k:<52s} macro CV {v.mean():.4f}")
''')

md("""
The two derived baselines are printed above with the cell they came from, so
they can be checked against §14 of `results.ipynb` — they are read off the grid,
not typed in.

Note what "best DNA-only" turns out to be: the **same architecture** as "best
overall", 0.002 below it. `locusfusion` barely moves across modality sets, which
is that model's own finding rather than a coincidence here. If you want the best
DNA-only *convolutional* number instead, it is `mdcnn` at 19 loci, 0.9171, and it
appears in the ladder below.

---
## 1. The nine arms at a glance
""")

code('''
g = (L.groupby(["run", "arm"], sort=False)
     .agg(cv=("cv", "mean"), test=("test", "mean"),
          cols=("cols", "mean"), support=("support", "mean"),
          secs=("secs", "mean"), drugs=("drug", "nunique")))
g = g.reindex([(RUN_LABEL[r], lab) for r, _, _, lab, _, _ in ARMS])
g.index.names = ["run", "arm"]

show(g.style.format({"cv": "{:.4f}", "test": "{:.4f}", "cols": "{:,.0f}",
                     "support": "{:,.0f}", "secs": "{:,.0f}", "drugs": "{:.0f}"})
     .background_gradient(cmap=CM_SEQ, subset=["cv", "test"]),
     "every lasso arm, macro over 11 drugs -- cols/support/secs are per-drug means")

_b = pd.DataFrame({"macro CV": {k: v.mean() for k, v in BASE_CV.items()}})
_b["macro test"] = pd.Series({k: v.mean() for k, v in BASE_TEST.items()})
show(_b.style.format("{:.4f}", na_rep="— (published CV only)"),
     "the four baselines")
''')

code('''
# One bar per arm plus the baselines, on one axis. Value labels sit outside the
# bar in ink -- AQUA's fill contrast is under 3:1, so the number is what carries
# the reading, not the fill.
#
# The WINDOWS arms are deliberately absent. Their CV is computed after selecting
# features on the same training split, so it is not the same quantity as every
# other bar here -- on this axis they would sort to the TOP (0.9350-0.9370),
# above every model in the project, which is an artifact and not a result.
# Section 4 gives them their own tables on TEST, where they are comparable.
_mdcnn_dna = nets[(nets.model == "mdcnn") & (nets.modset == "dna")
                  & (nets.task == "single-drug, all loci")].cv.mean()
rows = [(f"{lab}", g.loc[(RUN_LABEL[run], lab), "cv"], AQUA)
        for run, _, _, lab, _, _ in ARMS if RUN_LABEL[run] != "windows"]
rows += [(BASE_SHORT[k], v.mean(), RUST if "best" in k else BLUE)
         for k, v in BASE_CV.items()]
rows += [("mdcnn, dna, 19 loci", _mdcnn_dna, BLUE)]
rows = sorted(rows, key=lambda r: r[1])

fig, ax = plt.subplots(figsize=(9.6, 6.0))
y = np.arange(len(rows))
ax.barh(y, [r[1] for r in rows], height=0.66, color=[r[2] for r in rows],
        edgecolor="none", zorder=2)
for yi, (_, v, _c) in zip(y, rows):
    ax.text(v + 0.002, yi, f"{v:.4f}", va="center", ha="left", fontsize=7.4, color=INK)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.set_xlim(0.85, 0.95); ax.set_ylim(-0.7, len(rows) - 0.2)
ax.set_xlabel("macro CV AUC")
ax.set_title("Macro CV AUC by arm and baseline, aligned-locus and "
             "whole-genome arms, 11 shared drugs", loc="left", fontsize=10.5)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (AQUA, BLUE, RUST)],
          ["L1-logistic", "convolutional baselines", "best cells in the project"],
          loc="lower right")
tidy(ax)
plt.tight_layout(); plt.show()
''')

md("""
Three things to carry into the detail below.

- **The `onehot` 19-locus arm is level with the project's best cells.** 0.9241
  against 0.9290 and 0.9271 — inside the fold SD, on 228 nonzero coefficients.
- **The windows arms are missing from the chart on purpose.** Their CV is
  selection-optimistic — the 50 seed keys were chosen by a model fit on the same
  training split — so it is not the quantity the other bars are. On this axis
  they would sort *above every model in the project* (0.9350–0.9370), which is
  an artifact of the selection and not a result. §4 measures how large that
  artifact is (up to +0.059, entirely at small n) and reads TEST instead. They
  are in the table above so the number is not hidden, and in §4.1–§4.3 per drug.
- **Cost does not track width.** The 426k-column genome arm fits in 58 s; the
  22k-column `onehot` alignment takes 1,405 s. Sparsity, not size, is what costs.

---
## 2. `lasso_wholeseq_20260901` — the aligned loci

The arm that matters most, because its input is *identical* to what the Part I
CNNs are handed: a one-hot over every column of every locus, cohort-constant
columns dropped. Two encodings — `onehot` (one indicator per (column, base),
wild type included) and `delta` (nonzero only where the isolate differs from
H37Rv) — crossed with two locus universes.

### 2.1 Per-drug CV AUC
""")

code('''
WS = [a[3] for a in ARMS if a[0] == "lasso_wholeseq_20260901"]
t = per_drug("cv", WS)
show(t.style.format({"n": "{:,.0f}"}, na_rep="—")
     .format("{:.4f}", subset=[c for c in t.columns if c != "n"], na_rep="—")
     .background_gradient(cmap=CM_SEQ, axis=None,
                          subset=[c for c in t.columns if c != "n"]),
     "CV AUC per drug, aligned-locus arms and the four baselines "
     "(drugs by cohort size, smallest first)")
print("MACRO", macro(t).round(4).to_dict())
''')

md("""### 2.2 Per-drug held-out test AUC""")

code('''
t = per_drug("test", WS)
show(t.style.format({"n": "{:,.0f}"}, na_rep="—")
     .format("{:.4f}", subset=[c for c in t.columns if c != "n"], na_rep="—")
     .background_gradient(cmap=CM_SEQ, axis=None,
                          subset=[c for c in t.columns if c != "n"]),
     "held-out test AUC per drug -- one 20% split at seed 42, scored with the "
     "best CV fold's model. SD-CNN/MD-CNN publish CV, so they are absent here.")
print("MACRO", macro(t).round(4).to_dict())
''')

md("""
### 2.3 The two effects, per drug

The macro hides which of the two axes is doing the work. Both are paired on the
drug, so a cell that exists on one side only would be dropped rather than
averaged in — here all 44 exist.
""")

code('''
_c = L.pivot_table(index="drug", columns="arm", values="cv").reindex(DRUGS)
enc = pd.DataFrame({
    "per-drug loci": _c["onehot · per-drug loci"] - _c["delta · per-drug loci"],
    "19 loci":       _c["onehot · 19 loci"] - _c["delta · 19 loci"]})
loci = pd.DataFrame({
    "onehot": _c["onehot · 19 loci"] - _c["onehot · per-drug loci"],
    "delta":  _c["delta · 19 loci"] - _c["delta · per-drug loci"]})

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
for ax, (tbl, title, xlab) in zip(axes, [
        (enc, "Encoding effect by drug", "CV AUC, onehot − delta"),
        (loci, "Locus-universe effect by drug", "CV AUC, 19 loci − per-drug loci")]):
    y = np.arange(len(tbl)); w = 0.36
    for i, col in enumerate(tbl.columns):
        ax.barh(y + (i - 0.5) * (w + 0.05), tbl[col], height=w, zorder=2,
                color=AQUA_RAMP[2 - 2 * i], edgecolor="none", label=col)
    ax.axvline(0, color=AXIS, lw=1.0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d}  n={int(NVAL[d]):,}" for d in tbl.index], fontsize=7.8)
    ax.set_xlabel(xlab); ax.set_title(title, loc="left")
    ax.legend(loc="lower right"); tidy(ax)
# ONCE, not per-axis: the two share a y-axis, so inverting both cancels out.
axes[0].invert_yaxis()         # smallest cohort at the top, as in every table
plt.tight_layout(); plt.show()

try:
    from scipy.stats import wilcoxon
    _p = lambda a, b: wilcoxon(a.dropna(), b.dropna()).pvalue
except ImportError:
    _p = lambda a, b: float("nan")
for name, tbl in (("encoding", enc), ("locus universe", loci)):
    for col in tbl.columns:
        v = tbl[col]
        print(f"{name:15s} {col:15s} mean {v.mean():+.4f}   "
              f"{(v > 0).sum()}/11 drugs positive")
print(f"\\nencoding, 19 loci     : Wilcoxon p = "
      f"{_p(_c['onehot · 19 loci'], _c['delta · 19 loci']):.4f}  "
      f"(2/2048 is the floor at n=11 -- it means 11/11, nothing finer)")
''')

md("""
**The encoding is worth about a point, on every drug, in both universes.**
`onehot` and `delta` span the same column space — `delta` is ordinary dummy
coding with H37Rv as the reference level — so they carry identical information.
L1 is not invariant to that reparametrisation: penalising "has base *b* here" is
not penalising "differs from H37Rv here". Keeping the wild-type level as its own
penalised column is worth +0.009 at 19 loci and +0.014 at the per-drug sets,
11 of 11 drugs both times.

This is a caution for every variant-token model in the project, all of which are
`--delta` by construction. It does not transfer automatically — those models are
not L1-penalised — but this is the one place the reference coding has been
measured directly.

**The locus universe is worth five times more, and it is a small-cohort effect.**
+0.049 (`onehot`) and +0.054 (`delta`) macro, but read the panel rather than the
macro: ETHIONAMIDE gains 0.148 and RIFAMPICIN gains 0.004. The drugs that gain
are the ones whose own 2–3 gene map is too small to learn from — and, per the
ablation below, part of what they gain is co-resistance rather than mechanism.

### 2.4 Where ETHIONAMIDE's +0.148 comes from

ETO takes the largest locus-universe gain of any drug. Two explanations compete:
its per-drug map (`ethA+ethR+inhA`) **omits `fabG1`**, which carries the dominant
ETO mechanism (the *fabG1–inhA* operon promoter), or the 19-locus model is
reading MDR co-occurrence off other drugs' genes. Adding one locus at a time
separates them.
""")

code('''
ab = pd.read_csv(EXP / "lasso_wholeseq_20260901" / "ablation_eto" / "summary.csv")
ab = ab.set_index("genes")[["cv_auc_mean", "test_auc", "n_features", "cv_support_mean"]]
ab.columns = ["CV AUC", "test AUC", "columns", "nonzero coefs"]
# The two anchors live in the parent arms, not in this folder.
_eto = L[(L.drug == "ETHIONAMIDE") & (L.arm.isin(
    ["onehot · per-drug loci", "onehot · 19 loci"]))].set_index("arm")
for key, arm in (("ethA+ethR+inhA  (the per-drug map)", "onehot · per-drug loci"),
                 ("all 19 curated loci", "onehot · 19 loci")):
    r = _eto.loc[arm]
    ab.loc[key] = [r.cv, r.test, r.cols, r.support]
ab = ab.reindex([
    "ethA+ethR+inhA  (the per-drug map)", "ethA+ethR+inhA+fabG1",
    "ethA+ethR+inhA+rpoB", "ethA+ethR+inhA+pncA", "fabG1",
    "rpoB+pncA+katG+embB", "all 19 curated loci"])
show(ab.style.format({"CV AUC": "{:.4f}", "test AUC": "{:.4f}",
                      "columns": "{:,.0f}", "nonzero coefs": "{:,.0f}"})
     .background_gradient(cmap=CM_SEQ, subset=["CV AUC"]),
     "ETHIONAMIDE only, onehot, one locus added at a time. Fold spread across "
     "these cells is 0.010-0.016, so gaps under ~0.012 are not resolved.")
''')

md("""
**Both explanations are true and they are the same size.** Adding `fabG1` — WHO
tier 1 for ETO — buys +0.097. Adding `rpoB`, which *cannot* cause ethionamide
resistance, buys +0.085. The difference between them is 0.012, at the edge of
what these folds resolve.

Two rows sharpen it:

- **`fabG1` alone, 26 columns and 11 nonzero coefficients, reaches 0.7262** —
  beating the drug's entire official 3-gene map (0.6641) by +0.062. The per-drug
  locus map is worse than one locus it leaves out.
- **`rpoB+pncA+katG+embB`, with zero ETO mechanism, reaches 0.7241** — +0.060
  over that same map. Co-resistance alone predicts ethionamide about as well as
  ethionamide's own genes do.

So "more loci help" is established, and "more loci carry more mechanism" is not.
Both claims were being made by the same +0.049.

---
## 3. `lasso_wholegenome_20260903` — the whole genome

Every `(position, ref, alt)` key called anywhere in the drug's cohort: 28k
columns for LEVOFLOXACIN's 269 isolates, 899k for RIFAMPICIN's 17,582. `delta`
only — a wild-type indicator per site is dense at ~10⁶ sites — so its matched
control is the 19-locus **`delta`** arm, not the `onehot` headline.

The 19 curated loci are 0.9% of the genome and every one of them is a resistance
gene for some drug. That makes two questions unanswerable from inside them:
whether the curation leaves signal outside, and — since everything in view is a
resistance gene — whether coefficients landing on resistance genes means
anything. Genome-wide, both become measurable.

### 3.1 Per-drug, against its matched control
""")

code('''
cols = ["delta · 19 loci", "delta · whole genome"]
cv = per_drug("cv", cols); te = per_drug("test", cols)
t = pd.DataFrame({"n": cv["n"],
                  "CV, 19 loci": cv[cols[0]], "CV, genome": cv[cols[1]],
                  "CV Δ": cv[cols[1]] - cv[cols[0]],
                  "test, 19 loci": te[cols[0]], "test, genome": te[cols[1]],
                  "test Δ": te[cols[1]] - te[cols[0]]})
for k in BASE_CV:
    t[BASE_SHORT[k] + " CV"] = BASE_CV[k].reindex(DRUGS)
_num = [c for c in t.columns if c != "n"]
show(t.style.format({"n": "{:,.0f}"})
     .format("{:.4f}", subset=_num, na_rep="—")
     .background_gradient(cmap=CM_DIV, subset=["CV Δ", "test Δ"], vmin=-0.04, vmax=0.04)
     .background_gradient(cmap=CM_SEQ, axis=None,
                          subset=[c for c in _num if "Δ" not in c]),
     "whole genome against the matched 19-locus delta arm, per drug")
print("macro CV  19 loci %.4f   genome %.4f   delta %+.4f  (%d/11 wins)"
      % (cv[cols[0]].mean(), cv[cols[1]].mean(),
         (cv[cols[1]] - cv[cols[0]]).mean(), (cv[cols[1]] > cv[cols[0]]).sum()))
print("macro test 19 loci %.4f   genome %.4f   delta %+.4f  (%d/11 wins)"
      % (te[cols[0]].mean(), te[cols[1]].mean(),
         (te[cols[1]] - te[cols[0]]).mean(), (te[cols[1]] > te[cols[0]]).sum()))
try:
    from scipy.stats import wilcoxon
    print("Wilcoxon on CV: p = %.3f" % wilcoxon(cv[cols[1]], cv[cols[0]]).pvalue)
except ImportError:
    pass
''')

md("""
**The other 99.1% of the genome is worth +0.0022, 6 wins of 11, p = 0.78.** 110×
the sequence and 27× the columns buy nothing measurable, and the `onehot` arm on
0.9% of the genome is 0.007 *ahead*. The returns to more sequence ended well
before the genome did: 2–3 loci → 19 was +0.05, 19 loci → 4.41 Mb is +0.002.

The test column is friendlier — +0.0087, 8 of 11 — but it is one 20% split per
drug, swinging ±0.05 on the small cohorts, against five estimates behind each CV
number. Where the two disagree at this size, CV is the one to quote.

Read the Δ column rather than the macro, though: this is a **cancellation**, not
per-drug equality. AMIKACIN +0.040 and MOXIFLOXACIN +0.019 against LEVOFLOXACIN
−0.030, with the other eight inside ±0.010. Nothing here says *which* extra
sequence helped AMIKACIN, and with one seed the honest reading of two positive
outliers in eleven is that they are outliers.

### 3.2 Where the weight lands

The question the curated input could not ask. Every nonzero coefficient of the
selected model is matched against **WHO 2023 Catalogue Table 21** (tier 1 + tier
2 candidate resistance genes) by gene name *and* Rv tag, and the |coefficient|
mass split three ways. An intergenic region flanking a Table-21 gene counts as
inside — that is where the Table-22 promoter windows live, e.g. the
*fabG1–inhA* operon promoter is annotated `Rv1482c-fabG1`.
""")

code('''
_f = EXP / "lasso_wholegenome_20260903" / "who_mass.csv"
if not _f.is_file():
    print("who_mass.csv not built -- "
          "python results/experiments/lasso_wholegenome_20260903/who_mass.py "
          "--run .../all_drugs")
else:
    w = pd.read_csv(_f).set_index("drug").reindex(DRUGS)
    w = w.rename(columns={"this drug %": "this drug", "other drug %": "other drug",
                          "outside %": "outside"})
    mass = w[["this drug", "other drug", "outside"]]

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    y = np.arange(len(mass)); left = np.zeros(len(mass))
    # Ordered buckets (most specific -> least), so a single-hue ordinal ramp,
    # not three unrelated hues. A 2px surface gap separates the segments.
    for col, colour in zip(mass.columns, AQUA_RAMP):
        ax.barh(y, mass[col], left=left, height=0.66, color=colour, zorder=2,
                edgecolor=SURFACE, linewidth=1.2, label=col)
        left += mass[col].values
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d}  {int(w.support[d])} coefs" for d in mass.index],
                       fontsize=7.8)
    ax.set_xlim(0, 100); ax.set_xlabel("share of |coefficient| mass (%)")
    ax.invert_yaxis()          # smallest cohort at the top, as in every table
    ax.set_title("Coefficient mass by WHO Table-21 membership, per drug", loc="left")
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.26))
    tidy(ax)
    plt.tight_layout(); plt.show()

    tab = mass.copy()
    tab["nonzero coefs"] = w["support"]
    tab["columns"] = w["cols"]
    tab.loc["MACRO"] = [mass["this drug"].mean(), mass["other drug"].mean(),
                        mass["outside"].mean(), w["support"].mean(), w["cols"].mean()]
    show(tab.style.format({"this drug": "{:.1f}%", "other drug": "{:.1f}%",
                           "outside": "{:.1f}%", "nonzero coefs": "{:,.0f}",
                           "columns": "{:,.0f}"})
         .background_gradient(cmap=CM_SEQ, subset=["outside"]),
         "|coefficient| mass by bucket, and how many coefficients it is spread over")
    _lead = mass.idxmax(axis=1)
    print("outside is the largest bucket in %d/11 drugs, a majority in %d/11"
          % ((_lead == "outside").sum(), (mass["outside"] > 50).sum()))
    print("exceptions:", ", ".join(_lead[_lead != "outside"].index))
''')

md("""
**62.4% of the weight sits outside the WHO catalogue**, 27.1% on the drug's own
Table-21 genes, 10.6% on another drug's. Outside is the largest bucket for 9 of
11 drugs and an outright majority for 8. The exceptions are LEVOFLOXACIN, which
has 22 nonzero coefficients in total, and STREPTOMYCIN at 43.2% vs 41.8%.

The model is not ignoring the biology. For ETHIONAMIDE the single largest
coefficient out of 127,697 is `1673425:C>T` — the *fabG1–inhA* operon promoter,
the canonical *c-15t* mechanism — found first, with no curation, no locus list
and no gene annotation anywhere in the objective. But it is a minority of the
weight; the rest lands heavily on the *PE_PGRS* / *PPE* / *esx* repeat families,
which are the standard lineage markers of *M. tuberculosis*.

PYRAZINAMIDE is the sharpest case: **18.6% on other drugs' resistance genes
against 5.9% on its own**. That is MDR co-occurrence measured genome-wide, and it
is the same effect §2.4 isolated one locus at a time.

**What this does not license.** A mass split is not a variance decomposition —
62% of the weight is not 62% of the AUC, and nothing here measures what survives
deleting it. Two readings stay open: population structure (lineage correlates
with resistance in this cohort) or short-read calling artifacts (PE/PPE/esx is
exactly where variant calling is least reliable). Separating them needs a
lineage-held-out split or a masked arm, and neither exists in this project.

---
## 4. `lasso_windows_20260903` — 200 bp windows

Take the genome model's 50 largest coefficients, open a 200 bp window around
each, refit on those windows alone. The encoded footprint is 8,111 bp on average
— **0.18% of the genome** — against the parent's 426,457 columns.

Three arms, because reading one alone cannot separate a context effect from an
encoding effect:

| arm | columns | what it isolates |
|---|---|---|
| `delta · top-50 keys` | the ≤50 selected keys | what the selection alone is worth |
| `delta · 200bp windows` | every cohort key in the windows | + local context, encoding fixed |
| `onehot · 200bp windows` | those keys + a wild-type indicator per site | + the encoding, columns fixed |

**Read TEST, not CV.** The 50 keys came from the parent's best-fold model, fitted
inside the same 80% training split at seed 42, so the held-out 20% took no part
in choosing the windows — but the CV folds did. §4.3 measures the size of that.

### 4.1 Per-drug held-out test AUC
""")

code('''
WIN = [a[3] for a in ARMS if a[0] == "lasso_windows_20260903"]
an = pd.read_csv(EXP / "lasso_windows_20260903" / "analysis.csv")
parent = an.groupby("drug")[["parent_cv", "parent_test", "parent_cols"]].first()

te = per_drug("test", WIN, baselines=False)
te.insert(1, "parent (whole genome)", parent.parent_test.reindex(DRUGS))
for k, v in BASE_TEST.items():
    te[BASE_SHORT[k]] = v.reindex(DRUGS)
_num = [c for c in te.columns if c != "n"]
show(te.style.format({"n": "{:,.0f}"}).format("{:.4f}", subset=_num, na_rep="—")
     .background_gradient(cmap=CM_SEQ, axis=None, subset=_num),
     "held-out test AUC per drug -- the clean column for this run")
print("MACRO", te[_num].mean().round(4).to_dict())
''')

md("""### 4.2 Per-drug CV AUC — selection-optimistic, kept for §4.3""")

code('''
cv = per_drug("cv", WIN, baselines=False)
cv.insert(1, "parent (whole genome)", parent.parent_cv.reindex(DRUGS))
for k, v in BASE_CV.items():
    cv[BASE_SHORT[k]] = v.reindex(DRUGS)
_num = [c for c in cv.columns if c != "n"]
show(cv.style.format({"n": "{:,.0f}"}).format("{:.4f}", subset=_num, na_rep="—")
     .background_gradient(cmap=CM_SEQ, axis=None, subset=_num),
     "CV AUC per drug -- inflated by feature selection on the training split; "
     "the baselines in this table are NOT")
print("MACRO", cv[_num].mean().round(4).to_dict())
''')

md("""### 4.3 The three contrasts, paired over 11 drugs""")

code('''
_t = L[L.run == "windows"].pivot_table(index="drug", columns="arm", values="test")
_v = L[L.run == "windows"].pivot_table(index="drug", columns="arm", values="cv")
P = parent.parent_test
CON = [("onehot windows vs parent", _t["onehot · 200bp windows"] - P),
       ("delta windows vs parent",  _t["delta · 200bp windows"] - P),
       ("top-50 keys vs parent",    _t["delta · top-50 keys"] - P),
       ("context (windows − top-50)",
        _t["delta · 200bp windows"] - _t["delta · top-50 keys"]),
       ("encoding (onehot − delta, windows)",
        _t["onehot · 200bp windows"] - _t["delta · 200bp windows"])]
rows = []
for name, d in CON:
    d = d.reindex(DRUGS).dropna()
    try:
        from scipy.stats import wilcoxon
        p = wilcoxon(d).pvalue
    except Exception:
        p = float("nan")
    rows.append({"contrast": name, "mean": d.mean(), "median": d.median(),
                 "wins": f"{(d > 0).sum()}/11", "Wilcoxon p": p,
                 "mean w/o LFX": d.drop("LEVOFLOXACIN", errors="ignore").mean()})
con = pd.DataFrame(rows).set_index("contrast")
show(con.style.format({"mean": "{:+.4f}", "median": "{:+.4f}",
                       "Wilcoxon p": "{:.3f}", "mean w/o LFX": "{:+.4f}"})
     .background_gradient(cmap=CM_DIV, subset=["mean"], vmin=-0.02, vmax=0.02),
     "paired contrasts on held-out TEST AUC, 11 drugs")

# CV inflation against cohort size -- the reason CV is quoted separately.
infl = (_v["delta · top-50 keys"] - parent.parent_cv).reindex(DRUGS)
fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.scatter(NVAL.reindex(DRUGS), infl, s=70, color=AQUA, zorder=3,
           edgecolor=SURFACE, linewidth=1.5)
ax.axhline(0, color=AXIS, lw=1.0, zorder=2)
ax.set_xscale("log"); ax.set_xlabel("labelled isolates (log scale)")
ax.set_xlim(180, 30000)
ax.set_ylim(infl.min() - 0.012, infl.max() + 0.012)
plt.draw()                     # labels are placed in display space
label_points(ax, NVAL.reindex(DRUGS).values, infl.values, list(DRUGS))
ax.set_ylabel("CV AUC, top-50 refit − parent")
ax.set_title("CV inflation from selecting features on the training split, by cohort size",
             loc="left")
try:
    from scipy.stats import spearmanr
    rho, pv = spearmanr(NVAL.reindex(DRUGS), infl)
    ax.annotate(f"Spearman rho = {rho:.2f}, p = {pv:.4f}", (0.97, 0.94),
                xycoords="axes fraction", ha="right", fontsize=8, color=INK2)
except ImportError:
    pass
tidy(ax, xgrid=False, ygrid=True)
plt.tight_layout(); plt.show()

_big = NVAL.reindex(DRUGS) >= 7000
print("CV inflation: n>=7000 %+.4f   n<7000 %+.4f"
      % (infl[_big].mean(), infl[~_big].mean()))
print("TEST change  : n>=7000 %+.4f   n<7000 %+.4f"
      % ((_t["delta · top-50 keys"] - P).reindex(DRUGS)[_big].mean(),
         (_t["delta · top-50 keys"] - P).reindex(DRUGS)[~_big].mean()))
''')

md("""
**1. 0.18% of the genome matches the whole genome.** `onehot · windows` is
+0.0042 macro over the parent, but the median is −0.0011, it wins 5 of 11,
p = 0.90, and dropping LEVOFLOXACIN takes the mean to −0.0001. The two are
indistinguishable, and `delta · windows` reaches that in **2.6 s per drug**
against the parent's ~60 s. The useful result is the negative one: nothing
outside those 19–49 windows is doing measurable work.

**2. The top 50 coefficients are locations, not the signal.** Restricted to
exactly the 50 keys the parent chose, the model *loses* 0.0100 test AUC, winning
1 of 11 (p = 0.002). Restoring their 200 bp neighbourhoods recovers it and more:
**+0.0134, 10 of 11, p = 0.002** — the largest and most consistent effect in the
run. The coefficient mass says the same: in `delta · windows` it splits 49.9% on
the seed keys and 50.1% on neighbours the parent never selected, and in six of
eleven drugs the neighbours take the majority — CAPREOMYCIN 81%, PYRAZINAMIDE
79%, ETHAMBUTOL 71%, ISONIAZID 69%, ETHIONAMIDE 65%, and RIFAMPICIN at 50.1%,
which is an exact split rather than a sixth case.

So the parent's support is not a set of uniquely informative sites. L1 picked one
column from each of ~50 correlated local blocks, as L1 does. **A top-k
coefficient list from this model is a list of neighbourhoods, and reading it as a
list of sites loses half the signal** — which applies directly to §3.2's WHO
measurement, computed over exactly that support.

**3. Selecting on training data inflates CV by up to +0.059, entirely at small n.**
The scatter is nearly rank-ordered by cohort size. Same 50 columns, CV rises with
the selection while test falls: at n ≥ 7000 the CV effect is −0.0015 and at
n < 7000 it is **+0.0341**, running to +0.0593 at LEVOFLOXACIN. This is a
measurement of a pipeline that is easy to run by accident — select top-k, refit,
quote cross-validation — and at n < 3000 it reports up to six points of AUC that
do not exist. Honest CV needs re-selection inside every fold, i.e. a genome-wide
refit per fold; this run did not do that, and the gap between its two columns is
the reason to.

**4. The one-hot advantage does not survive outside the curated loci.** §2.3
measured `onehot` at +0.009 over `delta` on the 19 loci. Here, with columns held
fixed, it is **+0.0008, 7 of 11, p = 0.50** — zero. It is also not free: the
wild-type block takes the mean fit from 2.6 s to 250 s, 96× for nothing. The
indicators are not idle — they absorb 30.5% of total coefficient mass — the
reparametrisation just leaves the fitted function unchanged.

---
## 5. Every arm, every drug

The two tables the rest of the notebook is slices of.
""")

code('''
for metric, label in (("cv", "CV AUC"), ("test", "held-out test AUC")):
    t = per_drug(metric)
    t.loc["MACRO"] = macro(t)
    t.loc["MACRO", "n"] = np.nan
    _num = [c for c in t.columns if c != "n"]
    show(t.style.format({"n": "{:,.0f}"}, na_rep="")
         .format("{:.4f}", subset=_num, na_rep="—")
         .background_gradient(cmap=CM_SEQ, axis=None,
                              subset=pd.IndexSlice[DRUGS, _num]),
         f"{label} -- every lasso arm and every baseline, 11 drugs. "
         "Windows-arm CV is selection-optimistic (see 4.3).")
''')

code('''
# What each arm cost and how sparse it came out -- the columns that stop a
# per-drug AUC from being read as free.
c = (L.pivot_table(index="drug", columns="arm", values="cols")
     .reindex(index=DRUGS, columns=ARM_ORDER))
s = (L.pivot_table(index="drug", columns="arm", values="support")
     .reindex(index=DRUGS, columns=ARM_ORDER))
both = pd.concat({"columns": c, "nonzero coefficients": s}, axis=1)
both.loc["MACRO"] = both.mean()
show(both.style.format("{:,.0f}", na_rep="—"),
     "design-matrix width and fitted support, per drug and arm")
''')

md("""
---
## How to argue with these numbers

- **Single seed everywhere.** Mean per-drug fold SD ≈0.016, running 0.002
  (RIFAMPICIN) to 0.057 (LEVOFLOXACIN). Differences under ~0.01 are not
  resolved, which includes the whole-genome result and the lasso-vs-network
  parity in §1.
- **LEVOFLOXACIN is 269 isolates** — a 54-isolate test set — and it drives every
  positive macro mean in the windows run. Contrasts there are quoted with and
  without it; treat any per-drug LFX number as unresolved.
- **Wilcoxon p-values here are consistency statistics, not significance gates.**
  At n = 11 the smallest reachable two-sided p is 2/2048 = 0.001, which is
  exactly "11 of 11 in the same direction" and carries no effect size. A high p
  is not evidence of equivalence — the test has almost no power at eleven points.
  The eleven drugs are also not independent (shared isolates, MDR co-occurrence),
  which makes these p-values anti-conservative.
- **CV in the windows arms is selection-optimistic** and only TEST is clean.
  §4.3 is the size of the problem, not a correction for it.
- **CV AUC in this family reproduces only to ~4e-4.** `lasso_wholeseq._fit`,
  which every run here imports, never passes `random_state`, so liblinear's
  coordinate-descent permutation comes from the global NumPy state. Test metrics
  and `best_C` reproduce exactly; CV moves in the fourth decimal. That is also
  the floor for what counts as an effect — §4's +0.0008 encoding gain is about
  twice the solver's own jitter, which is the precise sense in which it is zero.
- **Lineage is not controlled anywhere.** §3.2 is the strongest statement of the
  problem, and it applies to the networks too — they are handed 19 loci that are
  themselves lineage-correlated. A lineage-held-out split is the highest-value
  follow-up in this notebook.
- **The baselines are not all the same kind of number.** SD-CNN and MD-CNN are
  published CV. "Best overall" and "best DNA-only" are this project's own best
  *complete* cells, selected on the same CV they report, so they are an upper
  bound rather than a model anyone could have chosen in advance.
- **DNA only, in every lasso arm.** Protein and biophysical features are
  deterministic functions of the DNA at the same positions and would be nearly
  collinear with the one-hot columns, so for a linear model they mostly buy
  penalty. That makes "best DNA-only" the fairer of the two derived baselines.
- **No ridge arm**, so "sparsity is doing the work" is not established here —
  only that a sparse model suffices.
""")

# ================================================================== write
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(cells)} cells)")

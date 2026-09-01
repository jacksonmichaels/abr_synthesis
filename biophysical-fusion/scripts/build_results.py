"""Generate notebooks/results.ipynb — the figures, for showing the project.

    python scripts/build_results.py
    python scripts/build_results.py -o /tmp/preview.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb

`overview.ipynb` is the working view: every cell of the task x modality x model
cube, as tables, for deciding what to run next. This is the PRESENTATION view —
the figures that carry the project's actual findings, in the order you would say
them out loud.

The notebook is in four parts, and the split is the point of it:

  I    the ESTABLISHED architectures (mdcnn / cisfusion / late_fusion, +tf) --
       the big fusion models the project started with, and what the modality
       and locus ablations on them say
  II   the PIVOT -- SHAP over those models says the signal is a handful of
       columns, so stop encoding sequence and tokenize the deviations
  III  the EXPERIMENTAL models -- six aggregators over that variant tokenizer,
       run as measurements rather than as a leaderboard
  IV   `locusfusion` -- what those measurements were assembled into

Each part carries the other side's best cell as a reference line or a single
marked bar, so neither is read in isolation, and neither is read as if it were
the same experiment.

Both notebooks read the same tidy table (`results_frame.load_frame`) and the
same baseline tables (`bigtb_baselines`), so the two cannot disagree about what
ran.

Figures are matplotlib, light surface, dark ink. **A figure title names the
subject and the slice, never the finding** — the claim goes in the markdown
around it, where it can be qualified, and a title cannot carry a caveat. Nothing
explanatory is drawn inside a chart either (no "0.5 would be chance", no "see
the note"); what may be drawn is axis labels, data values, series names,
statistical results, and reference lines labelled with what they are. See
`~/.claude/CLAUDE.md`.

Also deliberately: no text is ever drawn on top of a saturated fill — every value label sits outside its bar in ink
colour — and the tables shade with pale tints under dark text rather than the
reverse. Colour is used for one job only, identity: blue is the established
architectures, orange is the experimental variant-token family (deep orange for
`locusfusion`, which is that family's product rather than one of its probes),
aqua is the non-neural baseline. Where the story is a single relationship rather
than several categories (parameters vs accuracy), the chart is one hue with
direct labels.
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


# ============================================================== title
md("""
# Genotype → phenotype antibiotic resistance in *M. tuberculosis*

Predicting per-drug resistance (R/S) for 11 antibiotics from gene sequence, over
**17,942 clinical isolates**, against the two published CNNs of the BIG-TB
benchmark.

**The question this project asks.** BIG-TB feeds its CNN a DNA one-hot and
nothing else. We ask what *derived and adjacent* views of the same locus are
worth — the amino-acid translation, its physicochemical profile, the
WHO-catalogue promoter window upstream of it — and whether an architecture that
can **pair** those views beats one that merely concatenates them.

**The arc, in one paragraph.** We built the large fusion models first
(`cisfusion`, `late_fusion`, `mdcnn`) and ablated them across modalities and
locus sets. SHAP on those models then said something that changed the design:
the attribution over a 4,500–19,000-column input collapses onto **a handful of
columns**, and separately the genome census says the median isolate deviates
from the H37Rv reference at only ~14 columns across all 19 loci. So the sequence
encoder is mostly spending capacity on constants. That motivated a **variant
tokenizer** — one token per deviation from reference — and a family of six
deliberately far-apart **aggregators** over it, each run as a measurement of one
question rather than as a contender. `locusfusion` is what those measurements
were assembled into: the same tokenizer, fused within a locus and then across
loci, at **a tenth of `mdcnn`'s parameters and roughly 1% of `cisfusion`'s**.

**How every number here is judged.** 5-fold cross-validated AUC, macro-averaged
over the 11 drugs shared with the baselines. Held-out test AUC is recorded
because it is what the papers publish, but it is one 20% split of one best-fold
model and swings ±0.05 on the small drugs. Single seed throughout, so
**differences under ~0.01 are not resolved** — they are drawn here, but they are
not claims.

> Generated — edit `scripts/build_results.py`, not the `.ipynb`.
> `python scripts/build_results.py && jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb`
""")

# ============================================================== setup
code('''
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import display, Image as IPImage

_here = Path.cwd()
PROJECT = next(p for p in [_here, *_here.parents] if (p / "bigtb_ref.py").exists())
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from results_frame import load_frame, TASKS, STD_MODSETS
from bigtb_baselines import SD_BASE_CV, MD_BASE_CV, ALL_DRUGS

frame, meta = load_frame()

# ---------------------------------------------------------------- families
# The split this notebook is organised around, and the ONLY place it is
# defined. `models/__init__.py` draws the same line: DELTA_ARCHS is exactly
# locusfusion + EXPERIMENTAL_MODELS, the architectures that consume
# reference-difference input, and everything else reads dense sequence.
AGGREGATORS = ["catalogue", "additive", "noisyor", "gatedpool", "deepsets", "fm"]
EXPERIMENTAL = set(AGGREGATORS) | {"locusfusion"}      # the variant-token family
BASELINE_MODELS = {"sparse_baseline", "sparse_baseline_trainvocab"}
ESTABLISHED = (set(meta["MODELS"]) - EXPERIMENTAL - BASELINE_MODELS)


def family(model):
    if model in BASELINE_MODELS:
        return "baseline"
    return "experimental" if model in EXPERIMENTAL else "established"


frame["family"] = frame.model.map(family)

# ---------------------------------------------------------------- palette
# Light surface only. Text NEVER sits on a saturated fill anywhere in this
# notebook -- value labels go outside the bar, in ink. Colour does one job:
# identity. Blue = the established architectures, orange = the experimental
# variant-token family, aqua = the baseline that is not a network.
SURFACE   = "#fcfcfb"   # chart surface
INK       = "#0b0b0b"   # primary text
INK2      = "#52514e"   # secondary text
MUTED     = "#898781"   # axis / tick labels
GRID      = "#e1e0d9"   # hairline gridline
AXIS      = "#c3c2b7"   # baseline / axis rule
BLUE      = "#2a78d6"   # established architectures
ORANGE    = "#eb6834"   # experimental aggregators
RUST      = "#a8420f"   # locusfusion -- same family, deeper: it is the product
AQUA      = "#1baf7a"   # the baseline that is not a network
PALE      = "#f0efec"   # neutral fill
PALE_BLUE = "#cde2fb"   # header tint, established
# Tints of the family hue, for when a chart compares one model against ITSELF
# at different settings. Task and locus set are not identities, so they get a
# sequential ramp inside the family colour rather than three unrelated hues --
# reusing BLUE for "per-drug loci" would say "established", which is a lie.
TINTS     = ["#f8d3c1", "#ef9264", ORANGE]


def hue(model):
    """Colour follows the entity, never its rank."""
    if model in BASELINE_MODELS:
        return AQUA
    if model == "locusfusion":
        return RUST
    return ORANGE if model in EXPERIMENTAL else BLUE


mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "figure.dpi": 120,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
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
    ax.grid(axis="x" if xgrid else "y", visible=xgrid or ygrid)
    if ygrid:
        ax.grid(axis="y", visible=True)
    return ax


# Direct labels beat a legend when there are ten points, but a fixed ladder of
# offsets collides as soon as two models sit within a factor of ~1.5 of each
# other on a log axis -- and this file is rebuilt whenever a run lands, so a
# hand-tuned ladder would need re-tuning every time. Placement is therefore
# greedy over candidate offsets, in points, first non-colliding one wins.
CANDIDATES = [(0, 17), (0, -29), (0, 35), (0, -47), (58, 5), (-58, 5),
              (58, -13), (-58, -13), (0, 53), (0, -65), (72, 22), (-72, 22)]


def label_points(ax, xs, ys, texts, fontsize=7.2, pad=3.0):
    """Direct labels with hairline leaders, placed so none overlaps another.

    Boxes are estimated from the text rather than measured with a renderer --
    ~0.60 em per character is close enough for DejaVu Sans at these sizes, and
    it keeps the function usable before the figure has been drawn. Points are
    labelled in x order and every placed box (plus every marker) is an obstacle
    for the next, so the result is deterministic and does not drift between
    rebuilds.
    """
    xs, ys, texts = list(xs), list(ys), list(texts)
    trans = ax.transData
    placed = []                                    # boxes already occupied
    for x, y in zip(xs, ys):                       # markers are obstacles too
        px, py = trans.transform((x, y))
        placed.append((px - 7, py - 7, px + 7, py + 7))

    def overlaps(box):
        l, b, r, t = box
        return any(not (r < L or l > R or t < B or b > T) for L, B, R, T in placed)

    for i in np.argsort(np.asarray(xs, dtype=float)):
        lines = texts[i].split(chr(10))     # literal newline, whatever this file escapes it as
        w = max(len(ln) for ln in lines) * fontsize * 0.60
        h = len(lines) * fontsize * 1.45
        px, py = trans.transform((xs[i], ys[i]))
        for dx, dy in CANDIDATES:
            # dy is the offset of the text ANCHOR; the box grows away from the
            # point, which is what keeps a label from covering its own marker.
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
                    va="bottom" if dy >= 0 else "top",
                    linespacing=1.4, zorder=5,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6,
                                    shrinkA=0, shrinkB=5))


def macro(task=None, model=None, modset=None, df=None):
    """Macro CV AUC over the drugs present, for a slice."""
    s = frame if df is None else df
    if task:   s = s[s.task == task]
    if model:  s = s[s.model == model]
    if modset: s = s[s.modset == modset]
    return s.groupby("drug").cv.mean().mean() if len(s) else float("nan")


# One row per (task, modality set, model) that finished ALL 11 shared drugs.
# Every figure below slices this; an incomplete cell is never averaged against
# a complete one.
best = (frame.groupby(["task", "modset", "model", "family"])
        .agg(cv=("cv", "mean"), n=("drug", "nunique"),
             params=("n_params", "mean"), loci=("n_loci", "max"))
        .reset_index().query("n == 11").sort_values("cv", ascending=False))


def top_of(fam, task=None, df=None):
    """Best complete cell for one family (optionally within one task)."""
    s = best if df is None else df
    s = s[s.family == fam]
    if task:
        s = s[s.task == task]
    return None if s.empty else s.sort_values("cv", ascending=False).iloc[0]


SD_BASE, MD_BASE = SD_BASE_CV.mean(), MD_BASE_CV.mean()
FOLD_SD = frame[frame.task != "multi-drug"].cv_sd.mean()
print(f"{meta['n_rows']:,} drug-runs  |  {meta['n_cells']} (task x modality x model) cells")
print(f"established  : {', '.join(sorted(ESTABLISHED))}")
print(f"experimental : {', '.join(sorted(EXPERIMENTAL))}")
print(f"baseline     : {', '.join(sorted(BASELINE_MODELS))}")
print(f"baselines -- SD-CNN {SD_BASE:.4f} (leak-corrected)   MD-CNN {MD_BASE:.4f}")
print(f"mean per-drug fold SD {FOLD_SD:.4f}  <- the resolution limit on every number here")
''')

# ============================================================== headline
md("""
---
## 1. Where the project stands

Two published CNNs are the reference points: **SD-CNN** (one model per drug) and
**MD-CNN** (13 drugs from one shared trunk). The SD-CNN number below is
**leak-corrected** — its published figure is inflated by a train/test
`stratify` mismatch that put ~80% of its "test" isolates in training, which we
found while reproducing it and re-scored from the authors' own saved models.
MD-CNN needed no correction and we reproduced it end to end (macro CV 0.9212
against their 0.9222).

The two tiles in the middle are the whole story of this notebook: the two
families land in the same place, and one of them is two orders of magnitude
smaller.

> **"Best cell" means the max over that model's five modality sets**, so it
> favours whichever model varies most across them — and the two families differ
> a lot in exactly that. It is the right tile for *how high has anything
> reached*; it is the **wrong** number for *which architecture is better*. §12
> pairs them cell by cell, and the two orderings disagree.
""")

code('''
est = top_of("established")
exp = top_of("experimental")
bas = top_of("baseline")

tiles = [
    ("Best ESTABLISHED cell", f"{est.cv:.4f}",
     f"{est.model} · {est.modset}\\n{int(est.loci)} loci · {est.params/1e6:.1f}M params",
     BLUE),
    ("Best EXPERIMENTAL cell", f"{exp.cv:.4f}",
     f"{exp.model} · {exp.modset}\\n{int(exp.loci)} loci · {exp.params/1e6:.2f}M params",
     RUST),
    ("Both vs MD-CNN", f"{est.cv - MD_BASE:+.4f}  /  {exp.cv - MD_BASE:+.4f}",
     f"published joint baseline {MD_BASE:.4f}\\nall three see all 19 loci",
     AXIS),
    ("Cells run", f"{meta['n_cells']}",
     f"{meta['n_rows']:,} drug-runs\\n{len(meta['MODELS'])} models",
     AXIS),
]

fig, axes = plt.subplots(1, 4, figsize=(11.6, 2.0))
for ax, (label, value, sub, colour) in zip(axes, tiles):
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=PALE, edgecolor="none", zorder=0))
    # the family colour is a 3pt rule under the tile, never a fill behind text
    ax.add_patch(plt.Rectangle((0, 0), 1, 0.035, transform=ax.transAxes,
                               facecolor=colour, edgecolor="none", zorder=1))
    ax.text(0.5, 0.82, label, ha="center", va="center", transform=ax.transAxes,
            fontsize=8.5, color=INK2)
    ax.text(0.5, 0.52, value, ha="center", va="center", transform=ax.transAxes,
            fontsize=20 if len(value) <= 6 else 15, color=INK)
    ax.text(0.5, 0.18, sub, ha="center", va="center", transform=ax.transAxes,
            fontsize=7, color=INK2, linespacing=1.6)
fig.suptitle("Macro CV AUC over the 11 shared drugs", y=1.10, fontsize=11,
             color=INK, fontweight="bold")
plt.tight_layout(); plt.show()

print(f"{'family':<14}{'best cell':>10}{'params':>10}   which cell")
for name, r in (("established", est), ("experimental", exp)):
    print(f"{name:<14}{r.cv:>10.4f}{r.params/1e6:>9.2f}M   {r.model} · {r.modset}")
# same task as `est`, so the ratio is between cells a reader could choose
# between -- the largest established cell overall is a multi-drug one
_big = best[(best.family == "established") & (best.task == est.task)] \
       .nlargest(1, "params").iloc[0]
print(f"\\n{exp.model} is {est.params / exp.params:.0f}x smaller than {est.model} "
      f"for {exp.cv - est.cv:+.4f} of AUC, and {_big.params / exp.params:.0f}x "
      f"smaller than {_big.model} ({_big.params/1e6:.0f}M).")
print(f"\\nsparse L1-logistic baseline (no GPU, no network): {bas.cv:.4f}"
      f"  at {int(bas.loci)} loci")
''')

md("""
### How this notebook is organised

| part | what is in it | colour |
|---|---|---|
| **I — §2–§6** | the **established** architectures: `mdcnn`, `cisfusion`, `late_fusion` and their transformer-trunk variants. What the modalities buy, what the loci buy, what parameters buy. | blue |
| **II — §7** | **the pivot.** SHAP over those models, and what it says the input actually is. | — |
| **III — §8–§11** | the **experimental** models: six aggregators over the variant tokenizer, run as measurements. | orange |
| **IV — §12** | **`locusfusion`** — the same tokenizer with a locus hierarchy, which is what Part III's measurements were assembled into. | deep orange |
| **V — §13–§14** | coverage, the full table, and every reason these numbers could be wrong. | — |

Parts I and III each carry the other's best cell as a reference, so neither is
read in isolation. They are **not** the same experiment and are never stacked
into one ranking without saying so.
""")

# ==================================================================
# ============================ PART I ==============================
# ==================================================================
md("""
---
---

# Part I — The established architectures

`late_fusion` (our per-block encoder net), `mdcnn` (BIG-TB's own
locus-as-channel topology), `cisfusion` (promoter and CDS concatenated per
locus), and the `+tf` variants that swap a transformer trunk in for the
convolutional one. These read **dense sequence**: a one-hot over every column of
every locus. They are what the project's modality question was originally asked
with, and every ablation in §4–§6 is an ablation of these.

**The experimental family is excluded from §2–§6** and appears only as a marked
reference bar, because it reads a different input representation. Mixing the two
into one ranking would attribute to architecture what is really encoding.
""")

# ---------------------------------------------------------- 2. by task
md("""
## 2. The three tasks

Three tasks: one model per drug on that drug's own 2–3 genes (`single-drug`),
one model per drug on all 19 curated loci (`single-drug, all loci`), and one
model with 11 outputs (`multi-drug`). Bars are macro CV AUC at each model's best
modality set; the vertical rule is that task's published baseline.

The last bar in each panel, in **deep orange**, is that task's best
**experimental** cell — carried here purely so the blue bars are not read in a
vacuum. It is Part III and Part IV's subject, not Part I's, and because every
bar is a max over that model's modality sets it is **not** a fair head-to-head;
§12 does that pairing properly.
""")

code('''
# One bar per MODEL, at that model's best modality set. All 40 cells per panel
# was unreadable, and "how good is each model" is the question; the modality
# axis is figure 4's job and the full list is section 14.
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharex=True)
# every panel gets the SAME row height, so a 4-model panel does not draw bars
# three times as thick as a 13-model one and read as three times as important
_est = best[best.family == "established"]
_rows = max(_est[_est.task == t].model.nunique() for t in TASKS) + 1
for ax, task in zip(axes, TASKS):
    s = (_est[_est.task == task].sort_values("cv", ascending=False)
         .groupby("model", as_index=False).first())
    ref = top_of("experimental", task)          # one comparison bar, marked
    if ref is not None:
        s = pd.concat([s, ref.to_frame().T], ignore_index=True)
    s["label"] = s.model + "   ·" + s.modset
    s = s.sort_values("cv")
    base = SD_BASE if task == "single-drug" else MD_BASE
    name = "SD-CNN" if task == "single-drug" else "MD-CNN"
    y = np.arange(len(s))
    ax.barh(y, s.cv, height=0.68, color=[hue(m) for m in s.model],
            edgecolor="none", zorder=2)
    for yi, v in zip(y, s.cv):                       # labels OUTSIDE the bar
        ax.text(v + 0.008, yi, f"{v:.4f}", va="center", ha="left",
                fontsize=7, color=INK)
    ax.axvline(base, color=INK2, lw=1.2, zorder=3)
    # under the bars, not over them -- at the top it collided with the title
    ax.text(base, -1.15, f"{name} {base:.4f} ", color=INK2, fontsize=7.5,
            va="center", ha="right")
    ax.set_yticks(y); ax.set_yticklabels(s.label, fontsize=7.5)
    for tick, m in zip(ax.get_yticklabels(), s.model):
        if m not in ESTABLISHED:
            tick.set_color(RUST)                     # name the outsider as one
    # 0.5 is chance for an AUC, so the axis starts there rather than at 0.
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
Read **down** a panel for what the model choice buys, and **across** panels for
what the task buys. Three things to take from it:

- **Locus set beats multi-task sharing.** The middle panel (one model per drug,
  19 loci) is the strongest of the three. "Joint models win" was confounded —
  joint runs saw 19 loci and single-drug runs saw 2–3. Matched at 19, the joint
  advantage is ≈0 or negative.
- **The transformer trunks (`+tf`) lose, everywhere they ran.** That is the
  failure Part II explains, and it is why Part III exists.
- **The experimental reference bar is level with the top blue bar** at 19 loci
  and behind it at 2–3. Hold that thought until §12.
""")

# ---------------------------------------------------------- 3. per drug
md("""
---
## 3. Per-drug, against both baselines

The macro hides everything. Each drug is a different mechanism, a different
class balance, and a different amount of headroom — RIFAMPICIN starts at 0.976
because the *rpoB* RRDR is a tight positional signal a one-hot already captures,
while ETHIONAMIDE starts at 0.69.
""")

code('''
# Established only, and single-drug only: the baselines are per-drug models, so
# a joint cell is not the like-for-like comparison the reader will assume.
_sd = frame[(frame.family == "established") & (frame.task != "multi-drug")]
bestpd = _sd.sort_values("cv", ascending=False).groupby("drug").first()
order = (bestpd.cv - SD_BASE_CV).sort_values().index
y = np.arange(len(order))

fig, ax = plt.subplots(figsize=(9.5, 5.6))
for yi, d in zip(y, order):
    lo, hi = SD_BASE_CV[d], bestpd.cv[d]
    ax.plot([lo, hi], [yi, yi], color=AXIS, lw=1.6, zorder=1,
            solid_capstyle="round")
ax.scatter(SD_BASE_CV[order], y, s=46, color=MUTED, zorder=3,
           edgecolor=SURFACE, linewidth=1.6, label="SD-CNN (leak-corrected)")
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

_won = int((bestpd.cv[order] > SD_BASE_CV[order]).sum())
print(f"ahead of SD-CNN on {_won} of {len(order)} drugs")
''')

md("""
**Ahead of SD-CNN on all 11 drugs.** The gap is widest exactly where the
baseline had the most headroom, and narrowest on RIFAMPICIN, which had almost
none.

`single-drug, all loci` cells **cannot** be quoted against SD-CNN — that
baseline sees the per-drug gene map, so it is a different input. They *are*
input-matched to MD-CNN, which uses all 19 by its own rule.
""")

# ---------------------------------------------------------- 4. modalities
md("""
---
## 4. What the extra modalities buy — and where

The project's core question, asked of the established architectures. Averaged
over all 11 drugs the gain is small, but averaging destroys the result: two
drugs carry almost all of it and five get nothing. Each bar is that drug's best
single-added-modality cell minus its own DNA-only cell, at matched loci and
matched architecture.
""")

code('''
sd = frame[(frame.task == "single-drug") & (frame.family == "established")]
sd = sd[sd.model.isin(["mdcnn", "cisfusion", "late_fusion"])]
dna = sd[sd.modset == "dna"].groupby(["drug", "model"]).cv.mean()
gain = {}
for d in ALL_DRUGS:
    rows = sd[(sd.drug == d) & (sd.modset != "dna")]
    if rows.empty:
        continue
    per = rows.groupby(["modset", "model"]).cv.mean()
    deltas = {}
    for (ms, mo), v in per.items():
        # only the single-added sets, so the label NAMES a modality; the
        # all-four cell is reported in the prose instead
        if (d, mo) in dna.index and ms in ("dna_protein", "dna_biophysical",
                                           "dna_regulatory"):
            deltas.setdefault(ms, []).append(v - dna[(d, mo)])
    if deltas:
        ms, vals = max(((k, v) for k, v in deltas.items()),
                       key=lambda kv: np.mean(kv[1]))
        gain[d] = (np.mean(vals), ms.replace("dna_", ""))

g = pd.Series({k: v[0] for k, v in gain.items()}).sort_values()
which = {k: v[1] for k, v in gain.items()}
y = np.arange(len(g))
promoter = {"ETHIONAMIDE", "ISONIAZID"}
colors = [ORANGE if d in promoter else BLUE for d in g.index]

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.barh(y, g.values, height=0.66, color=colors, edgecolor="none", zorder=2)
ax.axvline(0, color=AXIS, lw=1)
for yi, (d, v) in zip(y, g.items()):
    off = 0.0022 if v >= 0 else -0.0022
    ax.text(v + off, yi, f"{v:+.3f}   {which[d]}", va="center",
            ha="left" if v >= 0 else "right", fontsize=7.5, color=INK)
ax.set_yticks(y); ax.set_yticklabels(g.index, fontsize=8.5)
ax.set_xlim(min(g.min() * 1.6, -0.01), g.max() * 1.55)
ax.set_xlabel("best SINGLE added modality minus that drug's own DNA-only cell (CV AUC)")
ax.set_title("CV AUC gain from one added modality, by drug", loc="left")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
          ["promoter-mechanism drugs"], loc="lower center",
          bbox_to_anchor=(0.5, -0.20))
tidy(ax); plt.tight_layout(); plt.show()

print(f"macro over the 11 drugs: {g.mean():+.4f}     "
      f"median: {g.median():+.4f}     five drugs at or below {g.nsmallest(5).max():+.4f}")
''')

md("""
**The gain lands where the resistance mechanism predicts it.**

- **ETHIONAMIDE** and **ISONIAZID** gain from `regulatory` — both are driven by
  the *fabG1–inhA* operon promoter (`c-15t`), which is structurally invisible to
  a coding-sequence one-hot. No amount of CDS modelling can recover a variant
  that is not in the CDS.
- **PYRAZINAMIDE** gains from `protein` / `biophysical` — *pncA* loss of
  function, where hundreds of distinct inactivating substitutions exist and most
  are absent from training. "Does this substitution break the protein"
  generalises where memorising positions cannot.
- **KANAMYCIN** and **AMIKACIN** gain nothing, and should not: their target is
  *rrs*, 16S **rRNA**, which has no protein product for the descriptors to
  describe.

A negative control that came out right is worth as much as a positive one. This
figure is also the hypothesis Part III ends up testing directly, with a model
small enough that the answer is a single controlled comparison rather than an
average over architectures (§10).
""")

# ---------------------------------------------------------- 5. loci
md("""
---
## 5. What the loci buy

Same established models, same modalities, same schedule — only the gene set
changes, from each drug's own 2–3 genes to all 19 curated loci. Paired on
(drug, modality set, model), so a cell that exists on only one side is dropped
rather than averaged in.
""")

code('''
_e = frame[frame.family == "established"]
a = _e[_e.task == "single-drug"]
b = _e[_e.task == "single-drug, all loci"]
m = a.merge(b, on=["drug", "modset", "model"], suffixes=("_a", "_b"))
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
    ax.scatter([r["per-drug loci"]], [yi], s=26, color=SURFACE,
               edgecolor=c, linewidth=1.6, zorder=3)
    ax.scatter([r["all 19 loci"]], [yi], s=42, color=c, zorder=3,
               edgecolor=SURFACE, linewidth=1.2)
    ax.text(max(r["per-drug loci"], r["all 19 loci"]) + 0.006, yi,
            f"{r['delta']:+.3f}", va="center", ha="left", fontsize=7.5, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([f"{d}   ({int(t.n_lo[d])} → 19)" for d in t.index], fontsize=8)
ax.set_xlim(0.66, 1.02); ax.set_xlabel("CV AUC")
ax.set_title("CV AUC at per-drug gene map vs. all 19 loci, established models",
             loc="left")
ax.legend([Line2D([0], [0], color=BLUE, lw=2,
                  marker="o", markerfacecolor=SURFACE, markersize=5),
           Line2D([0], [0], color=ORANGE, lw=2)],
          ["hollow = per-drug loci, filled = all 19 loci", "discussed below"],
          loc="lower center", bbox_to_anchor=(0.5, -0.19), ncol=2)
tidy(ax); plt.tight_layout(); plt.show()

print(f"macro over the paired drugs: {t.delta.mean():+.4f}")
''')

md("""
Handing a single-drug model the full 19 loci is **larger than the modality
gain**, and the single cheapest improvement in the project.

Two rows are not what they look like. **KANAMYCIN gains most** because its
per-drug map is a *single* gene (`rrs`), so it also loses the *eis* promoter —
its best-known resistance mechanism — and at 19 loci that comes back.
**LEVOFLOXACIN's loss is not a finding**: 269 labelled isolates and a fold SD of
~0.037, so it is noise with a large amplitude.

The same comparison for the experimental family is in §12, and it is roughly
twice this size — the locus set matters *more* to a model that tokenizes
deviations, because more loci means more tokens rather than more constants.
""")

# ---------------------------------------------------------- 6. parameters
md("""
---
## 6. Parameters buy nothing above ~1M

Every point is a complete **established** cell at 19 loci — same input, same
schedule, same seed. The x-axis spans nearly two orders of magnitude.
""")

code('''
# One point per architecture at its BEST modality set: plotting all five
# modality cells stacked five labels on one spot. The story is a single
# relationship, so it is one hue with direct labels.
s19 = (best[(best.task == "single-drug, all loci") & (best.family == "established")]
       .sort_values("cv", ascending=False).groupby("model", as_index=False).first())
s19 = s19.dropna(subset=["params"]).sort_values("params")
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.scatter(s19.params, s19.cv, s=90, color=BLUE, edgecolor=SURFACE, linewidth=1.6,
           zorder=3)
for i, r in enumerate(s19.itertuples()):           # direct label, no legend
    ax.annotate(f"{r.model}\\n{r.params/1e6:.2f}M  {r.cv:.4f}", (r.params, r.cv),
                textcoords="offset points", xytext=(0, 14 if i % 2 == 0 else -28),
                ha="center", fontsize=7.4, color=INK2, linespacing=1.5)
ax.set_xscale("log"); ax.set_xlabel("parameters (log scale)")
ax.set_ylabel("macro CV AUC")
lo, hi = s19.cv.min(), s19.cv.max()
ax.set_ylim(lo - 0.010, hi + 0.009)
ax.set_xlim(s19.params.min() / 2.4, s19.params.max() * 2.4)
ax.set_title("Parameter count vs. macro CV AUC, established models, all 19 loci",
             loc="left")
tidy(ax, ygrid=True); plt.tight_layout(); plt.show()

# The claim is about the TOP of the range, not its span: the two best cells are
# the ones a reader would choose between, so quote their gap rather than a
# max-minus-min that the worst model happens to set.
_t2 = s19.nlargest(2, "cv")
print(f"top two by CV: {' and '.join(_t2.model)} — "
      f"{_t2.params.max() / _t2.params.min():.0f}x the parameters, "
      f"{_t2.cv.max() - _t2.cv.min():.4f} of AUC apart "
      f"(mean per-drug fold SD {FOLD_SD:.4f})")
''')

md("""
The two best cells are **9× apart in parameters and 0.0015 apart in AUC** — an
order of magnitude inside the fold SD, which runs 0.003–0.037 by drug. And the
one model that is clearly worse, `late_fusion`, is not worse for lack of
capacity: it is the *second largest* here, and ~90% of its 45.9M parameters sit
in the single fully-connected layer that follows its flatten. Size is
uncorrelated with rank across all three.

Only three architectures reach 19 loci, so this panel is thin on its own. §11
draws the same axis with both families on it and spans four orders of magnitude.

This is consistent with the two capacity experiments the project already ran:
`joint_capacity` (no arm beat its control; dropout and weight decay clearly
hurt) and `setfusion_scaling` (62 arms across four width axes, closed nothing).

**Above roughly a million parameters, capacity is not the binding constraint on
this task.** That is the last result of Part I and the first premise of Part II:
if more capacity is not the answer, the input representation has to be.
""")

# ==================================================================
# ============================ PART II =============================
# ==================================================================
md("""
---
---

# Part II — The pivot: what SHAP said about the input

Part I ends with capacity ruled out. The next question is what those models are
actually reading, and SHAP answers it. `scripts/shap_attribution.py` runs
GradientSHAP over the best fold of each `full_run_v2` `mdcnn` cell and writes
per-column attribution to `results/analysis/shap/`.

## 7. Where the attribution actually lands

For each drug: how many input columns carry 90% of the total |SHAP|, against
how many columns the model was given.
""")

code('''
SHAP_DIR = PROJECT / "results" / "analysis" / "shap" / "full_run_v2" / "mdcnn"
_have_shap = (SHAP_DIR / "columns_all.csv").is_file()
if not _have_shap:
    print(f"no SHAP tables under {SHAP_DIR} -- section 7 is skipped")
else:
    cols = pd.read_csv(SHAP_DIR / "columns_all.csv")
    blks = pd.read_csv(SHAP_DIR / "blocks_all.csv")
    # `columns_all.csv` keeps only the top columns of each block, so a column's
    # share of the WHOLE input is its share of its block times that block's own
    # share. Anything below the per-block cutoff is missing, which is why the
    # captured mass is tracked and censored rows are drawn as bounds.
    cm = cols.merge(blks[["drug", "cell", "block", "share"]],
                    on=["drug", "cell", "block"])
    cm["total_share"] = cm.share_of_block * cm.share
    width = blks.groupby(["drug", "cell"]).length.sum()

    rows = []
    for (d, cell), g in cm.groupby(["drug", "cell"]):
        g = g.sort_values("total_share", ascending=False)
        cum = g.total_share.cumsum()
        captured = float(g.total_share.sum())
        rows.append(dict(drug=d, cell=cell, n_cols=int(width[(d, cell)]),
                         n90=int((cum < 0.90).sum() + 1),
                         top1=float(g.total_share.iloc[0]),
                         captured=captured,
                         censored=captured < 0.90))   # cutoff bit before 90%
    conc = pd.DataFrame(rows).query("cell == 'all_modalities'") \\
             .set_index("drug").sort_values("n_cols")

    y = np.arange(len(conc))
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    for yi, (d, r) in zip(y, conc.iterrows()):
        ax.plot([r.n90, r.n_cols], [yi, yi], color=AXIS, lw=1.6, zorder=1,
                solid_capstyle="round")
        ax.scatter([r.n_cols], [yi], s=44, color=MUTED, zorder=3,
                   edgecolor=SURFACE, linewidth=1.4)
        ax.scatter([r.n90], [yi], s=54, zorder=4, edgecolor=BLUE, linewidth=1.8,
                   color=SURFACE if r.censored else BLUE)
        note = "≥" if r.censored else ""
        ax.text(r.n90 * 0.82, yi, f"{note}{r.n90}", va="center", ha="right",
                fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(conc.index, fontsize=8.5)
    ax.set_xlim(0.5, conc.n_cols.max() * 2.4)
    ax.set_xlabel("input columns (log scale)")
    ax.set_title("Input columns carrying 90% of |SHAP|, mdcnn, all modalities",
                 loc="left")
    ax.legend([Line2D([0], [0], marker="o", color=SURFACE, markerfacecolor=BLUE,
                      markeredgecolor=BLUE, markersize=7, lw=0),
               Line2D([0], [0], marker="o", color=SURFACE, markerfacecolor=SURFACE,
                      markeredgecolor=BLUE, markersize=7, lw=0),
               Line2D([0], [0], marker="o", color=SURFACE, markerfacecolor=MUTED,
                      markeredgecolor=MUTED, markersize=7, lw=0)],
              ["columns carrying 90% of |SHAP|",
               "hollow = lower bound (table truncated per block)",
               "columns in the input"],
              loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.21))
    tidy(ax); plt.tight_layout(); plt.show()

    ok = conc[~conc.censored]
    print(f"resolved for {len(ok)}/{len(conc)} drugs; "
          f"median columns carrying 90% of the attribution: {ok.n90.median():.0f}"
          f"   (out of {conc.n_cols.min():,}-{conc.n_cols.max():,} columns)")
    print(f"single top column takes {ok.top1.median():.0%} of the budget, median over those drugs")
''')

md("""
**Ninety percent of what an established model reads is a handful of columns out
of thousands.** On the three censored drugs the per-block cutoff in the saved
table bites before 90% is reached, so their counts are lower bounds and are
drawn hollow. What they have in common is that their signal is genuinely spread
across many positions rather than sitting on one, which is the honest reading of
a censored count and not a reason to discount it.

Two independent facts point the same way:

1. **The attribution is sparse** — this figure.
2. **The input is nearly constant.** Censused over 19 loci × 17,943 isolates
   against each alignment's `MT_H37Rv` row, the median isolate differs from the
   reference at **0–3 columns of a 2.5 kb gene, and at 14 columns across all 19
   loci.** *M. tuberculosis* is clonal; there is very little sequence variation
   to encode in the first place.

Together they say a dense-sequence encoder spends ~99.9% of its capacity
restating constants. The `token_signal` diagnostic measured that directly on
`setfusion`: only **0.14% of an encoded token varied with the genotype**,
attention sat flat at exactly 1/8, and plain logistic regression on the
encoder's own output beat the trained model. That is why every `+tf` bar in §2
loses, and it is not a tuning failure.

### The honest caveat, which is also a result

We tested SHAP's other claim and it failed. The all-modalities model gave its
**DNA input 1.0%** of the attribution budget; `shap_followup` retrained without
DNA to see whether that was a real statement about predictive value.
""")

code('''
_ab = (frame[(frame.task == "single-drug") & (frame.model == "mdcnn")]
       .groupby("modset").cv.mean())
_show = [m for m in ["all_modalities", "dna", "no_regulatory", "no_dna",
                     "regulatory_only"] if m in _ab.index]
lab = {"all_modalities": "dna+protein+biophys+reg", "dna": "dna",
       "no_regulatory": "dna+protein+biophys", "no_dna": "protein+biophys+reg",
       "regulatory_only": "regulatory"}
for k in _show:
    bar = "█" * int(round((_ab[k] - 0.5) * 60))
    print(f"{lab[k]:>26s}  {_ab[k]:.4f}  {bar}")
print(f"\\ndropping DNA costs {_ab.get('no_dna', np.nan) - _ab['all_modalities']:+.4f} "
      f"-- from an input SHAP scored at 1% of the budget")
''')

md("""
Dropping the input SHAP called 1% costs ~0.10 AUC. **Attribution share
mis-ranks predictive value**, and every share number in the SHAP notebook needs
that caveat. What survives is the *positional* claim in the figure above — where
the signal sits along a locus, which the follow-up did not contradict and which
is the claim the tokenizer is built on.

### So: stop encoding the sequence, tokenize the deviation

Run on reference-difference input (`--delta`, which zeroes every column matching
the real `MT_H37Rv` record) and emit **one token per surviving column**. A token
then exists *only* where the genotype deviates from wild type, so 100% of it
varies with the genotype by construction, and:

- a susceptible isolate is the **empty set**, scored against a learned `[WT]`
  sentinel per locus;
- 19 loci × up to 4,066 columns collapses to a **median ~14 tokens**;
- exact position **survives** — each token carries its own coordinate through a
  sinusoidal encoding, better than `mdcnn`'s 9-fold pooling.

Once you have done that, there is no long sequence left to encode, and the
question stops being *how do I read a genomic block* and becomes **how do I
aggregate ~14 sparse pieces of evidence**. That is Part III.
""")

# ==================================================================
# ============================ PART III ============================
# ==================================================================
md("""
---
---

# Part III — The experimental models

Six aggregators (`models/experimental_models.py`), one shared tokenizer, run as
**measurements rather than as a leaderboard**. Every one consumes an identical
*flat* variant token set — every block's variants concatenated, no locus
hierarchy — deliberately, so that the only thing varying across the six is the
aggregation operator. Each answers one question that the established models
cannot ask.

**None of the six beats the best established model, and that is not the point of
them.** The three things they measure are in §10 and §11, and they are what
Part IV was designed from.

### The argument against softmax attention

Softmax **normalises**, so it is a *relative* selector. With one informative
token and thirteen neutral ones the weights must sum to 1, so the operator
cannot say "this token, absolutely, whatever else is present" — it has to spend
mass on the neutral tokens. That is the mechanism behind the flat 1/8 attention
`token_signal` measured: a structural property of the operator, not a training
failure. A needle detector wants an **absolute, monotone** aggregator, where
adding a neutral token cannot dilute the signal. So the six are chosen to sit
deliberately far apart on exactly that axis.
""")

md("""
## 8. What each of the six asks

| `--arch` | aggregator | the question it exists to answer |
|---|---|---|
| `catalogue` | learned scalar per **exact** variant id, summed | how far does pure memorisation get? (this *is* logistic regression on the variant matrix, and a learned WHO-style catalogue) |
| `additive` | `Σ w(features_v)` | does featurising the variant buy generalisation to substitutions never seen in training? |
| `noisyor` | `1 − Π(1 − p_v)` | "susceptible unless something confers resistance", as an architecture |
| `gatedpool` | sigmoid gate, no softmax | is *normalisation* the thing that broke attention here? |
| `deepsets` | `ρ([Σφ, max φ, log1p(count)])` | does attention buy anything at all over plain additivity? |
| `fm` | factorization machine, rank 8 | is epistasis worth anything, priced at `O(T·k)` instead of `O(T²)`? |

`catalogue` and `additive` are a **matched pair and the difference between them
is a measurement**: identical tokenizer, identical aggregator, identical
optimizer, one thing changed — whether a variant's weight comes from its
*identity* or from its *features* (locus, exact position, which residue it
became, the biophysical consequence). `catalogue`'s weight table is
zero-initialised, so a substitution absent from training contributes exactly
nothing at test time. It is the **control for generalisation**, not a contender.

Two properties that come free with staying additive, and that no model in Part I
has: `additive.contributions()` returns signed per-token contributions that
**sum exactly to the logit** — no SHAP, no sampling error, no share-vs-value
problem from §7 — and a learned `uncovered_w` per (block, drug) means **a
missing gene is distinguished from a wild-type gene**.
""")

# ---------------------------------------------------------- 9. the family
md("""
## 9. The family, at both locus settings

`all_modalities`, 5-fold CV, macro over 11 drugs. The rules are the best
**established** cell at each locus setting, so the reference is Part I rather
than another aggregator.
""")

code('''
fam = (best[best.model.isin(AGGREGATORS)]
       .pivot_table(index="model", columns="task", values="cv"))
fpar = (best[best.model.isin(AGGREGATORS)]
        .pivot_table(index="model", columns="task", values="params"))
_TT = [t for t in ["single-drug", "single-drug, all loci"] if t in fam.columns]
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
# The two reference rules sit 0.016 apart on a 0.11 axis, so their labels
# collide if both are centred on the rule. They are pushed to opposite sides,
# below the bars, where the one-bar `noisyor` row leaves the space free.
for i, task in enumerate(_TT):
    r = top_of("established", task)
    ax.axvline(r.cv, color=INK2, lw=1.1, ls="-" if i else "--", zorder=3)
    ax.text(r.cv, -1.05, (f"{r.model} {r.cv:.4f}, {task} " if i == 0
                          else f" {r.model} {r.cv:.4f}, {task}"),
            color=INK2, fontsize=7, va="center", ha="right" if i == 0 else "left")
ax.set_yticks(y); ax.set_yticklabels(fam.index, fontsize=9)
ax.set_ylim(-1.5, len(fam) - 0.4)
ax.set_xlim(0.84, 0.95)
ax.set_xlabel("macro CV AUC, all modalities (parameter count in parentheses)")
ax.set_title("Variant-set aggregators by locus setting", loc="left")
ax.legend(loc="lower left")          # lower right is where the 19-locus rule label goes
tidy(ax); plt.tight_layout(); plt.show()

print("not one of the six wins at either locus setting -- that is expected, and")
print("the three results below are what they were run for.\\n")
for task in _TT:
    r = top_of("established", task)
    b = fam[task].max()
    print(f"{task:24s}  best aggregator {b:.4f}   best established {r.cv:.4f} "
          f"({r.model})   gap {b - r.cv:+.4f}")
''')

md("""
**Headline: not one of the six wins.** They were not built to. `catalogue` is a
deliberate control, `noisyor` is a monotone architecture that *cannot* learn a
protective variant, and the other four are single-operator probes. Presenting
them as a leaderboard would be the worst version of this section — six models
nobody asked about, none of which wins, and a reader's takeaway of "a lot of
things were tried that did not work."

What they are for is the next two figures.
""")

# ---------------------------------------------------------- 10. the measurement
md("""
---
## 10. `catalogue` → `additive`: what featurising a variant is worth

The cleanest measurement in the project. Same tokenizer, same aggregator, same
optimizer; the only change is whether a variant's weight is looked up by its
*identity* or computed from its *features*.

The stated hypothesis was *pncA* / PYRAZINAMIDE — hundreds of distinct
inactivating substitutions, most absent from training, exactly the mechanism §4
attributes the biophysical gain to. **The measurement came back saying something
different, and better.**
""")

code('''
def cat_vs_add(task):
    s = frame[(frame.task == task) & (frame.modset == "all_modalities")]
    c = s[s.model == "catalogue"].set_index("drug").cv
    a = s[s.model == "additive"].set_index("drug")
    if c.empty or a.empty:
        return None
    d = (a.cv - c).dropna()
    return pd.DataFrame({"n": a.n_valid.reindex(d.index), "gain": d}).dropna()


_TASK = "single-drug, all loci"
cva = cat_vs_add(_TASK)
cva_pd = cat_vs_add("single-drug")

try:                                   # rho is the claim; p is nice to have
    from scipy.stats import spearmanr
    rho, pval = spearmanr(cva.n, cva.gain)
    _stat = f"Spearman(n, gain) = {rho:+.3f},  p = {pval:.4f}"
except Exception:
    rho = cva.n.corr(cva.gain, method="spearman")
    _stat = f"Spearman(n, gain) = {rho:+.3f}"

# Eleven points, four of them within a factor of 1.4 in n -- as a scatter the
# labels collide whatever ladder they are put on. Ordering the drugs BY cohort
# size and reading down is the same claim with no overlap, and the rank
# correlation is stated rather than eyeballed.
cva = cva.sort_values("n", ascending=False)
y = np.arange(len(cva))
fig, ax = plt.subplots(figsize=(9.2, 5.4))
ax.axvline(0, color=AXIS, lw=1, zorder=1)
for yi, r in zip(y, cva.itertuples()):
    ax.plot([0, r.gain], [yi, yi], color=ORANGE, lw=2, zorder=2,
            solid_capstyle="round")
    ax.scatter([r.gain], [yi], s=52, color=ORANGE, zorder=4,
               edgecolor=SURFACE, linewidth=1.4)
    ax.text(r.gain + 0.0018, yi, f"{r.gain:+.4f}", va="center", ha="left",
            fontsize=7.5, color=INK)
    if cva_pd is not None and r.Index in cva_pd.index:   # does it replicate?
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
          ["all 19 loci", "same measurement at per-drug loci"],
          loc="lower right", ncol=1)
ax.annotate(_stat, xy=(0.5, -0.135), xycoords="axes fraction", ha="center",
            fontsize=8.5, color=INK)
tidy(ax); plt.tight_layout(); plt.show()      # value axis is x here

print(f"{'drug':<14}{'n':>8}{'gain @19 loci':>15}{'gain @per-drug loci':>22}")
for d in cva.sort_values("n").index:
    g2 = cva_pd.gain.get(d, np.nan) if cva_pd is not None else np.nan
    print(f"{d:<14}{int(cva.n[d]):>8,}{cva.gain[d]:>15.4f}{g2:>22.4f}")
print(f"\\nmacro: {cva.gain.mean():+.4f} at 19 loci"
      + (f",  {cva_pd.gain.mean():+.4f} at per-drug loci" if cva_pd is not None else ""))
''')

md("""
**The gain is not about *pncA* at all** — PYRAZINAMIDE, the drug the hypothesis
named, gains +0.005. It is almost perfectly a **small-cohort effect**, and the
rank correlation with cohort size is about as strong as an 11-point correlation
can be.

The claim to make: *featurising a variant instead of memorising its identity
buys generalisation, and the size of that gain is set almost entirely by how
little labelled data the drug has.* Above roughly 8,000 isolates a learned
catalogue has seen enough; below it, features matter, and at n = 269 they are
worth 0.07 AUC.

That is a quantitative, mechanistic result about **sample efficiency**, and it
is the one that transfers: it says something usable about deploying to a new
drug, a new region, or a newly-approved compound where labels are scarce — which
is precisely the regime a learned WHO-style catalogue fails in.

`catalogue` is also the only model in the family whose parameter count scales
with the data (52k → 445k at 19 loci), and it is the worst of the five that ran
at both settings. Memorisation is expensive *and* it does not generalise.
""")

# ---------------------------------------------------------- 11. ablation
md("""
---
## 11. Two more measurements: the gate, and the price

**The attention ablation.** `gatedpool` is the minimal edit to the thing that
failed — the per-drug attention query with the softmax deleted, so the gate is
an *absolute* relevance score rather than a share of a fixed budget. `deepsets`
deletes the gate entirely: sum, max, count, and nothing that decides which
tokens matter. The decision rule was written before the run: *"if `gatedpool`
closes a gap that `deepsets` does not, normalisation was the problem; if
`deepsets` matches it, the gating was never doing anything."*
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
print(f"\\nmean per-drug fold SD: {FOLD_SD:.4f}  "
      f"-- both differences are several times smaller")
''')

md("""
**Deleting the gating machinery entirely costs nothing.** Combined with
`token_signal`'s flat 1/8 attention, the finding is that *on this task the
selection mechanism is not where the performance comes from* — a sum and a max
over the same tokens does the same job. Worth reporting precisely **because** it
is negative: it explains why three transformer arms failed, and it means the
contribution here is the **representation** (tokenize the variants, featurise
them), not the attention.

It also saved a sweep. The natural next move was more attention variants, which
`setfusion_scaling` had already shown is how you burn 62 arms learning nothing.

**And the price.** Put every complete cell at 19 loci on one axis, both families:
""")

code('''
p19 = (best[(best.task == "single-drug, all loci") & best.params.notna()]
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
# after the scale and the limits: label_points works in pixels, so it needs the
# data transform it will actually be drawn with
label_points(ax, p19.params, p19.cv,
             [f"{r.model}\\n{r.params/1e6:.3f}M  {r.cv:.4f}"
              for r in p19.itertuples()], fontsize=7.1)
ax.set_title("Parameter count vs. macro CV AUC, both families, all 19 loci",
             loc="left")
ax.legend(loc="lower right")
tidy(ax, ygrid=True); plt.tight_layout(); plt.show()

_add = p19[p19.model == "additive"]
_lf = p19[p19.model == "late_fusion"]
if len(_add) and len(_lf):
    a, l = _add.iloc[0], _lf.iloc[0]
    print(f"additive     {a.cv:.4f}  at {a.params:,.0f} parameters")
    print(f"late_fusion  {l.cv:.4f}  at {l.params:,.0f} parameters")
    print(f"             {a.cv - l.cv:+.4f} AUC for a {l.params / a.params:.0f}x "
          f"parameter reduction")
''')

md("""
`additive` — a **sum of per-variant weights, and nothing else** — beats every
`late_fusion` cell at 19 loci with roughly a thousandth of the parameters. `fm`
gets within 0.007 of it at another 3× smaller.

That is the sharpest possible statement of what Part II claimed: **the signal is
sparse and mostly additive.** `fm`'s rank-8 interaction term, which exists to
price epistasis, is *worse* than plain additivity at 19 loci — the interactions
cost more in variance than they recover in signal.
""")

# ==================================================================
# ============================ PART IV =============================
# ==================================================================
md("""
---
---

# Part IV — `locusfusion`: the measurements, assembled

Part III says: tokenize the deviations, featurise them, and do not bother with a
normalising selector. `locusfusion` is the model built from that, plus the one
thing the flat aggregators deliberately gave up.

**Fuse at the gene, then across genes.** Every architecture in Part I fuses the
modalities and the loci in the *same* step. This one is two stages: all of
*rpoB*'s evidence — its CDS one-hot, its translation, that translation's
biophysical profile, its promoter window — is fused into a single **locus
representation** first, and only those 19 locus representations then talk to
each other. A resistance mechanism is a property of a gene; a resistance
*phenotype* is a property of the set of genes. The six aggregators hold that
hierarchy flat on purpose, so that this is the one thing added back.

It inherits the rest directly: variant tokens on `--delta` input, a learned
`[WT]` sentinel per locus so a pan-susceptible isolate is 19 sentinels and
nothing else, exact coordinates through a sinusoidal encoding, and attribution
that is free and exact — `forward(..., return_attn=True)` names the locus and
column each drug read.
""")

code('''
# Built separately, because it is read off a real LocusFusionNet rather than
# off this frame: `python scripts/locusfusion_diagram.py --check` asserts its
# parameter total against the number the 19-locus run recorded.
_diagram = PROJECT / "results" / "figures" / "locusfusion_shapes.png"
if _diagram.is_file():
    display(IPImage(filename=str(_diagram)))
else:
    print("not built — run: python scripts/locusfusion_diagram.py --check")
print("--layout full draws the reference sheet: every stage, the 42-dim token "
      "layout, the parameter split.")
''')

md("""
## 12. Where it lands

Same figure logic as §5 and §6, for one model: the locus set moves it, the
modalities do not, and it does the job at 1% of Part I's parameters.
""")

code('''
lf = best[best.model == "locusfusion"].copy()
piv = lf.pivot_table(index="modset", columns="task", values="cv")
piv = piv.reindex(index=[m for m in STD_MODSETS if m in piv.index],
                  columns=[t for t in TASKS if t in piv.columns])

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4),
                         gridspec_kw={"width_ratios": [1.25, 1]})

# -- left: locusfusion across every task x modality cell it has run
ax = axes[0]
x = np.arange(len(piv.index)); w = 0.24
for i, task in enumerate(piv.columns):
    ax.bar(x + (i - (len(piv.columns) - 1) / 2) * (w + 0.045), piv[task],
           width=w, color=TINTS[i], edgecolor="none", label=task, zorder=2)
ax.set_xticks(x); ax.set_xticklabels(piv.index, fontsize=8)
ax.set_ylim(0.85, 0.945); ax.set_ylabel("macro CV AUC")
ax.axhline(MD_BASE, color=INK2, lw=1.1, zorder=3)
# left, where the shortest bars are -- on the right it sat on the tallest one
ax.text(-0.45, MD_BASE, f"MD-CNN {MD_BASE:.4f}", color=INK2,
        fontsize=7.5, va="bottom", ha="left")
ax.set_title("locusfusion: macro CV AUC by modality set and task", loc="left")
ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.30))
tidy(ax, xgrid=False, ygrid=True)

# -- right: locusfusion against the best established model MATCHED CELL BY
# CELL, which is the comparison the reader wants and the one a "best cell"
# headline gets wrong. Taking each model's max over five modality sets rewards
# variance: locusfusion's spread across them is ~0.001 and mdcnn's is ~0.008,
# so mdcnn draws five tickets on a wider distribution and its max wins even
# where its typical cell loses. Pairing on the modality set removes that.
ax = axes[1]
_T = "single-drug, all loci"
_rival = top_of("established", _T).model
_paired = (frame[(frame.task == _T) & frame.model.isin(["locusfusion", _rival])]
           .groupby(["modset", "model"]).cv.mean().unstack().dropna())
_paired = _paired.reindex([m for m in STD_MODSETS if m in _paired.index])
yy = np.arange(len(_paired))
for yi, (ms, r) in zip(yy, _paired.iterrows()):
    ahead = r.locusfusion > r[_rival]
    ax.plot([r[_rival], r.locusfusion], [yi, yi], lw=2, zorder=2,
            color=RUST if ahead else AXIS, solid_capstyle="round")
    ax.scatter([r[_rival]], [yi], s=42, color=BLUE, zorder=3,
               edgecolor=SURFACE, linewidth=1.4)
    ax.scatter([r.locusfusion], [yi], s=48, color=RUST, zorder=4,
               edgecolor=SURFACE, linewidth=1.4)
    ax.text(max(r.locusfusion, r[_rival]) + 0.0006, yi,
            f"{r.locusfusion - r[_rival]:+.4f}", va="center", ha="left",
            fontsize=7.5, color=INK)
ax.axvline(MD_BASE, color=INK2, lw=1.0, zorder=1)
ax.text(MD_BASE, -0.95, f"MD-CNN {MD_BASE:.4f} ", color=INK2, fontsize=7,
        va="center", ha="right")
ax.set_yticks(yy); ax.set_yticklabels(_paired.index, fontsize=8)
ax.set_ylim(-1.4, len(_paired) - 0.4)
ax.set_xlim(0.9155, 0.9272)
ax.set_xlabel("macro CV AUC — single-drug, all 19 loci")
ax.set_title(f"locusfusion vs. {_rival}, paired by modality set", loc="left")
# upper left: the only region with no data in it, and the bottom row is where
# the MD-CNN rule label goes
ax.legend([Line2D([0], [0], marker="o", lw=0, markersize=7, color=RUST),
           Line2D([0], [0], marker="o", lw=0, markersize=7, color=BLUE)],
          ["locusfusion", _rival], loc="upper left", ncol=1)
tidy(ax)

plt.tight_layout(); plt.show()

_d = piv[_T].max() - piv["single-drug"].max()
print(f"locusfusion, best cell:  {piv['single-drug'].max():.4f} at per-drug loci"
      f"  ->  {piv[_T].max():.4f} at 19 loci   ({_d:+.4f} from the locus count alone)")
print(f"spread across all five modality sets at 19 loci: "
      f"{piv[_T].max() - piv[_T].min():.4f}")
print()

# Matched, over the five standard modality sets: the mean is the like-for-like
# statement, the max is what a "best cell" headline reports, and the spread is
# how much room the max had to be lucky in.
_all = (frame[(frame.task == _T) & frame.modset.isin(STD_MODSETS)]
        .groupby(["model", "modset"]).cv.mean().unstack()
        .reindex(["locusfusion", "mdcnn", "cisfusion", "late_fusion"])
        .dropna(how="all"))
print(f"{'model':<14}{'mean of 5':>11}{'max of 5':>10}{'spread':>9}"
      f"   paired against locusfusion")
for mdl, row in _all.iterrows():
    wins = ("" if mdl == "locusfusion" else
            f"   locusfusion ahead on {int((_all.loc['locusfusion'] > row).sum())}"
            f"/{int(row.notna().sum())} cells")
    print(f"{mdl:<14}{row.mean():>11.4f}{row.max():>10.4f}"
          f"{row.max() - row.min():>9.4f}{wins}")
print()
print("the max-of-5 ordering and the mean-of-5 ordering disagree; the mean is the")
print("matched one. See the note in the text.")
''')

md("""
Three things, and they are the payoff for Parts II and III.

**1. It gets from DNA alone what the Part I models need all four modalities to
reach.** Its spread across the five modality sets at 19 loci is about a
thousandth of AUC — inside noise. The modalities have not stopped mattering;
they have been *absorbed into the token*, which already carries the substituted
residue and its biophysical consequence as features. That is `additive`'s
featurisation result (§10) showing up as an architecture property.

**2. The locus count is what moves it**, and by roughly twice what it moves the
Part I models (§5). That follows from the representation: more loci means more
*tokens*, where for a dense encoder it mostly means more constants.

**3. Matched cell by cell it is ahead of every established architecture**, at a
tenth of `mdcnn`'s parameters and ~1% of `cisfusion`'s — and unlike them, its
attribution is exact and names a locus and a column rather than needing SHAP.

### Why §1 and §2 appear to say the opposite

They report each model's **best** cell, which is a max over five modality sets,
and that rule rewards variance rather than quality. `locusfusion` moves 0.0012
across the five; `mdcnn` moves 0.0075 and `cisfusion` 0.0129. So `mdcnn` gets
five draws from a distribution six times wider, and its *maximum* comes out
ahead (0.9246 vs 0.9242) even though its *typical* cell is behind. Pair on the
modality set and the ordering reverses.

The uncomfortable part is that the flatness in the left panel — the finding —
is exactly what costs `locusfusion` the max-selection in §1. A "best cell"
headline is not neutral between a stable model and an unstable one. Read the
right panel, not the §1 tiles, for which architecture is actually better.

### The honest limits

- **+0.0022 on the mean is inside noise.** Fold SD is 0.0172 and this is one
  seed. *Matches, at 1–10% of the parameters* is the claim that survives;
  *beats* is not, and multi-seeding is what would settle it.
- **On the joint task it loses**, and not narrowly: 0.9118 against
  `late_fusion`'s 0.9219. Everything above is the single-drug task.
- **Held-out test AUC is a tie** — 0.9101 against `mdcnn`'s 0.9109, averaged
  over the same five cells.
- **Two drugs go the other way and they go together**: AMIKACIN (−0.014) and
  KANAMYCIN (−0.020) at `all_modalities`, both *rrs* drugs. A variant tokenizer
  over 16S rRNA is the case worth looking at next.
- The aggregators have **no joint arm at all**, so nothing in Part III can be
  quoted in a claim about multi-task sharing.
""")

# ==================================================================
# ============================ PART V ==============================
# ==================================================================
md("""
---
---

# Part V — Coverage, the full table, and the caveats

## 13. What is done and what is not

Grey is complete, amber is short of 11 drugs, blank is not run. Columns are
grouped by family: established first, then the experimental aggregators,
`locusfusion`, and the non-network baselines.
""")

code('''
cov = (frame.groupby(["task", "modset", "model"]).drug.nunique()
       .rename("drugs").reset_index())
piv = cov.pivot_table(index=["task", "modset"], columns="model", values="drugs")
_order = ([m for m in sorted(ESTABLISHED) if m in piv.columns]
          + [m for m in AGGREGATORS if m in piv.columns]
          + [m for m in ["locusfusion"] if m in piv.columns]
          + [m for m in sorted(BASELINE_MODELS) if m in piv.columns])
piv = piv.reindex(columns=_order)
piv = piv.reindex(index=[(t, m) for t in TASKS for m in meta["MODSETS"]
                         if (t, m) in piv.index])


def _cell(v):
    if pd.isna(v):
        return f"background-color:{SURFACE}; color:{MUTED}"
    if v == 11:
        return f"background-color:{PALE}; color:{INK}"
    return f"background-color:#fdf0d0; color:{INK}"          # pale amber, dark ink


FAM_TINT = {"established": PALE_BLUE, "experimental": "#fbdcce",
            "baseline": "#cdf0e2"}
# The family grouping goes in the column headers. It has to be a table-style
# selector rather than a Styler.map: map paints <td>, and a <th> painted by an
# apply is overwritten by the cell styler that runs after it.
_head = [{"selector": f"th.col_heading.level0.col{i}",
          "props": f"background-color:{FAM_TINT[family(c)]}; color:{INK};"
                   "font-weight:600; font-size:11px; text-align:center"}
         for i, c in enumerate(piv.columns)]

display(piv.style.format("{:.0f}", na_rep="").map(_cell)
        .set_caption("drugs completed per cell (11 = complete) — "
                     "blue header = established, orange = experimental, green = baseline")
        .set_table_styles([
            {"selector": "caption",
             "props": f"caption-side:top; font-size:11px; color:{INK2};"
                      "text-align:left; padding-bottom:6px"},
            {"selector": "th",
             "props": f"background-color:{SURFACE}; color:{INK};"
                      "font-weight:600; font-size:11px; text-align:left"},
            {"selector": "td", "props": "font-size:11px; text-align:center"},
        ] + _head))
''')

md("""
The gaps that matter, and what they forbid:

- **No joint arm for any of the six aggregators.** `locusfusion` and all three
  established CNNs have one; the aggregators do not. Nothing in Part III can be
  in a multi-task claim.
- **`noisyor` has no per-drug-loci number.** All 11 rerun jobs reported
  `COMPLETED` in under four minutes and wrote nothing, dying on a CUDA illegal
  memory access inside the tokenizer's `topk` that reproduces **clean on CPU**
  and ran clean for the same model at 19 loci. It reads as node-specific rather
  than a model bug. Separately: a job that crashes should not exit 0, and the
  sbatch wrapper swallowed 11 tracebacks into clean exit codes.
- **`residual_catalogue` was never run** — the `additive` variant that adds the
  exact-identity table on top of the featurised weight, i.e. memorise what you
  have seen and featurise what you have not. Its own docstring calls it "likely
  the strongest single model in this file." Either run it or delete the claim.

## 14. The full table
""")

code('''
tab = (best[["family", "task", "model", "modset", "loci", "cv", "params"]]
       .sort_values(["family", "task", "cv"], ascending=[True, True, False]).copy())
tab["vs baseline"] = tab.apply(
    lambda r: r.cv - (SD_BASE if r.task == "single-drug" else MD_BASE), axis=1)
tab["params"] = tab.params / 1e6

display(tab.style
        .format({"cv": "{:.4f}", "vs baseline": "{:+.4f}",
                 "params": "{:.3f}M", "loci": "{:.0f}"}, na_rep="—")
        .map(lambda v: f"background-color:{FAM_TINT[v]}; color:{INK}", subset=["family"])
        .hide(axis="index")
        .set_caption("every complete cell (11/11 drugs), by family then task, best first")
        .set_table_styles([
            {"selector": "caption",
             "props": f"caption-side:top; font-size:11px; color:{INK2};"
                      "text-align:left; padding-bottom:6px"},
            {"selector": "th",
             "props": f"background-color:{PALE}; color:{INK};"
                      "font-weight:600; font-size:11px; text-align:left"},
            {"selector": "td",
             "props": f"color:{INK}; font-size:11px; padding:3px 10px"},
        ]))
''')

# ============================================================== caveats
md("""
---
## How to argue with these numbers

Every one of these is a reason a figure above could be wrong, and they are here
because a reviewer will find them anyway.

- **Single seed, everywhere.** Fold SD runs 0.003–0.037 by drug. **Differences
  under ~0.01 are unresolved**, which includes the gap between the best
  established cell and the best experimental one in §1 and §12. Multi-seeding
  the headline cells is a prerequisite for reporting any of this.
- **The two families are not the same experiment.** Part I reads dense sequence;
  Parts III–IV read reference-difference input. Where they appear on one axis
  (§11, §12) that is stated; they are never silently pooled into one ranking.
- **`single-drug, all loci` cannot be quoted against SD-CNN.** That baseline
  sees the per-drug gene map, so it is a different input. It is input-matched to
  MD-CNN only.
- **A "best cell" is optimistic, and unevenly so.** §1, §2 and §3 select on the
  same CV they report, so they are an upper bound rather than a model you could
  have chosen in advance — and the size of that bias scales with how much the
  model varies across the cells being maximised over. `cisfusion` varies 0.0129
  across modality sets and `locusfusion` 0.0012, so the same rule flatters them
  very differently. Any head-to-head has to be paired on the cell, as §12 does.
- **Held-out test AUC is favourably selected** — best of 5 folds by validation
  AUC, where BIG-TB uses fold 4 unconditionally. Judge on CV; test is reported
  only because it is what the papers publish.
- **SHAP attribution share mis-ranks predictive value** (§7), which is why Part
  II leans on the *positional* concentration and not on the modality shares. The
  per-column table is also truncated per block, so three drugs in that figure are
  lower bounds rather than counts.
- **The variant-token family requires the reference-difference encoding**, so it
  cannot see anything constant across the cohort. That carries no discriminative
  signal by construction, but it is a real restriction rather than a free lunch.
- **`noisyor`'s monotone constraint is a real cost, not just a low bar.** Every
  factor is in (0,1), so it *cannot* learn a protective variant — a lineage
  marker correlating with susceptibility is invisible to it, and at n = 269 the
  resulting bias dominates (LEVOFLOXACIN 0.610). Its first version also had the
  polarity inverted against this project's R=0/S=1 convention and came out
  **below chance at 0.4956**; that was caught by reading the training curve, not
  the AUC. Pre-fix results are archived under
  `results/archive/noisyor_polarity_bug_20260825/`.
- **Lineage confounding is not controlled anywhere.** *M. tuberculosis* lineages
  carry many neutral variants, and a model that learns lineage would look like a
  model that learns resistance on this split. The variant-token models make this
  more directly checkable, not less — their attention names the locus and column
  it read — but it has not been checked yet.
- **`setfusion` and the 150-epoch runs are excluded** from every figure.
  setfusion's tokens are near-collinear at initialization so its numbers are a
  lower bound rather than an architecture verdict, and `full_run` /
  `alllocus_run` ran at a schedule where the epoch cap was binding on 40% of
  folds.
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

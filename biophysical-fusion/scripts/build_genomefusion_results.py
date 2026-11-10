"""Builds notebooks/genomefusion/genomefusion_results.ipynb -- what the finished
GenomeFusion arms say about the features they were built to test.

    python scripts/build_genomefusion_results.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/genomefusion/genomefusion_results.ipynb

Reads only what is on disk: the result JSONs under results/experiments/genomefusion/
(via genomefusion.results), the locus_sets frame (via its collect.py) and the
lineage split files. It trains nothing. The markdown read-outs are written by hand
and dated, so re-read them after a refresh.

Resistant counts come from the split files, NOT from `n_R` in the GenomeFusion
JSONs. GenomeFusion encodes R=1/S=0 while the shared metric code counts y==0 as
`n_R`, so those two keys are swapped in its results (AUC is unaffected).

The longer version of this notebook, with the training-trace and per-baseline
figures, is in notebooks/genomefusion/superseded/.
"""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "notebooks" / "genomefusion" / "genomefusion_results.ipynb"

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "id": f"cell-{len(cells)}",
                  "source": t.strip("\n").splitlines(keepends=True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {"jupyter": {"source_hidden": True}},
                  "id": f"cell-{len(cells)}",
                  "execution_count": None, "outputs": [],
                  "source": t.strip("\n").splitlines(keepends=True)})


# ---------------------------------------------------------------- title + setup
md(r"""
# GenomeFusion results

GenomeFusion is LocusFusion over the whole H37Rv genome. All arms use the same 55 lineage
folds as the baselines (11 drugs x 5 folds). Results as of 2026-11-06, when the whole-genome
arm finished all 11 drugs.

| Question | Arms | Result |
|---|---|---|
| Does the new machinery reproduce LocusFusion? | `gf19p`, `gf19r` vs LocusFusion v5 | Yes, a tie |
| Do promoter bases add signal? | `gf19p` vs `gf19` | Yes, HL +0.008 |
| Do WHO regulatory annotations add signal beyond the promoter window? | `gf19r` vs `gf19p` | No detectable gain |
| Does the rest of the genome help? | whole genome vs `gf19r` | No, HL -0.040; lower on 45/55 folds |
""")

code(r"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from IPython.display import display
from scipy.stats import spearmanr

PROJECT = Path.cwd()
while not (PROJECT / "results_frame.py").exists():
    PROJECT = PROJECT.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "results" / "experiments" / "locus_sets"))

import collect                                    # the locus_sets + lineage_split frame
import paper_style
from paper_style import MODEL, NEUTRAL, ramp
from genomefusion.results import load as gf_load, load_other_19_locus, fold_provenance
from results_helpers import WilcoxonComparisons, fold_matrix
from bigtb_baselines import ALL_DRUGS

paper_style.apply()
%matplotlib inline
plt.rcParams["figure.dpi"] = 220     # the inline default is too coarse for these figures

G = gf_load()
F = pd.concat([collect.load(), G, load_other_19_locus()], ignore_index=True)
EXP = PROJECT / "results" / "experiments"
GF = EXP / "genomefusion"

# The LocusFusion family shares copper and is separated by mark shape; see paper_style.
COPPER, BLUE, TEAL = MODEL["locusfusion"], MODEL["cnn"], MODEL["lasso"]
GF_T = ("genomefusion", "gf_token")
ARM = {  # key: (label, task, model, modset, colour, marker, filled)
    "genome": ("GenomeFusion, whole genome", "single:size6776", *GF_T, COPPER, "X", True),
    "gf19r":  ("GenomeFusion, 19 + WHO regulatory", "single:gf19r", *GF_T, COPPER, "^", True),
    "gf19p":  ("GenomeFusion, 19 + promoters", "single:gf19p", *GF_T, COPPER, "o", True),
    "gf19":   ("GenomeFusion, 19, no promoters", "single:gf19", *GF_T, COPPER, "v", True),
    "v5":     ("LocusFusion v5", "single:bigtb19", "locusfusion_v5", "v5_fixed", COPPER, "D", False),
    "cnn":    ("stacked CNN (DNA + regulatory)", "single:bigtb19", "mdcnn", "dna_regulatory", BLUE, "s", True),
    "lasso":  ("LASSO (one-hot)", "single:bigtb19", "lasso", "dna", TEAL, "D", True),
}


def key(arm):
    return ARM[arm][1:4]


def folds(arm):
    return fold_matrix(F, ALL_DRUGS, *key(arm))


def mark(ax, x, y, arm, s=62, alpha=1.0, label=None):
    _, _, _, _, colour, marker, filled = ARM[arm]
    ax.scatter(x, y, s=s, marker=marker, alpha=alpha, zorder=3, label=label,
               facecolors=colour if filled else "none",
               edgecolors=NEUTRAL["surface"] if filled else colour,
               linewidths=1.0 if filled else 1.5)


def handle(arm, ms=8):
    _, _, _, _, colour, marker, filled = ARM[arm]
    return Line2D([], [], ls="", marker=marker, ms=ms, label=ARM[arm][0],
                  mfc=colour if filled else "none", mec=colour, mew=1.0 if filled else 1.5)


FOLD_KEY = [Line2D([], [], ls="", marker="o", ms=4.5, alpha=0.45, color=NEUTRAL["ink2"], label="one fold"),
            Line2D([], [], ls="", marker="o", ms=9, color=NEUTRAL["ink2"], label="mean of 5 folds")]


def clean(ax, axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def run(arm_dir, drug):
    return json.loads((GF / arm_dir / f"{drug}.json").read_text())


def fmt_p(p):
    return "p<0.0001" if p < 1e-4 else f"p={p:.2g}"


def resistant_count(drug):
    # development rows only; the split file's R column is 1 for resistant
    s = pd.read_csv(EXP / "lineage_split" / "splits" / f"{drug}.csv")
    return int(s.loc[s.role == "train", "R"].sum())


M = {a: folds(a) for a in ARM}
N_R = pd.Series({d: resistant_count(d) for d in ALL_DRUGS})
assert all(run("size6776", d)["complete"] and len(run("size6776", d)["cv_folds"]) == 5 for d in ALL_DRUGS)
BY_SCARCITY = N_R.sort_values().index.tolist()
print("macro CV AUC: " + ", ".join(f"{a} {M[a].values.mean():.4f}" for a in ARM))
""")

# ---------------------------------------------------------------- 1. contrasts
md(r"""
## The planned contrasts

Each row is one feature added or one comparison the arms were built for: two-sided exact
Wilcoxon over the 55 fold-paired differences, with the Hodges-Lehmann (HL) shift and its exact
95% CI, Holm-adjusted across the 8 rows.
""")

code(r"""
CONTRASTS = [("gf19p", "gf19", "+ promoter bases\n(gf19p - gf19)"),
             ("gf19r", "gf19p", "+ WHO regulatory annotation\n(gf19r - gf19p)"),
             ("gf19p", "v5", "gf19p - LocusFusion v5"),
             ("gf19r", "v5", "gf19r - LocusFusion v5"),
             ("genome", "gf19r", "+ rest of the genome\n(whole genome - gf19r)"),
             ("genome", "v5", "whole genome - LocusFusion v5"),
             ("genome", "cnn", "whole genome - stacked CNN"),
             ("genome", "lasso", "whole genome - LASSO")]
W = WilcoxonComparisons(F, ALL_DRUGS)
for a, b, name in CONTRASTS:
    W.folds(name, "", key(a), key(b))
T = W.table()
T["folds_up"] = [int(((M[a] - M[b]) > 0).values.sum()) for a, b, _ in CONTRASTS]

fig, ax = plt.subplots(figsize=(8.8, 5.6))
clean(ax)
ax.axvline(0, color=NEUTRAL["muted"], lw=1.1, zorder=1)
for y, r in enumerate(T.itertuples()):
    ax.hlines(y, r.lo, r.hi, color=NEUTRAL["ink2"], lw=2)
    ax.scatter([r.hl], [y], s=58, color=NEUTRAL["ink"], edgecolors=NEUTRAL["surface"], linewidths=1, zorder=3)
    ax.text(1.02, y, f"{r.hl:+.4f}   {r.folds_up}/55 up   Holm {fmt_p(r.p_holm)}", va="center", fontsize=9,
            transform=ax.get_yaxis_transform())
ax.set_yticks(range(len(T)), T.group)
ax.set_ylim(len(T) - 0.5, -0.5)
ax.set_xlabel("difference in CV AUC, 55 paired folds")
ax.set_title("Paired fold differences in CV AUC by contrast, 11 drugs, lineage split", loc="left")
ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=7, color=NEUTRAL["ink"], label="Hodges-Lehmann shift"),
                   Line2D([], [], color=NEUTRAL["ink2"], lw=2, label="exact 95% CI"),
                   Line2D([], [], color=NEUTRAL["muted"], lw=1.1, label="zero difference")],
          loc="upper center", bbox_to_anchor=(0.45, -0.11), ncol=3, fontsize=9)
fig.tight_layout()
plt.show()
print(T[["group", "delta", "hl", "lo", "hi", "folds_up", "p", "p_holm"]]
      .assign(group=T.group.str.replace("\n", " ")).round(5).to_string(index=False))
""")

md(r"""
- **Parity holds.** Against LocusFusion v5, `gf19p` is HL -0.0007 and `gf19r` HL +0.0009, both
  Holm p=1. The new machinery is not the reason for any whole-genome gap.
- **Promoter bases add signal:** HL +0.0079 (CI +0.0040 to +0.0126), Holm p=0.0003.
- **WHO regulatory annotation adds nothing detectable** beyond the promoter window: HL +0.0004
  (CI -0.0022 to +0.0029).
- **The whole genome scores lower:** macro CV 0.8491 against 0.9146 for `gf19r`, HL -0.0403
  (CI -0.0743 to -0.0261), lower on 45/55 folds. It also falls below the 19-locus stacked CNN
  (HL -0.0315) and LASSO (HL -0.0249). The mean difference is larger, -0.0655, because of two drugs
  shown next.
""")

# ---------------------------------------------------------------- 2. per drug
md(r"""
## Whole genome by drug

The whole-genome arm against its feature-matched 19-locus arm (`gf19r`) and LocusFusion v5.
Drugs are ordered by resistant isolates in the development set, scarcest first.
""")

code(r"""
SERIES = ["genome", "gf19r", "v5"]
y = np.arange(len(BY_SCARCITY))
fig, ax = plt.subplots(figsize=(8.8, 6.2))
clean(ax)
for i, d in enumerate(BY_SCARCITY):
    ax.hlines(i, M["genome"].loc[d].mean(), M["gf19r"].loc[d].mean(), color=NEUTRAL["axis"], lw=2.5, zorder=1)
    mark(ax, M["genome"].loc[d].values, [i] * 5, "genome", s=24, alpha=0.4)
    mark(ax, M["gf19r"].loc[d].values, [i] * 5, "gf19r", s=20, alpha=0.4)
for arm in SERIES:
    mark(ax, M[arm].loc[BY_SCARCITY].mean(axis=1).values, y, arm, s=105 if arm == "genome" else 72)
ax.set_yticks(y, [f"{d.title()} (R={N_R[d]:,})" for d in BY_SCARCITY])
ax.set_ylim(len(y) - 0.4, -0.6)
ax.set_xlabel("CV AUC")
ax.set_title("CV AUC by drug and arm, drugs ordered by resistant isolates, lineage split", loc="left")
ax.legend(handles=[handle(a) for a in SERIES] + FOLD_KEY, loc="upper center",
          bbox_to_anchor=(0.45, -0.1), ncol=3, fontsize=9)
fig.tight_layout()
plt.show()

gap = (M["genome"] - M["gf19r"]).mean(axis=1)
rho = spearmanr(N_R.reindex(gap.index), gap)
floored = {d: np.mean([max(f["history"]["val_auc"][20:]) for f in run("size6776", d)["cv_folds"]])
           for d in ALL_DRUGS}
TAB = pd.DataFrame({"R isolates": N_R,
                    "whole genome": M["genome"].mean(axis=1),
                    "gf19r": M["gf19r"].mean(axis=1),
                    "LF v5": M["v5"].mean(axis=1),
                    "genome - gf19r": gap,
                    "folds lower /5": (M["genome"] < M["gf19r"]).sum(axis=1),
                    "genome, epoch >= 21": pd.Series(floored)}).loc[BY_SCARCITY]
display(TAB.round(3))
print(f"gap vs resistant isolates: Spearman rho={rho.statistic:.3f}, p={rho.pvalue:.4f}, n=11 drugs")
print(f"whole genome macro CV {TAB['whole genome'].mean():.4f}; "
      f"checkpoints restricted to epoch >= 21: {TAB['genome, epoch >= 21'].mean():.4f}")
print(f"GPU hours, 11 drugs: whole genome {sum(run('size6776', d)['seconds'] for d in ALL_DRUGS) / 3600:.1f}, "
      f"gf19r {sum(run('gf19r', d)['seconds'] for d in ALL_DRUGS) / 3600:.1f}")
""")

md(r"""
- **The gap tracks resistant isolates:** Spearman rho=0.91, p=0.0001 across the 11 drugs. This was
  not a planned test and it cannot separate scarcity from drug class.
- **The fluoroquinolones drive the mean:** LEVO -0.211 and MOXI -0.196. Across the other nine the
  mean gap is -0.035.
- **The two best-sampled drugs nearly close it:** RIF +0.001 (lower on 1/5 folds), INH -0.007.
- **Cost:** 102.6 GPU-hours against 4.1 for `gf19r`, about 25 times.
- **Checkpoint selection slightly flatters the arm.** Selecting from epoch 21 on takes the macro from
  0.8491 to 0.8424, with most of the change on LEVO (0.713 to 0.677), CAP (0.811 to 0.798) and MOXI
  (0.695 to 0.680). So the gap is if anything understated.
""")

# ---------------------------------------------------------------- 3. features per drug
md(r"""
## Promoter and regulatory effects by drug

Where each 19-locus feature moved CV AUC. Drugs are ordered by the promoter effect. No per-drug
test is reported: with 5 folds the exact two-sided p cannot go below 0.0625.
""")

code(r"""
PANELS = [("gf19p", "gf19", "gf19p - gf19 (promoter bases)"),
          ("gf19r", "gf19p", "gf19r - gf19p (WHO regulatory annotation)")]
order = (M["gf19p"] - M["gf19"]).mean(axis=1).sort_values(ascending=False).index
y = np.arange(len(order))
fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.2), sharey=True)
for ax, (a, b, name) in zip(axes, PANELS):
    clean(ax)
    d_ = (M[a] - M[b]).loc[order]
    ax.axvline(0, color=NEUTRAL["muted"], lw=1.1, zorder=1)
    for i, drug in enumerate(order):
        mark(ax, d_.loc[drug].values, [i] * 5, a, s=22, alpha=0.4)
    mark(ax, d_.mean(axis=1).values, y, a, s=80)
    for i, v in enumerate(d_.mean(axis=1).values):
        ax.text(1.01, i, f"{v + 0.0:+.3f}".replace("-0.000", "0.000").replace("+0.000", "0.000"), va="center",
                fontsize=8.5, transform=ax.get_yaxis_transform())
    ax.set_xlabel("difference in CV AUC")
    ax.set_title(name, loc="left", fontsize=10)
axes[0].set_yticks(y, [d.title() for d in order])
axes[0].set_ylim(len(y) - 0.4, -0.6)
fig.legend(handles=FOLD_KEY + [Line2D([], [], color=NEUTRAL["muted"], lw=1.1, label="zero difference")],
           loc="lower center", ncol=3, fontsize=9)
fig.suptitle("Fold differences in CV AUC by drug and feature, 19-locus arms, lineage split",
             x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0.06, 1, 1))
plt.show()
""")

md(r"""
- **Promoter bases help most where a promoter mechanism is catalogued:** ETO +0.051 (5/5 folds up, as
  predicted from fabG1 c-15t), KAN +0.017 (*eis*), INH +0.016 (c-15t). CAP (+0.016) and LEVO (+0.014)
  also rise, with no such prediction made, and LEVO's folds range from -0.023 to +0.050.
- **The regulatory annotation has no clear per-drug effect.** ETO +0.010 and KAN +0.008 are the
  largest gains; LEVO's -0.032 sits on four folds of 17-23 isolates.
""")

md(r"""
### Ethionamide by locus set

The only drug with WHO tier 1+2 and lab64 GenomeFusion runs, chosen because promoters matter
most for it. All four arms carry regulated-locus features.
""")

code(r"""
ETO = "ETHIONAMIDE"
rows = [("WHO tier 1+2 + regulatory", "gf_who_t12r"), ("19 + WHO regulatory", "gf19r"),
        ("lab64 + regulatory", "gf_lab64r"), ("whole genome", "size6776")]
eto = pd.DataFrame([{"locus set": name,
                     "loci": run(a, ETO)["n_loci"],
                     "active loci per isolate": round(run(a, ETO)["active_loci_mean"], 1),
                     "CV AUC": run(a, ETO)["cv_auc_mean"],
                     "folds below LF v5 /5": int(sum(f["auc"] < v for f, v in
                                                     zip(run(a, ETO)["cv_folds"], M["v5"].loc[ETO].values)))}
                    for name, a in rows])
display(eto.round(3))
print(f"LocusFusion v5 on the same folds: {M['v5'].loc[ETO].mean():.3f}")
""")

md(r"""
Sets of 17, 42 and 148 loci all land within 0.012 of each other and of v5, while the whole genome is
0.056 below v5 and lower on all 5 folds. So on this drug extra curated loci cost little and the loss
appears somewhere between 148 and 6,776 loci; no run yet places it, and these sets also differ in content.
""")

# ---------------------------------------------------------------- 4. dynamics
md(r"""
## Training dynamics

Validation AUC by epoch, averaged over each drug's 5 folds, drugs ordered as in the by-drug
figure. Every fold ran at least 79 epochs. Ticks mark the median epoch the checkpoints came from.
""")

code(r"""
EPOCHS = 79
cmap = LinearSegmentedColormap.from_list("copper_seq", ramp(COPPER, 6)[::-1])  # light = low, dark = high
PANELS = [("size6776", ARM["genome"][0]), ("gf19r", ARM["gf19r"][0])]


def curve(arm, drug):
    return np.mean([f["history"]["val_auc"][:EPOCHS] for f in run(arm, drug)["cv_folds"]], axis=0)


fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4), sharey=True, layout="constrained")
for ax, (arm, name) in zip(axes, PANELS):
    H = np.array([curve(arm, d) for d in BY_SCARCITY])
    im = ax.imshow(H, aspect="auto", cmap=cmap, vmin=0.5, vmax=1.0, interpolation="nearest",
                   extent=(0.5, EPOCHS + 0.5, len(BY_SCARCITY) - 0.5, -0.5))
    best = [np.median([f["best_epoch"] for f in run(arm, d)["cv_folds"]]) for d in BY_SCARCITY]
    ax.scatter(best, range(len(best)), marker="|", s=110, lw=2, color=NEUTRAL["ink"], zorder=3)
    ax.set_xlabel("epoch")
    ax.set_title(name, loc="left", fontsize=10)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0].set_yticks(range(len(BY_SCARCITY)), [f"{d.title()} (R={N_R[d]:,})" for d in BY_SCARCITY])
fig.colorbar(im, ax=axes, label="validation AUC, mean of 5 folds", fraction=0.025, pad=0.02)
fig.legend(handles=[Line2D([], [], ls="", marker="|", ms=11, mew=2, color=NEUTRAL["ink"],
                           label="median checkpoint epoch")],
           loc="outside lower center", fontsize=9)
fig.suptitle("Validation AUC by epoch and drug, whole-genome and gf19r arms", x=0.01, ha="left", fontsize=12)
plt.show()


def first_reach(arm, drug, target):
    hit = np.nonzero(curve(arm, drug) >= target)[0]
    return int(hit[0]) + 1 if len(hit) else None


DYN = pd.DataFrame({"gf19r CV": M["gf19r"].mean(axis=1),
                    "whole genome: first epoch at 90% of gf19r CV":
                        {d: first_reach("size6776", d, 0.9 * M["gf19r"].loc[d].mean()) for d in ALL_DRUGS},
                    "gf19r: same target":
                        {d: first_reach("gf19r", d, 0.9 * M["gf19r"].loc[d].mean()) for d in ALL_DRUGS},
                    "whole genome final train loss":
                        {d: np.mean([f["history"]["train_loss"][-1] for f in run("size6776", d)["cv_folds"]])
                         for d in ALL_DRUGS},
                    "gf19r final train loss":
                        {d: np.mean([f["history"]["train_loss"][-1] for f in run("gf19r", d)["cv_folds"]])
                         for d in ALL_DRUGS}}).loc[BY_SCARCITY]
display(DYN.round(3))
""")

md(r"""
- **`gf19r` is near its final level almost immediately:** it reaches 90% of its CV AUC at epoch 1 on
  9/11 drugs (MOXI epoch 4, LEVO epoch 41).
- **The whole-genome warmup is general and varies widely:** RIF reaches the same target at epoch 6,
  EMB 12, INH 14, PZA 16, AMK 20, STR 30, ETO 35, KAN 46. LEVO, MOXI and CAP never reach it in the
  first 79 epochs. So `min_epochs` must stay at 50, and median checkpoints as late as epoch 48 (CAP)
  bear that out.
- **On the scarce drugs training continues without validation following:** LEVO ends at train loss
  0.157 and MOXI at 0.105 while their validation AUC stays near 0.7. That looks like overfitting, but
  is untested.
""")

# ---------------------------------------------------------------- 5. context
md(r"""
<a id="Context"></a>
## Context

GenomeFusion partitions H37Rv into 6,776 loci, one locus per base (3,749 genes + 3,027 intergenic
segments), with no 16-variant cap, and feeds the network only the loci an isolate changes.

| arm | locus universe | loci | regulated-locus features | active loci / isolate |
|---|---|---|---|---|
| `gf19` | GenomeFusion's 19 genes (*ahpC*, not *tlyA*), no promoter bases | 19 | no | ~5 |
| `gf19p` | the 19 + intergenic loci covering their 100 bp 5' flank (parity arm) | 35 | no | ~7 |
| `gf19r` | the 19 + loci touched by WHO regulatory regions | 42 | yes | ~8 |
| `size6776` | the whole genome | 6,776 | yes | ~707 |

LocusFusion v5 is the fair baseline: the corrected MSAs and the start-codon WHO promoter windows,
so a GenomeFusion gain over it cannot come from those data fixes. Its 19 genes differ from
GenomeFusion's by one, as the caveats note. The stacked
CNN and LASSO are shown at their strongest 19-locus cells.

**Fold pairing.** Every comparison is paired, so fold *i* must be the same held-out isolates in
every cell. Each run records a hash of its validation IDs:
""")

code(r"""
checked = cells_seen = 0
for d in ALL_DRUGS:
    prov = fold_provenance(d)
    hashed = prov[prov.shas.map(len) > 0]
    ref = next(r.shas for _, r in hashed.iterrows() if len(r.shas) == 5)
    bad = [r.cell for _, r in hashed.iterrows() if list(r.shas) != list(ref[:len(r.shas)])]
    assert not bad, f"{d}: folds do not pair: {bad}"
    checked += 1
    cells_seen += len(hashed)
protocols = {a: {json.dumps(run(a, d)["protocol"], sort_keys=True) for d in ALL_DRUGS}
             for a in ["size6776", "gf19r", "gf19p", "gf19"]}
assert len(set.union(*protocols.values())) == 1, "arms ran under different protocols"
print(f"{cells_seen} cells over {checked} drugs carry per-fold hashes, and each drug's cells agree.")
print("All four GenomeFusion arms share one training protocol: " + next(iter(protocols["gf19r"])))
""")

# ---------------------------------------------------------------- 6. caveats
md(r"""
## Caveats

- **The whole-genome gap has no attribution yet.** It contains the 19 loci, so its deficit is a cost of
  carrying the other loci, not evidence that they hold no signal. `gf_minus19p` (signal outside the 19)
  and the size ladder (cost against locus count) have not run. Do not use `gf_minus19`: it keeps
  the promoter loci, including c-15t.
- **Scarcity is a post-hoc reading over 11 drugs.** Resistant count is confounded with drug class and
  mechanism, and the two lowest-count drugs are both fluoroquinolones.
- **A non-significant row is not proof of equivalence.** No equivalence margin was set, so "tie" here
  means no detectable difference.
- **GenomeFusion's 19 genes are not LocusFusion's 19.** `LOCI_19` in `genomefusion/partition.py` has
  *ahpC* and not *tlyA*; LocusFusion v3, v5, the stacked CNN and LASSO have *tlyA* and not *ahpC*. So
  every parity row compares sets differing by one gene, the 19-locus arms lack capreomycin's *tlyA*,
  and `gf_minus19p` keeps *tlyA* while dropping *ahpC*. Found 2026-11-06; no run has fixed it.
- **`gf19r - gf19p` bundles two changes:** 7 more loci and the regulated-locus features. The
  `--no-regulated-features` control has not run. As of 2026-11-05 the fabG1 promoter locus links to
  fabG1 only, not inhA, so this arm may undersell c-15t's target.
- **The inputs differ between arms and baselines.** `gf19p` and `gf19r` are slight supersets of the
  19 genes, since loci are atomic. v5 has a per-drug distance-to-binding-site channel GenomeFusion lacks.
- **The protocols differ in batch size:** GenomeFusion 32, LocusFusion 128. Each cell is one seed.
- **Checkpoints are selected on noisy validation AUC from epoch 1.** This flatters CV on small folds:
  LEVO's four smallest hold 17-23 isolates.
- **The token is not a modality set,** so no pair here is modality-matched.
- **LASSO's fold pairing is weaker:** it has no per-fold hash and rests on the recorded split directory.
- **The cohort differs by one isolate** (17,943 vs 17,942). Validation hashes match, so the extra
  isolate is in training.
- **TEST AUC is left out.** Each arm scores TEST once from its best CV fold, so arms are not comparable
  unless they picked the same fold. Whole-genome LEVO's 0.445 comes from a near-untrained epoch-6 checkpoint.
- **Coverage is unknown in the input.** The VCFs hold variant sites only, so "no token" means "matches
  H37Rv or was not covered", which bites hardest genome-wide.
- **WHO tier 1+2 and lab64 cover ETO only.** The other 10 drugs have not run.
- **GenomeFusion result files mislabel the classes.** It encodes R=1, but the metric keys assume y=0 is
  resistant, so `n_R` and `n_S` are swapped and the `P_S_*` columns hold P(resistant). AUC and the class
  weights are unaffected. Counts here come from the split files; check `sens` and `spec` in those JSONs
  before using them.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT.relative_to(PROJECT)}  ({len(cells)} cells)")

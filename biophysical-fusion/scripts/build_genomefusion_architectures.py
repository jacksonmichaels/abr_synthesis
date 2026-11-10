"""Builds notebooks/genomefusion/genomefusion_architectures.ipynb -- the three
ways GenomeFusion can hold a regulatory-region -> gene relationship, six designs
further from LocusFusion, and the questions that need a biologist's answer.

    python scripts/build_genomefusion_architectures.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/genomefusion/genomefusion_architectures.ipynb

This is a DESIGN notebook, not a results notebook. It reads the finished
ETHIONAMIDE arms off disk and draws schematics; it trains nothing, loads no data
mount and needs no GPU, so it is safe on a login node. Plan C and the six further
designs are proposals: nothing said about them is a measurement.

The further designs' schematics come from scripts/genomefusion_design_figures.py
and their evidence from the CSVs that scripts/genomefusion_design_probes.py writes;
rerun that script (~2 min, in a compute allocation, not on a login node) to refresh.

Layout: figures first; lab questions, then plan detail under "Context" near the end.
Code cells are saved collapsed so the notebook reads as figures.
"""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "notebooks" / "genomefusion" / "genomefusion_architectures.ipynb"

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "id": f"cell-{len(cells)}",
                  "source": t.strip("\n").splitlines(keepends=True)})


def code(t):
    cells.append({"cell_type": "code", "metadata": {"jupyter": {"source_hidden": True}},
                  "id": f"cell-{len(cells)}", "execution_count": None,
                  "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})


# ---------------------------------------------------------------- title + setup
md(r"""
# GenomeFusion: the model and its limits

The model whose scores `genomefusion_results.ipynb` reports. Its figures use real data: the
corrected whole-genome partition, the token store the trainer reads, and one held-out isolate
followed through every step. Three plans for the promoter-gene link and six further designs follow.
""")

code(r"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from IPython.display import display

PROJECT = Path.cwd()
while not (PROJECT / "paper_style.py").exists() and PROJECT != PROJECT.parent:
    PROJECT = PROJECT.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import genomefusion_design_figures as DF
import genomefusion_method_figures as MF
import paper_style
from paper_style import MODEL, NEUTRAL

paper_style.apply()
%matplotlib inline
plt.rcParams["figure.dpi"] = 220     # the inline default is too coarse for these figures
# Plans and designs below read the pre-fix runs until the 2026-11-06 reruns finish.
GF = PROJECT / "results" / "experiments" / "genomefusion" / "superseded_ahpC_20261106"
FIGDIR = PROJECT / "results" / "figures" / "genomefusion"
FIGDIR.mkdir(parents=True, exist_ok=True)
COPPER = MODEL["locusfusion"]


def mark(ax, x, y, marker, s=62, alpha=1.0, filled=True, label=None, colour=COPPER):
    # filled marks carry a surface ring so they stay legible where series overlap
    ax.scatter(x, y, s=s, marker=marker, alpha=alpha, zorder=3, label=label,
               facecolors=colour if filled else "none",
               edgecolors=NEUTRAL["surface"] if filled else colour,
               linewidths=1.0 if filled else 1.5)


def clean(ax, axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


cohort = pd.read_csv(DF.PROBES / "cohort.csv").set_index("drug")
base = pd.read_csv(DF.PROBES / "baselines.csv", index_col=0)
""")

# ---------------------------------------------------------------- 0. the model in use
md(r"""
## From a whole genome to tokens

The partition (a), one isolate through every filter (b), where tokens fall on the partition (c),
and what a token holds (d). SAMEA2535332 is a resistant isolate held out of isoniazid training,
chosen for carrying each token kind, not for any prediction.
""")

code(r"""
T = MF.load()
MF.genome_to_tokens(T)
plt.show()
""")

md(r"""
- **A token is a difference from H37Rv.** A locus with no difference produces nothing, so this
  isolate's 607 tokens occupy 514 of 6,776 loci.
- **The filters remove more events than they keep:** of 1,468 events, 624 fall in masked bases and
  237 are synonymous, leaving 607.
- **A promoter is its own locus.** fabG1 c.-15C>T is a token in `Rv1482c-Rv1483`, a separate locus from
  *fabG1*, tied to it by an edge. That edge reaches *fabG1* but not *inhA*.
- **Channels:** M molecular weight, p isoelectric point, H hydrophobicity, Δ the change from H37Rv.
  Consequence: synonymous, missense, nonsense, frameshift, in-frame indel, start loss, stop loss,
  unreadable. tr truncation, c/r/i coding/RNA/intergenic, up exp(-d/50 bp) upstream of a start
  codon, sp log1p(span), o/h/i overlap/homopolymer/imprecise, ∅ `is_null`. Coord is the base offset
  from the locus's lower genome coordinate.
""")

md(r"""
### Where regulatory relationships come from

Part (c) above in full: the sources (a), every decision around *fabG1*-*inhA* (b) and *katG* (c), the
rule across the genome (d), and how the model uses an edge (e).
""")

code(r"""
R = json.loads((MF.FIG / "regulatory_trace.json").read_text())
MF.regulatory_edges(T, R)
plt.show()
""")

md(r"""
**Nothing here is decided per isolate.** WHO regions tag loci when the partition is built. Each locus's
single target is fixed when the model is built, and every forward pass only looks it up. What varies by
isolate is which loci are active, and each token's two upstream channels, which come from its position.

- **Steps, in order:** a locus takes the first alphabetical WHO name that resolves to another named locus;
  failing that, the next start codon on the + strand; then on the − strand. Each table row in (b) and (c)
  is this rule applied to one locus.
- **Most edges are not regulatory:** 74 of 4,419 come from a WHO name. The other 4,345 are the
  next-gene fallback, and 2,411 of those start from a coding gene.
- **The fallback is not always the neighbouring gene:** the + strand is tried first whatever the
  distance, so 445 intergenic loci skip a nearer − strand gene. `Rv3879c-Rv3880c` links to *mutT4*,
  33,250 bp away, rather than the adjacent *espK*.
- **Names decide what can be linked:** 1,924 of 3,680 coding genes have only an Rv tag and cannot be a
  target, so *katG* gets no edge of its own, and 13 WHO-named targets cannot resolve.
- **One target loses information:** c-15t's locus keeps *fabG1* and drops *inhA*; `Rv3854c-Rv3855` keeps
  *ethA* and drops *ethR*.
- **WHO bases are counted per row:** *fabG1* and *inhA* are one region listed twice, so loci in it count
  double. c-15t's locus shows 100 for a 50-base overlap, and this enters the `log_reg_bases` feature.
""")

md(r"""
## From tokens to the model

The same isolate's tensors at each step, with the real parameter count of each module. Only the
shapes depend on the isolate; the model holds one query, so each drug has its own model.

**What a pair is.** A pair is one isolate's tokens in one locus: the isolate has at least one token there.
This isolate has 607 tokens in 514 loci, so it has 514 pairs, and the other 6,262 loci have no pair and
never enter the model. In a batch, the same locus in two isolates makes two pairs. Each pair becomes
one row of stage 1: its [NULL] token followed by its tokens. Rows are grouped by token count so that each
group pads only to its own width. Four of this isolate's pairs:

| pair (locus) | tokens | `pair_n` | stage 1 tensor it joins | its row | stage 2 slot |
|---|---|---|---|---|---|
| *katG* | S315T | 1 | (452, 2, 128) | [NULL], S315T | 206 |
| `Rv1482c-Rv1483` | fabG1 c.-15C>T | 1 | (452, 2, 128) | [NULL], c.-15C>T | 434 |
| *ctpI* | one frameshift | 1 | (452, 2, 128) | [NULL], the token | 17 |
| `Rv0387c` | 9 changes | 9 | (1, 17, 128) | [NULL], 9 tokens, 7 masked | 51 |

Each pair comes out of stage 1 as one 128-value locus vector, read at its [NULL] position. Those 514
vectors, plus the reference token in slot 0, are the 515 slots of stage 2. Padding every isolate to every
locus would instead take 6,776 x 10 = 67,760 token slots for this isolate, against the 1,121 used here
(607 tokens + 514 [NULL]).
""")

code(r"""
MF.tokens_to_model(T)
plt.show()
""")

md(r"""
Stage 1 sees tokens only within a locus, and stage 2 sees one vector per locus, so two variants in
different loci interact only through those vectors. The per-locus residual is 56% of the parameters.
""")

md(r"""
### Trained values for this isolate

One forward pass of the corrected whole-genome isoniazid model, replayed step by step. Attention
shows what the model looked at for one isolate, and does not show what drives the prediction.
""")

code(r"""
trace = MF.FIG / "forward_trace.json"
if trace.exists():
    F = json.loads(trace.read_text())
    MF.forward_pass(T, F)
    plt.show()
    print(f"checkpoint: {F['weights']}")
    print(f"stage 2, layer 2, mean incoming: " + ", ".join(f"{r['slot']} {r['weight']:.3f}" for r in F["stage2"][1]["incoming_mean"][:5]))
    print(f"query: reference {F['query']['reference']:.4f}, maximum {F['query']['max']:.4f}, uniform {F['query']['uniform']:.4f}")
else:
    print("Not available yet: the corrected whole-genome isoniazid model has not finished training.")
""")

md(r"""
## Limits

Each limit as measured on this cohort, then how the token counts vary across it.
""")

code(r"""
C, G, F_, S_, E = T["cohort"], T["genome"], T["example"]["funnel"], T["funnel_sample"], T["example"]
tot = C["tokens_total"]
ETA2 = C["lineage_eta2"]
n_s = T["funnel_sample_n"]
limits = [
    ("Mask", f"{G['bases']['masked']:,} bp ({100 * G['meta']['masked_frac']:.1f}%): {G['meta']['rlc_regions']} low-confidence regions and {G['meta']['masked_category_genes']} PE/PPE and mobile-element genes; mean {C['masked_events_mean']:.0f} events per isolate discarded",
     f"Nothing in these bases can be seen. WHO tier 1+2 genes with no locus: {', '.join(g for v in T['who_t12_not_in_partition'].values() for g in v)} (pyrazinamide)."),
    ("Coverage", "not measurable: the VCFs hold variant sites only",
     "No token means matches H37Rv or was not covered, and the two cannot be separated."),
    ("Synonymous changes", f"dropped before training: median {S_['synonymous']['median']:.0f} per isolate (n = {n_s})",
     "Any signal they carry, including lineage, is absent. The `synonymous` channel is always zero."),
    ("Mixed and low-quality calls", f"median {S_['called_uncalled']['median']:.0f} uncalled records per isolate, {S_['intermediate_af']['median']:.0f} of them at intermediate allele frequency (n = {n_s}); {C['uncalled_tokens']:,} N tokens ({100 * C['uncalled_tokens'] / tot:.1f}%)",
     "Allele frequency 0.25-0.75 becomes N and at most 0.25 becomes reference, so heteroresistance is not represented as a frequency."),
    ("Insertions", f"{C['insertion_tokens']:,} tokens ({100 * C['insertion_tokens'] / tot:.1f}%) carry length and consequence only",
     "The inserted sequence is not in the token, and the base channel shows the reference base."),
    ("Imprecise and structural calls", f"IMPRECISE records left as reference (median {S_['imprecise']['median']:.0f} per isolate, n = {n_s}); symbolic ALT skipped; uncalled indels over 100 bp ignored",
     "The `imprecise` flag is therefore always zero, and symbolic structural variants produce no token."),
    ("Unused channels", f"{len(T['never_nonzero'])} of 57 never nonzero: {', '.join(T['never_nonzero'])}",
     "54 channels carry information. `is_null` is unused because [NULL] does not pass through `tok_proj`."),
    ("One locus per base", f"{G['overlap_bases']:,} bases lie in two genes; {C['overlap_tokens']:,} tokens carry the overlap flag",
     "Each such base goes to the gene with the nearer start codon, so the other gene does not see it."),
    ("Position", f"coord ignores strand: katG S315T has coord {E['tokens'][0]['coord']:,.0f}, not codon {E['tokens'][0]['codon'] + 1}",
     "On minus-strand genes coord runs 3' to 5'. The codon number is not a channel."),
    ("Promoter-to-gene link", f"{R['counts']['who']} of {G['n_edges']:,} edges come from a WHO name; the rest are the next-gene fallback, {R['intergenic_downstream']['over_500']} intergenic ones over 500 bp, {R['intergenic_plus_with_nearer_minus']['n']} skipping a nearer gene",
     "The regulated-locus features mostly describe a neighbouring gene, not known regulation. Each locus has one target, so c-15t's locus links to fabG1 and not inhA."),
    ("Unnamed genes", f"{R['unnamed_cds']:,} of {R['named_cds'] + R['unnamed_cds']:,} coding genes cannot be an edge target; {len(R['unresolved_names'])} WHO-named targets cannot resolve",
     f"{R['tagged_by_branch']['none']} of {R['n_regulatory']} WHO-tagged loci get no edge and {R['tagged_by_branch']['plus'] + R['tagged_by_branch']['minus']} fall back to the next gene."),
    ("Locus identity", f"locus_residual is {100 * T['params']['locus_residual_params'] / T['params']['total']:.0f}% of {T['params']['total']:,} parameters; {C['loci_active_lt10']} loci are active in under 10 isolates, {C['loci_never_active']} in none",
     "Rare loci's residuals get few training updates. Identity also includes an `is_19` feature, so the whole-genome arm has a curated prior and does not start blind."),
    ("Interaction", "stage 1 is within a locus; stage 2 sees one vector per locus",
     "No attention between individual variants in different loci."),
    ("Memory", f"stage 2 grows with the square of active loci: up to {C['active']['max']:,}. A budget of {C['batch']['budget']:,} slots puts {C['batch']['isolates_below_requested']:,} isolates ({100 * C['batch']['isolates_below_requested'] / C['n']:.1f}%) in batches below {C['batch']['requested']}, minimum {C['batch']['min_batch']}",
     f"Tokens are uncapped: up to {C['tokens']['max']:,} per isolate and {C['pair_tokens_max']:,} in one locus. A 19-locus model costs far less; see the GPU hours in the results notebook."),
    ("Empty isolates", f"{C['zero_token_isolates']} isolate has no tokens",
     "It has no lineage call either, and reaches the query through the reference token alone."),
    ("Population structure", f"no homoplasy or phylogeny channel; lineage explains {100 * ETA2:.0f}% of the variance in active-locus count",
     "Most of what an isolate presents tracks its ancestry relative to H37Rv, and lineage markers are ordinary tokens."),
]
display(pd.DataFrame(limits, columns=["Limit", "Measured", "What it means"]).style.hide(axis="index")
        .set_properties(**{"text-align": "left", "white-space": "normal", "vertical-align": "top"})
        .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}]))
""")

code(r"""
MF.cohort_limits(T)
plt.show()
""")

md(r"""
Median 752 tokens over 593 active loci, and 85.7% of (isolate, locus) pairs hold one token. The
clusters are lineage: top-level lineage explains 88% of the variance in active-locus count, with
medians of 535 in lineage 4 (H37Rv's own), 836 in lineage 3, 861 in lineage 2 and 1,167 in lineage 1.
Of the 12 busiest loci, only *gyrA* is one of the curated 19.
""")

md(r"""
> **The sections below still use the pre-fix runs,** whose 19 loci had *ahpC* instead of *tlyA*, and
> will be updated when the 2026-11-06 reruns finish. The figures above do not use those results.
""")

# ---------------------------------------------------------------- 1. the plans
md(r"""
## Three plans for the promoter-gene relationship

The path each plan gives a promoter variant from the genome to the prediction.
""")

code(r"""
PLANS = [
    ("A. anonymous distance (built)",
     ["c-15t base", "intergenic\nlocus 2841", "own slot in\nacross-loci stage", "drug query"],
     "exp(-15/50) on the + strand", "no link to fabG1"),
    ("B. feature concat + gated edge (built)",
     ["c-15t base", "intergenic\nlocus 2841", "own slot, embedding\n+ reg_gate x fabG1", "drug query"],
     "+ 29 fabG1 features", "link = 1 scalar"),
    ("C. declared units (proposal)",
     ["c-15t base", "fabG1 unit,\ncoord -15", "one slot for\nthe whole unit", "drug query"],
     "routed by the unit table", "link = a CSV row"),
]

fig, axes = plt.subplots(3, 1, figsize=(9.2, 6.0))
for ax, (name, boxes, above, below) in zip(axes, PLANS):
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.4); ax.axis("off")
    ax.text(0, 2.15, name, fontsize=10.5, fontweight="bold", color=NEUTRAL["ink"], va="top")
    xs = [0.1, 2.6, 5.4, 8.4]
    w = [2.1, 2.4, 2.7, 1.5]
    for i, (x, bw, label) in enumerate(zip(xs, w, boxes)):
        edge = COPPER if i in (1, 2) else NEUTRAL["axis"]
        ax.add_patch(FancyBboxPatch((x, 0.75), bw, 0.72, boxstyle="round,pad=0.06",
                                    linewidth=1.3, edgecolor=edge, facecolor="white"))
        ax.text(x + bw / 2, 1.11, label, ha="center", va="center", fontsize=8.6,
                color=NEUTRAL["ink"])
        if i < len(xs) - 1:
            ax.add_patch(FancyArrowPatch((x + bw + 0.05, 1.11), (xs[i + 1] - 0.05, 1.11),
                                         arrowstyle="-|>", mutation_scale=9,
                                         linewidth=1.0, color=NEUTRAL["ink2"]))
    ax.text(xs[1] + w[1] / 2, 1.80, above, ha="center", va="center", fontsize=8.0,
            color=NEUTRAL["ink2"], linespacing=1.35)
    ax.text(xs[2] + w[2] / 2, 0.42, below, ha="center", va="center", fontsize=8.5,
            color=NEUTRAL["ink"], fontweight="bold")

fig.suptitle("Path from the fabG1 c-15t promoter variant to the prediction, by plan",
             fontsize=11.5, y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.95))
for ext in ("pdf", "png"):
    fig.savefig(FIGDIR / f"fig_architectures.{ext}", dpi=200,
                facecolor="white", bbox_inches="tight")
plt.show()
""")

md(r"""
The choice is not only about score. The catalogued promoter-gene links are facts we
already hold, so it also matters which plan states them in a form a biologist can
audit: a distance (A), one global scalar (B) or a row in a table (C). Mechanics are
under [Context](#Context).

Evidence so far, on ETO, the maximum-effect drug: its canonical mechanism (*fabG1*
c-15t) is a promoter variant.
""")

code(r"""
ARMS = [("gf19", "A", "v"), ("gf19p", "A", "o"), ("gf19r", "B", "^"), ("size6776", "B", "X")]

fig, ax = plt.subplots(figsize=(8.8, 3.7))
clean(ax)
labels = []
for y, (folder, plan, marker) in enumerate(ARMS):
    d = json.loads((GF / folder / "ETHIONAMIDE.json").read_text())
    v = np.array([c["macro_auc"] for c in d["cv_folds"]])
    mark(ax, v, [y] * len(v), marker, s=28, alpha=0.45)
    mark(ax, [v.mean()], [y], marker, s=110)
    ax.text(v.mean(), y - 0.25, f"{v.mean():.3f}", ha="center", va="bottom", fontsize=9,
            bbox=dict(facecolor=NEUTRAL["surface"], edgecolor="none", pad=1), zorder=5)
    labels.append(f"{folder}, plan {plan}\n{d['n_loci']:,} loci, {d['active_loci_mean']:.1f} active, "
                  f"{d.get('n_regulated_edges') or 0:,} edges, {d['n_params'] / 1e6:.2f}M params")
v3 = base.loc["ETHIONAMIDE", "LF v3, 19 loci"]
ax.axvline(v3, color=COPPER, lw=1.2, ls="--", zorder=1)
ax.text(v3 + 0.003, len(ARMS) - 0.62, f"LocusFusion v3 {v3:.3f}", fontsize=9, va="center")
ax.set_yticks(range(len(ARMS)), labels, fontsize=8.8)
ax.set_ylim(len(ARMS) - 0.45, -0.65)
ax.set_xlabel("CV AUC (small marks: folds; large marks: mean of 5)")
ax.set_title("Ethionamide: CV AUC by GenomeFusion arm, lineage split", loc="left")
fig.tight_layout()
plt.show()
""")

md(r"""
- **Plan A:** `gf19` to `gf19p` is the value of the promoter bases alone: +0.051 on ETO,
  and HL +0.0079 across 11 drugs x 5 folds (p=1.3e-4).
- **Plan B:** `gf19r` gains a further +0.0095 on ETO, but adds 7 loci and bundles three
  changes, so the edge's own contribution is **unmeasured**.
- **Whole genome:** 0.747 against `gf19p`'s 0.793, down on 5/5 folds, with folds from 0.633
  to 0.858. Its 4,419 regulated-locus edges did not close that gap.
""")

# ---------------------------------------------------------------- 2. further designs
md(r"""
## Six designs further from LocusFusion

These change the network, the loss, the labels or the model class. **All six are
proposals: none has been built or trained.** Each must scale to the whole genome and at
least match LocusFusion v3 on the 19 loci or the per-drug loci. First, those parity targets
and where the whole-genome arm stands, read live from the result files.
""")

code(r"""
order = base["LF v3, 19 loci"].sort_values().index
y = np.arange(len(order))
genome = {}
for p in sorted((GF / "size6776").glob("*.json")):
    d = json.loads(p.read_text())
    genome[d["drug"]] = d["cv_auc_mean"]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 5.2), sharey=True,
                              gridspec_kw={"width_ratios": [2.2, 1]})
clean(ax)
clean(ax2)
for col, lab, marker, filled in [("LF v3, 19 loci", "LocusFusion v3, 19 loci", "o", False),
                                 ("LF v3, per-drug", "LocusFusion v3, per-drug loci", "s", False),
                                 ("gf19p", "GenomeFusion, 19 + promoters", "o", True)]:
    mark(ax, base.loc[order, col].values, y, marker, filled=filled, label=lab)
g = [(v, i) for i, d in enumerate(order) if (v := genome.get(d)) is not None]
mark(ax, [v for v, _ in g], [i for _, i in g], "X", s=90, label="GenomeFusion, whole genome")
ax.set_yticks(y, [d.title() for d in order])
ax.set_ylim(-0.6, len(order) - 0.4)
ax.set_xlabel("CV AUC")
ax.set_title("CV AUC by drug and model, lineage split", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=9)

ink = NEUTRAL["ink2"]
ax2.hlines(y, cohort.loc[order, "train_R"], cohort.loc[order, "train"], color=NEUTRAL["axis"], lw=1.5)
mark(ax2, cohort.loc[order, "train"].values, y, "o", label="train isolates", colour=ink)
mark(ax2, cohort.loc[order, "train_R"].values, y, "o", filled=False, label="resistant", colour=ink)
ax2.set_xscale("log")
ax2.set_xlabel("isolates (log scale)")
ax2.set_title("Training cohort by drug", loc="left")
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=9)
fig.tight_layout()
plt.show()

print("macro CV:", base[["LF v3, 19 loci", "LF v3, per-drug", "gf19p"]].mean().round(4).to_dict())
print(f"whole-genome runs finished: {len(genome)}/{len(order)}")
""")

md(r"""
Tying LocusFusion on 19 loci does not predict the genome arm: LEV falls from 0.9555 to
0.7125 and ETO from 0.7934 to 0.7469. LEV has only 210 training isolates (64 R), and its
genome figure is flattered: selecting checkpoints from epoch 21 on gives 0.6767.

The next figure holds closed-form statistics on training and CV rows, with no gradient
training and no TEST scoring.
""")

code(r"""
DF.evidence()
plt.show()
""")

md(r"""
- **Variants:** 636,869 in all. Singletons are 61.6% of them but 2.5% of tokens, and the
  0.4% carried by 1,000+ isolates are 74.6% of tokens. Identical carrier sets reduce them
  to 112,266 patterns.
- **Discordant pairs:** nearest R and S isolates in a sublineage differ by a median of
  61-154 variants, against ~875 per isolate. The drug's own loci appear on the R-only side
  in 51-92% of pairs and the S-only side in 4-26%.
- **Pooled labels:** LEV gains 0.080 and AMK and CAP about 0.02, while KAN and ETO lose
  about 0.016 and MOX 0.039. Sharing has to be soft.
- **Closed-form score:** the stratified score peaks at 0.868 on `19p` and 0.835 on the
  genome, against 0.9134 for LocusFusion v3 on 19 loci.

Which parts of GenomeFusion each design keeps:
""")

code(r"""
DF.overview()
plt.show()
""")

md(r"""
**Recommended first build:** Effect Field with and without the Clade Delta loss, plus
Pattern Forest. That tests attention, lineage handling and deep learning itself for little
new code. Retriever-Reader follows if Effect Field misses parity through lost epistasis,
and Resistome Factorisation if LEV still fails on the genome.

The Clade Delta row assumes Effect Field's encoder, since it is a loss rather than an
architecture. Pattern Forest keeps locus grouping only through its burden features.
""")

md(r"""
### 1. Effect Field

Each token gets its own resistance effect, and effects combine by noisy-OR: one causal
variant is enough, with no attention anywhere. The effect comes from a small network
plus shrunken terms on a genome -> locus -> codon -> allele tree.
""")

code(r"""
DF.effect_field()
plt.show()
""")

md(r"""
Cost is ~875 tokens per isolate. Alleles carried by a single isolate get no free term
and rely on the network, so rare variants pool by locus and consequence. It yields a
genome-wide table of variant effects with exact per-prediction attribution. The main
risk is lost epistasis; an optional low-rank pairwise term is the first fallback.
""")

md(r"""
### 2. Clade Delta

A loss that can wrap any encoder. Training pairs each resistant isolate with its nearest
susceptible relative in the same level-2 sublineage, so what the two share carries little
weight. A small unmatched BCE term keeps scores comparable across clades.
""")

code(r"""
DF.clade_delta()
plt.show()
""")

md(r"""
Cost is 61-154 tokens per pair with an additive encoder. It targets lineage confounding,
but LEV's 0.445 TEST AUC is not evidence for that: the score comes from a near-untrained
checkpoint (see Caveats). It cannot learn from resistance fixed across a whole clade.
""")

md(r"""
### 3. Retriever-Reader

A cheap per-drug scorer, fit inside the training fold, keeps the top 64 tokens per
isolate; a small set transformer reads only those. Reader cost no longer depends on
genome size.
""")

code(r"""
DF.retriever_reader()
plt.show()
""")

md(r"""
Attention falls from ~500k pairs at the across-loci stage to ~4k. On 19 loci the reader
nearly always sees every token, so this is the safest route to parity. Retriever recall on
known loci can be measured before any reader is trained. Anything the retriever misses
is lost.
""")

md(r"""
### 4. Genome Pyramid

Tokens are pooled into occupied coordinate bins, with no loci: each bin summarises its
children, and drug latents attend over bins at every scale. Shifted grids stop a bin edge
from splitting a hotspot. Levels are drawn at 4x per step but are 16x in the design.
""")

code(r"""
DF.genome_pyramid()
plt.show()
""")

md(r"""
Cost is at most ~10k bin operations per isolate (estimated). With no partition, the
promoter-locus trap behind `gf19` and the overlap rules disappear. This is the furthest
from LocusFusion and the least certain on parity.
""")

md(r"""
### 5. Resistome Factorisation

Variant-by-drug effects are low-rank plus a sparse drug-specific term, so LEV can borrow
from MOX without being forced to equal it. The lower panel shows the masking this
requires.
""")

code(r"""
DF.resistome()
plt.show()
""")

md(r"""
The only design that addresses LEV's shortage of labels directly. Drug splits agree on
only 47-74% of fold assignments between the three drug pairs checked, so it needs one
model per target drug, with that drug's validation and test rows masked from every
auxiliary label or they leak. Earlier multi-drug LocusFusion arms were weak, which is the
reason for caution.
""")

md(r"""
### 6. Pattern Forest

Boosted trees on deduplicated variant patterns plus burden counts by locus, codon and
consequence. There is no neural network and no GPU, and trees can split on combinations
of variants directly.
""")

code(r"""
DF.pattern_forest()
plt.show()
""")

md(r"""
This is the cheapest test of whether deep models are needed at all. `abr_env` has no
xgboost or lightgbm, so it starts with sklearn on ~3,000 screened features (~54 MB)
unless an install is approved.
""")

md(r"""
### Side by side, and how to test them

Design claims, not measurements; no design has an AUC.

| design | cost per isolate | route to parity | main risk | first check |
|---|---|---|---|---|
| Effect Field | ~875 tokens | pooled effects per variant | lost epistasis | `19p` on LEV, ETO, PZA |
| Clade Delta | 61-154 tokens per pair | same encoder, new loss | clade-fixed resistance | with vs without the loss |
| Retriever-Reader | ~4k attention pairs | reads every token on 19 loci | retriever misses | recall on known loci |
| Genome Pyramid | <=~10k bin operations | uncertain | no locus identity | `19p` on LEV, ETO, PZA |
| Resistome Factorisation | ~875 tokens x rank | soft sharing across drugs | negative transfer, leakage | LEV and MOX, masked |
| Pattern Forest | CPU; ~54 MB features | splits on variant combinations | rare variants | `19p`, all 11 drugs |

1. **19-locus screen.** LEV, ETO and PZA on `19p`, then all 11 drugs for survivors. A full
   LocusFusion-class pass cost ~8.6 GPU-hours (`gf19p`).
2. **Parity gate.** 55-fold paired Wilcoxon against LocusFusion v3: 0.9134 on 19 loci,
   0.8852 per-drug. GenomeFusion needs a new `drug_loci` + WHO regulatory arm for the
   per-drug test. Proposed rule, not yet agreed: HL >= -0.005 and, if negative, raw
   p >= 0.05. That fails `gf19` (-0.0078, p=0.044) and passes `gf19p`.
3. **Genome canary.** LEV and ETO on the whole genome, judged no earlier than epoch 30.
4. **Full grid.** Only for designs that clearly beat `size6776` on those two drugs.
""")

# ---------------------------------------------------------------- 3. questions
md(r"""
## Questions for the lab

Each of these changes a decision we cannot make from the code, and each is
answerable without knowing anything about the model. Plan C's unit is defined under
[Context](#Context).

### A. What belongs in a unit

1. **Operons.** The WHO *inhA* region (813 bp) contains the entire *fabG1* CDS,
   and the *fabG1*-*inhA* promoter is the c-15t site. Should *fabG1* and *inhA*
   be **one unit** sharing that promoter, or two units with the promoter
   assigned to one of them? *Decides whether a locus may belong to more than
   one unit — which is the one place plan C can get ugly.*

2. **Coding regulatory regions.** 64 of the 126 loci a WHO region touches are
   coding; embC's region is 1,980 of 1,982 bp CDS. Are those genuinely
   regulatory sequence, or an artifact of how the catalogue drew region
   boundaries? *Decides whether they enter the unit table at all.*

3. **The 5' flank default.** Catalogued region extents run 2-1,982 bp, median
   100. For a gene with no catalogued region, is a flat 100 bp upstream window
   the right default for an M. tuberculosis promoter, or should it be
   gene-specific — and is there a better generic number? *Decides the `19p`
   flank and the default unit boundary.*

4. **Disjoint regions.** A few WHO regions come in separated segments (panD,
   rplC, tlyA, Rv2752c). Is each segment independently regulatory, or is one
   the real site? *Decides whether a unit can be non-contiguous.*

### B. What counts as known well enough to hard-code

5. **Beyond WHO Table-22.** We currently assert only Table-22 relationships
   (~61). Which other sources would you defend as fact rather than hypothesis —
   curated operon calls, the whiB7 regulon, TFOE/ChIP-seq networks? Name the
   ones you would put your name on. *Decides how big the unit table gets, and
   whether this stays ~61 rows or becomes thousands.*

6. **Where you would not trust geometry.** Plan A/B currently infer "upstream
   of" from the next start codon by strand. Where is that wrong often enough to
   matter in this genome — nested genes, leaderless transcripts, long 5' UTRs?
   *Decides whether the geometric fallback keeps any asserted role.*

### C. What would convince you

7. **The readout.** Would "38% of the *fabG1* unit's signal came from its
   promoter tokens, and the ethionamide query puts most of its weight on that
   unit" be a claim you would believe and could use? Or do you need per-base
   attribution? *Decides what we build for interpretation, and what goes in the
   figure.*

8. **Negative controls.** For one drug of your choice, name a handful of loci
   you would expect to be **irrelevant** — and one or two you would expect to
   matter that are not in the WHO 19. *Gives us a falsifiable test of the
   readout that we did not choose ourselves.*

9. **A held-out truth set.** Is there a regulator -> target set you trust
   enough that recovering it would count as evidence? *Decides whether the
   function-discovery claim is testable at all; with ~61 catalogued regions we
   currently do not think it is.*

10. **The uncatalogued hit.** If the model puts weight on a locus with no known
    relationship to the drug, is that a lead you would follow up, or noise you
    would dismiss? *Decides whether we keep attention over all 6,776 loci or
    restrict the model to units.*
""")

# ---------------------------------------------------------------- 4. context
md(r"""
<a id="Context"></a>
## Context

**Shared by all three plans,** so no comparison turns on them:

- **Partition.** Genes plus intergenic segments, one locus per base, no duplicated
  upstream windows; the RLC/PE-PPE mask removes 9.1% of the genome.
- **Token.** 57 channels, one token per changed base or indel event, nothing capped. Only
  the loci an isolate changes enter the network (~707 of 6,776 on average, max 2,466).
- **Two stages.** A shared encoder *within* each active locus reading out at a `[NULL]`
  token, then an encoder *across* the active loci plus a learned reference token, then
  one drug query attending over the loci.

All plans can surface an uncatalogued locus through stage-2 attention. Where they differ
(design properties, not measurements):

| | A. anonymous distance | B. feature concat + gated edge | C. declared units |
|---|---|---|---|
| relationship is expressed as | a distance in bp | 1 global scalar x shared embedding | membership in a unit |
| asserted relationships encoded | 0 | ~61 WHO, diluted by 4,350 geometric | ~61 WHO, undiluted |
| auditable as a standalone file | n/a (nothing asserted) | no | yes (~61-row unit table) |
| promoter base's coordinate | offset in its intergenic locus | offset in its intergenic locus | negative offset from the ATG |
| slots per gene in stage 2 | 2+ (gene, promoter) | 2+ (gene, promoter) | 1 (the unit) |
| handles coding WHO regions | no | yes (tags, never moves a base) | bases already in the gene's locus |
| handles acting-at-range / trans | no | yes, if catalogued | yes, if catalogued |
| measured on ETO | 0.7934 (`gf19p`) | 0.8029 (`gf19r`), 0.7469 (whole genome) | not run |

Plan C also halves the across-loci sequence length for every gene with a catalogued
promoter, and that stage is quadratic in it.

### Plan A: anonymous distance

**As built** (`gf19`, `gf19p`, and the token layer of every arm). Two token channels
(`layout.py:58`) carry `exp(-d / 50 bp)` to the next start codon on the `+` and `-` strand,
filled per-token from `partition.up_plus / up_minus` (`tokenizer.py:310-311`).

A token therefore says "this base is about 15 bp upstream of some `+`-strand gene", but
not **which** gene. The promoter gets its own slot in the across-loci stage, and the drug
query has to learn that this slot and the *fabG1* slot belong together from their
embeddings alone. It holds proximity, not the relationship. It breaks on the 64 WHO
regions that are coding, anything acting at range, and anything acting *trans*.

### Plan B: feature concat + gated edge

**As built** (`gf19r`, `size6776`). Three mechanisms stacked:

1. `annotate_regulatory()` (`partition.py:313`) tags the loci a WHO Table-22 region
   overlaps, without moving a base. 126 loci tagged, 64 coding.
2. `regulated_index()` (`partition.py:137`) reduces that to one `int32` per locus: the
   locus it acts on, or `-1`. It takes `L.regulates` (WHO, asserted) first, else
   `L.downstream` (the next start codon by strand, inferred geometrically), and `break`s on
   the first hit. `feature_matrix()` then builds each locus's row as
   `[own 29 features | the target's 29 features]` -> (6776, 58).
3. In the model (`model.py:147-157`) a tagged locus's embedding picks up the learned
   residual embedding of the gene it regulates, scaled by a single parameter `reg_gate`
   initialised at zero.

So there is no object in the system a biologist can point at. Four specific problems:

- **One global gate for 4,419 edges.** `reg_gate` is a single scalar (`model.py:116`). It
  cannot be large for *fabG1* and zero for 4,000 geometric guesses, so its gradient is an
  average dominated by the guesses.
- **~70:1 dilution.** Only ~61 WHO regions produce asserted edges; the rest come from the
  `downstream` fallback.
- **One edge per locus.** The `break` at `partition.py:156` keeps only the first target, so
  a locus over an operon keeps one gene.
- **One direction.** A promoter knows its gene; the gene does not know its promoter.

### Plan C: declared regulatory units

**Proposal.** Express the relationship as **membership in one unit**: a gene locus plus the
regulatory loci catalogued as acting on it. A tagged regulatory locus's tokens are routed
into its gene's within-locus encoder, and the across-loci stage runs over units.

A token takes its locus at `tokenizer.py:219` and the loader derives everything downstream
from `T["locus"]` and `store.sum_locus`, so this is a **remap of two arrays**: no
re-tokenizing 15.8M tokens, and the partition stays one-locus-per-base for accounting. The
model needs less machinery: `regulated_lid`, `reg_gate` and the regulated feature half all go.

1. **The relationship becomes data.** A ~61-row CSV: unit, member loci, coordinates, source
   (WHO Table-22 row / 5' flank).
2. **The coordinate becomes the catalogue's own numbering.** `coord` is
   `pos - locus_start[locus]` (`data.py:194`). Route a promoter token into *fabG1*'s locus
   and it goes negative, and -15 is how the catalogue writes c-15t.
3. **The readout is a within-unit attention share.** `return_attn` already exists
   (`model.py:237`). "38% of the *fabG1* unit's vector came from its promoter tokens" is a
   sentence about bases and genes.

LocusFusion v3 does this implicitly: its per-gene FASTAs carry a 100 bp 5' flank, so c-15t
sits inside `dna:fabG1`. Units also remove the dilution problem, because there is no gate to
average 61 asserted relationships against 4,350 guesses.
""")

# ---------------------------------------------------------------- 5. caveats
md(r"""
## Caveats

- Plan C is unbuilt and unmeasured. Every claim for it above is a design
  argument or an analogy to LocusFusion v3's padded FASTAs, not a result.
- `gf19r`'s +0.0095 over `gf19p` on ETO bundles three changes (7 more loci,
  regulated features, the edge) and rests on 5 folds of one drug, where the
  exact two-sided Wilcoxon p floors at 0.0625. It is not evidence that the
  edge works. The `--no-regulated-features` control has not run.
- Whether `reg_gate` ever moved off zero is unverified. The saved state dicts
  (`train.py:392`) would settle it in minutes, no GPU; until then, plan B's edge has
  no measured contribution in either direction.
- The whole-genome arm has finished only ETO (0.747) and LEV (0.7125); the other
  nine drugs were still running on 2026-11-05. It also sits near chance for its
  first ~20 epochs before finding the locus, so any run cut early or patienced
  before epoch 30 understates it — `min_epochs` must stay >= 50.
- LEV's whole-genome numbers are checkpoint artifacts. Selection tracks best weights from
  epoch 1, so fold 0 (n=23) restored epoch 6, near-untrained, and won best fold; TEST 0.445
  comes from that model. CV is flattered the same way (0.7125, or 0.6767 selecting from
  epoch 21 on). ETO is unaffected.
- Only 4,419 of 6,776 loci carry an edge, when the geometric fallback should
  hit nearly all of them. `by_gene` (`partition.py:150`) indexes on `L.gene`
  while `downstream` stores `gene or tag`, so tag-only genes likely fail the
  lookup. Unverified, and it would mean plan B is quietly weaker than designed.
- ETO is the maximum-effect drug for promoter mechanisms, chosen deliberately.
  Nothing here establishes that the ordering holds on drugs whose resistance is
  coding-only.
- None of the six further designs has been built or trained, so parity is untested
  for all of them. In their schematics the variant names are real, but bar lengths,
  heat values, trees and token positions are illustrative, not fitted.
- The closed-form scores are not models and not a floor on what a model reaches.
  Their settings were compared on validation folds, so they are slightly optimistic.
- The discordant-pair rates are descriptive, on training rows. R-only sets are 1.3-2x
  larger than S-only sets, which explains part of the gap, and only 0-10% of pairs
  differ by 25 variants or fewer.
- The transfer probe pools labels outright. Resistome Factorisation shares softly, so
  the probe motivates it but does not estimate it.
- The Genome Pyramid drops locus grouping, not annotation: its tokens still carry
  region and consequence channels computed from gene models.
- The proposed parity gate uses a margin, which departs from the Wilcoxon-only
  convention used for the paper. It is a go/no-go rule for this work, not a claim.
""")

nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"}},
    "nbformat": 4, "nbformat_minor": 5}

OUT.parent.mkdir(parents=True, exist_ok=True)
import json as _json
OUT.write_text(_json.dumps(nb, indent=1) + "\n")
print(f"wrote {OUT} ({len(cells)} cells)")

# 200 bp windows around the whole-genome lasso's top 50 coefficients

**Run 2026-09-10. 11 drugs x 3 arms, SLURM jobs 63965045-63965055, all COMPLETED.**

`lasso_wholegenome_20260903` fits L1-logistic on ~10^5-10^6 variant columns per
drug and lands at macro test AUC 0.9170. This run asks whether that model can be
reduced to the neighbourhood of its own top coefficients: take the 50 largest,
open a 200 bp window around each, refit on those windows alone. The encoded
footprint is 8,111 bp on average — **0.18% of the genome** — and 4,204 columns
against the parent's 426,457, a median 63-fold reduction.

The answer is that it can, but not for the reason the design implies: the 50
selected sites are not what carries the signal.

## Arms

| arm | columns | what it isolates |
|---|---|---|
| `delta-top50` | the <=50 selected keys | what the selection alone is worth |
| `delta-window` | every cohort key in the windows | + local context, encoding fixed |
| `onehot-window` | the same keys + a wild-type indicator per site | + the encoding, columns fixed |

Reading the requested arm on its own cannot separate a context effect from an
encoding effect, which is why all three were run. `delta-top50` and
`delta-window` cost 1-5 s per drug; the whole cost of the run is the wild-type
block in `onehot-window` (250 s mean, 934 s worst).

## Results, TEST AUC

TEST is the column to read. The 50 keys came from the parent's best-fold model,
which was fitted inside the same 80% training split at seed 42, so the held-out
20% took no part in choosing the windows. **CV is selection-optimistic** and
finding 3 measures by how much.

Paired over 11 drugs:

| contrast | mean | median | wins | Wilcoxon p | mean w/o LFX |
|---|---|---|---|---|---|
| `onehot-window` vs parent | +0.0042 | −0.0011 | 5/11 | 0.90 | −0.0001 |
| `delta-window` vs parent | +0.0034 | −0.0003 | 5/11 | 0.90 | −0.0009 |
| `delta-top50` vs parent | **−0.0100** | −0.0112 | 1/11 | **0.002** | −0.0113 |
| context (`delta-window` − `delta-top50`) | **+0.0134** | +0.0098 | **10/11** | **0.002** | +0.0105 |
| encoding (`onehot-window` − `delta-window`) | +0.0008 | +0.0005 | 7/11 | 0.46 | +0.0007 |

Macro test AUC: parent 0.9170, `delta-top50` 0.9070, `delta-window` 0.9204,
`onehot-window` 0.9213.

## Findings

### 1. 0.18% of the genome matches the whole genome

`onehot-window` is +0.0042 macro over the parent, but the median is −0.0011, it
wins 5 of 11, Wilcoxon p = 0.90, and dropping LEVOFLOXACIN — 269 isolates, a
54-isolate test set, the noisiest cell in the project — takes the mean to
−0.0001. The two models are indistinguishable. The positive macro is one small
drug, not a trend.

That is the useful result and it is a negative one about the genome: nothing
outside these 27-49 windows is doing measurable work. `delta-window` reaches it
in **2.6 s per drug** against the parent's ~60 s, on 63x fewer columns.

### 2. The top 50 coefficients are locations, not the signal

Restricted to exactly the 50 keys the parent selected, the model **loses**
0.0100 macro test AUC, winning 1 of 11 (p = 0.002). Restoring their 200 bp
neighbourhoods recovers it and more: +0.0134, 10 of 11, p = 0.002 — the largest
and most consistent effect in the run.

The coefficient mass says the same thing. In `delta-window`, mass splits
**49.9% on the 50 selected keys, 50.1% on neighbours the parent never
selected**, and in five of eleven drugs the neighbours take the majority
(CAPREOMYCIN 81%, PYRAZINAMIDE 79%, ETHAMBUTOL 71%, ISONIAZID 69%,
ETHIONAMIDE 65%).

So the parent's support is not a set of uniquely informative sites. L1 picked
one column from each of ~50 correlated local blocks, as L1 does; the information
is spread across the block, and half of it sits on columns the parent set to
zero. **A top-k coefficient list from this model is a list of neighbourhoods,
and reading it as a list of sites loses half the signal.** This applies directly
to the parent run's WHO-membership measurement, which was computed over the
support: the support is a lossy sample of where the model is actually looking.

### 3. Selecting features on training data inflates CV by up to +0.059, entirely at small n

Same 50 columns, `delta-top50` vs the parent, on the two metrics:

| | macro CV | macro TEST |
|---|---|---|
| n >= 7000 (5 drugs) | −0.0015 | −0.0118 |
| n < 7000 (6 drugs) | **+0.0341** | −0.0085 |

CV rises with the selection and test falls with it. The inflation is almost
perfectly rank-ordered by cohort size (Spearman rho = **−0.90**, p = 0.0002),
running from −0.0048 at RIFAMPICIN (n = 17,582) to +0.0593 at LEVOFLOXACIN
(n = 269), with ETHIONAMIDE +0.0444 and MOXIFLOXACIN +0.0447.

This is a measurement of a pipeline this project could easily have run by
accident: select top-k features, refit, quote cross-validation. At n < 3000 that
reports up to six points of AUC that do not exist. Honest CV requires
re-selecting inside every fold, which means a genome-wide refit per fold; this
run did not do that, and the gap between its two columns is the reason to.

### 4. The one-hot advantage does not survive outside the curated loci

`lasso_wholeseq_20260901` measured `onehot` at +0.009 macro over `delta` on the
19 loci. Here, with columns held fixed, it is **+0.0008, 7 of 11, p = 0.46** —
zero. It is also not free: the wild-type block takes the mean fit from 2.6 s to
250 s, a 96x cost for nothing.

The wild-type indicators are not idle, which is what makes this a real result
rather than a null one. They absorb **30.5% of total coefficient mass** (19.9%
to 42.0% across drugs), pulling the seed keys' share from 49.9% down to 33.2%.
The reparametrisation moves a third of the mass onto the new columns and leaves
the fitted function essentially unchanged.

## What "one-hot the window" means here

There is no whole-genome H37Rv FASTA in the corpus — `genomic_data/aligned`
holds the 19 loci and nothing else — so window sequence is reconstructed from
the cohort's own calls, not read off a reference. This costs nothing: a position
where no isolate carries a call is constant across the cohort, and a constant
column is collinear with the unpenalised intercept, so its L1 optimum is exactly
zero in every fold. The encoded window is therefore its polymorphic positions,
and at each such position the columns are one per distinct `REF>ALT` key
observed there plus one wild-type indicator, set where the isolate carries none
of them. That is the repo's `onehot` coding lifted from an alignment column to a
variant site. Indels are keyed by start position and treated as one allele
there, matching the parent.

Density is in the JSONs as `n_sites` against `window_bp`: 4.3% of window
positions are polymorphic for LEVOFLOXACIN's 269 isolates, rising to 26.7% for
ISONIAZID's 17,436, so the window count of columns is a cohort-size effect and
not a property of the locus.

## Limits

- CV is selection-optimistic throughout; only TEST is clean. Finding 3 is the
  size of the problem, not a correction for it.
- LEVOFLOXACIN (n = 269, 22 nonzero parent coefficients, so its "top 50" is 22)
  drives every positive macro mean in the run. Every headline here is quoted
  with and without it.
- Windows are seeded from one fit, the parent's best fold. A different fold
  would select a partly different 50, and finding 2 implies the overlap would be
  imperfect while the windows would land in much the same places.
- The parent could only use `delta` genome-wide, so `delta-window` is the
  matched arm; `onehot-window` differs from the parent in two ways at once.
- **CV AUC in this family reproduces only to ~4e-4.** ETHIONAMIDE was run twice,
  once interactively and once as SLURM job 63965048, same code and same seeds:
  `best_C`, `n_features`, `cv_support_mean` and every test metric came back
  identical, but `cv_auc_mean` moved by 3.9e-4 (`delta-window`) and 3.6e-4
  (`onehot-window`). The cause is `lasso_wholeseq._fit`, which every run in this
  family imports and which never passes `random_state`, so liblinear's
  coordinate-descent permutation is drawn from the global NumPy state. The
  headline numbers here are all TEST and are exactly reproducible; the CV
  columns are not. It also sets the floor for what counts as an effect:
  finding 4's +0.0008 encoding gain is about twice the solver's own jitter,
  which is the precise sense in which it is zero. Adding `random_state` to
  `_fit` would fix this, but it would also change every already-published CV
  number in `lasso_wholeseq_20260901` and `lasso_wholegenome_20260903`, so it
  is flagged here rather than done.

## Reproduce

    bash results/experiments/lasso_windows_20260903/submit.sh          # 11 jobs
    bash results/experiments/lasso_windows_20260903/submit.sh RIFAMPICIN

    python results/experiments/lasso_windows_20260903/analyze.py       # tables

Direct, without SLURM:

    python results/experiments/lasso_windows_20260903/lasso_windows.py \
        --drugs all \
        --wholegenome results/experiments/lasso_wholegenome_20260903/all_drugs \
        --out results/experiments/lasso_windows_20260903/all_drugs

`--top-k` sets how many parent coefficients seed the windows (50) and
`--half-window` their half-width (100, giving the 200 bp window).

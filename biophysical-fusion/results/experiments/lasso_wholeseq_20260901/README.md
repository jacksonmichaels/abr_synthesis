# `lasso_wholeseq_20260901` — L1-logistic on the whole aligned sequence

## The question

Every linear baseline this project has run so far was fed a *distilled* input.
`variant_aggregators_20260825/sparse_baseline.py` tokenises each isolate into at
most 512 differences from H37Rv and fits an L1-logistic on those, reaching macro
CV **0.878** over the 11 shared drugs. That is a strong number, but it is not
the comparison the networks deserve: the CNNs are handed the entire alignment,
every column, and left to find the signal themselves.

**Does a lasso on the raw one-hot alignment — no variant calling, no feature
engineering, no convolution — reach what the convolutional stack reaches?**

The bar is set by three numbers already in the project:

| reference | macro CV AUC |
|---|---:|
| SD-CNN, leak-corrected (published single-drug baseline) | 0.8636 |
| L1-logistic on the variant design matrix (`sparse_baseline`) | 0.878 |
| single-drug, all modalities, 19 loci (best cell in the project) | 0.9246 |

## The grid

2 encodings × 2 locus universes × 11 drugs = **44 cells**, DNA only.

| axis | values |
|---|---|
| **encoding** | `onehot` — one indicator per (alignment column, base), cohort-constant columns dropped. `delta` — the repo's H37Rv reference coding, nonzero only where the isolate differs. |
| **locus universe** | `perdrug` — BIG-TB's SD-CNN per-drug map (2–3 loci), the locus-matched arm. `all` — every curated locus on disk (19), what the joint runs see. |
| **penalty path** | C ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3}, reported in full. |

Results land in one folder per arm — `onehot_perdrug/`, `onehot_all/`,
`delta_perdrug/`, `delta_all/` — each with `{DRUG}__{tag}.json` and a
`summary.csv` in `run_experiment.py`'s schema, so the notebook builders read
them like any other run.

## Matched control

The matched control for the `perdrug` arms is **`full_run_v2`'s DNA-only
single-drug cells** — same drugs, same loci, same 5-fold protocol, same 80/20
test split at seed 42. The only thing that changes is the model.

For the `all` arms it is **`alllocus_run_v2`'s DNA-only single-drug cells**, on
the same reasoning. That axis is the one README finding 2 says actually moves
the number (+0.016 to +0.027 for handing a single-drug model the full 19 loci),
so the sweep spends half its jobs testing whether a linear model gets the same
lift from the same loci.

The **`sparse_baseline`** run is the other control, and the one this run is
really arguing with: same model family, same protocol, distilled input versus
raw input.

## Why both encodings

They carry the same information and span the same column space — `delta` is
ordinary dummy coding with H37Rv as the reference level — but **L1 is not
invariant to reparametrisation**, so they are not the same model. `onehot`
penalises "has base *b* here", wild-type included; `delta` penalises "differs
from H37Rv here". The smoke test already shows the gap is real: on
LEVOFLOXACIN, `perdrug`, `onehot` gets CV 0.934 and `delta` gets 0.892.

## What this run deliberately does not do

- **No other modality.** Protein and biophysical features are deterministic
  functions of the DNA at the same positions; for a *linear* model they are very
  nearly collinear with the one-hot columns and would mostly buy penalty, not
  signal. Regulatory windows are the one arm with a real case for inclusion
  (ETO +0.089, INH +0.031 for the networks) and are the obvious follow-up, not
  part of this grid.
- **No elastic-net, no ridge.** This is a lasso sweep. Ridge is the natural
  control for "is sparsity doing the work, or just shrinkage?" and is the second
  obvious follow-up.
- **No `--fit-vocab-on-train` arm.** The variant baseline needed one because its
  variant vocabulary was built on the full cohort. This one does not: the only
  cohort-wide decision here is dropping constant columns, and a constant column
  is collinear with the unpenalised intercept, so its L1 optimum is exactly zero
  — within a fold as well as over the cohort. Nothing is fit differently.
- **No standardisation.** The columns are 0/1 indicators already on a common
  scale. Standardising divides each by its own √(p(1−p)), which would inflate a
  singleton variant by two orders of magnitude relative to a common one — the
  opposite of what a rare-variant penalty should do.
- **Single seed**, like every other run here. Differences under ~0.01 are
  unresolved, not small.

## Protocol

Identical to `training/multimodal.run_modal_cv`, step for step: missing
phenotypes dropped before splitting; held-out test =
`train_test_split(test_size=0.2, random_state=42, stratify=y)`;
`StratifiedKFold(5, shuffle=True, random_state=42)` on the training split;
`class_weight="balanced"` as the linear analogue of the networks'
inverse-frequency alpha; AUC / AUC-PR (resistant positive) / `tb.get_threshold_val`
sens-spec through the same helpers; test scored once with the best CV fold's
model. `cv_auc_mean` is the best C on the grid, and every C's per-fold AUC and
support size is kept in the JSON so the selection is inspectable rather than
silently tuned.

## Reproduce

```bash
# from biophysical-fusion/
bash results/experiments/lasso_wholeseq_20260901/submit.sh          # all 44 cells

# or one cell, interactively
python results/experiments/lasso_wholeseq_20260901/lasso_wholeseq.py \
    --drugs ISONIAZID --encoding onehot --locus-set all \
    --out results/experiments/lasso_wholeseq_20260901/onehot_all
```

## Scale, measured

The 19 curated loci are 39,646 aligned nt, of which 16,283 columns vary at all
across the cohort. The `onehot`/`all` cell is therefore **33,218 columns x
17,436 isolates at 16,130 nonzeros per isolate** — 281 M nonzeros, 11.3 GB peak
RSS, 23 s per `liblinear` fit. That is the worst cell in the grid; the
`perdrug` cells are two orders of magnitude smaller (LEVOFLOXACIN: 94 columns).

`_locus_block` is verified against the repo's own `one_hot_nt` and
`delta_one_hot_nt` on real `rpsL` data: identical values on every retained
column, and every dropped column confirmed constant across the cohort.

## Findings

All 44 cells completed (SLURM 63877098–63877141, all `COMPLETED`). Macro CV AUC
over the 11 shared drugs, `compare.py --net-pick best`:

| | macro CV | |
|---|---:|---|
| SD-CNN, leak-corrected | 0.8636 | |
| lasso, `delta`, per-drug loci | 0.8613 | |
| lasso, `onehot`, per-drug loci | 0.8750 | |
| L1-logistic on the variant design matrix | 0.8788 | the distilled control |
| best DNA-only network cell, per-drug loci | 0.8940 | matched control |
| lasso, `delta`, 19 loci | 0.9152 | |
| best DNA-only network cell, 19 loci | 0.9184 | matched control |
| **lasso, `onehot`, 19 loci** | **0.9241** | |
| project's best cell (all modalities, 19 loci, single-drug) | 0.9246 | |

### 1. A lasso on the raw alignment matches the best cell in the project

**0.9241 against 0.9246** — a difference of 0.0005, which on a single seed is
no difference at all. The project's best cell is a 46 M-parameter network over
four modalities. This is L1-logistic regression on a one-hot alignment, DNA
only, at a mean of **228 nonzero coefficients out of 22,384 columns** — 1.0% of
the design matrix, and roughly 0.0005% of the network's parameter count.

That is the finding this run was built to test, and it lands on the side that
costs the project the most to accept.

### 2. Raw alignment beats called variants by +0.045, same model family

0.9241 against the variant-token L1's 0.8788, identical protocol, identical
solver, identical penalty path. The only difference is the input. Distilling
each isolate to ≤512 (locus, position, base) tokens against H37Rv was throwing
away four and a half points of macro AUC.

### 3. Reference coding costs, consistently

`onehot` beats `delta` in **both** locus universes — +0.009 at 19 loci
(0.9241 vs 0.9152), +0.014 at the per-drug sets (0.8750 vs 0.8613) — and on
**11 of 11 drugs in both**, without a single exception. The two arms span the
same column space, so this is
purely the reparametrisation: L1 applied to "has base *b* here" is not L1
applied to "differs from H37Rv here", and keeping the wild-type level as its own
penalised column turns out to be worth about a point.

This is a direct caution for the variant-token architectures, all of which are
built on `--delta` input by construction.

### 4. Locus universe dominates — more for the lasso than for the networks

Handing the model all 19 loci instead of the drug's own 2–3 is worth **+0.049**
(`onehot`, 0.8750 → 0.9241) and **+0.054** (`delta`). The same move is worth
**+0.024** to the networks (0.8940 → 0.9184). README finding 2 said locus
universe, not multi-task sharing, is what pays; this says so again and louder,
in a model with no architecture to confound it.

### 5. The networks earn their parameters on small inputs and not on large ones

At the per-drug locus sets the lasso **loses** to the DNA-only networks by
−0.019 (0.8750 vs 0.8940). At 19 loci it wins by +0.006 (0.9241 vs 0.9184).
Per drug at 19 loci that is 7 wins and 4 losses, and every margin is inside
±0.01 — unresolved on one seed — except ETHIONAMIDE (+0.016) and LEVOFLOXACIN
(+0.050, and LFX is 269 isolates, the noisiest drug here; do not lean on it).
The four losses are AMIKACIN (−0.002), ISONIAZID (−0.001), KANAMYCIN (−0.008)
and MOXIFLOXACIN (−0.002) — all of them noise.
**Parity is the honest reading**, not victory.

### 6. Part of the 19-locus lift is co-resistance, not mechanism

The lasso is readable, which is how this became visible. The largest
coefficients of the `onehot`/19-loci models are right about the mechanism and
also full of loci that cannot possibly be causal:

| drug | mechanistically correct picks | picks from other drugs' loci |
|---|---|---|
| ISONIAZID | `katG` ×3, `fabG1` ×2 | `ethA:1290`, `rpoB:1418` |
| PYRAZINAMIDE | `pncA` ×5 | `embB:319`, `embA:403`, `gid:292` |
| ETHIONAMIDE | `ethR:282`, `ethA` ×2, `fabG1:85` | `embC:1799`, `rpoB:39`, `katG:15`, `pncA:140` |

`rpoB` cannot cause isoniazid resistance and `embB` cannot cause pyrazinamide
resistance. What those columns carry is **MDR co-occurrence** — a strain
resistant to one first-line drug is likelier to be resistant to the others — and
lineage structure. The model is using it because it is there and it predicts.

This matters beyond this run: the +0.049 that finding 4 attributes to the locus
universe is *partly* co-resistance signal, and **the networks given the same 19
loci can exploit exactly the same thing**. It does not invalidate finding 4 — the
lift is real and reproducible — but "more loci help" and "more loci carry
mechanism" are not the same claim, and only the first is evidenced.

### 7. ETHIONAMIDE, the project's worst drug, moves the most

0.8118 here against 0.7954 for the best DNA-only network at 19 loci, 0.7590 for
the variant-token L1, and 0.6220 for the leak-corrected SD-CNN. The `fabG1`
coefficients in its top-8 are the *fabG1–inhA* operon promoter that README
finding 1 identifies as the dominant ETO mechanism — and this arm reaches it
from the coding alignment alone, with no regulatory modality loaded.

## What this run does not settle

- **Whether the co-resistance in finding 6 is confounding or legitimate signal.**
  Deciding that needs a lineage-stratified or held-out-by-lineage split, which
  no run in this project has. It is the highest-value follow-up here.
- **Whether the CNNs are redundant.** Parity at 19 loci on DNA is not parity
  everywhere; the best cell in the project uses four modalities, and this run
  loaded one.
- **Ridge.** Without an L2 arm, "sparsity is doing the work" is not established
  — only that a sparse model suffices.
- **Single seed**, like everything else here. Margins under ~0.01 are
  unresolved.

## Reproduce the table

```bash
python results/experiments/lasso_wholeseq_20260901/compare.py
python results/experiments/lasso_wholeseq_20260901/compare.py --net-pick mdcnn
```

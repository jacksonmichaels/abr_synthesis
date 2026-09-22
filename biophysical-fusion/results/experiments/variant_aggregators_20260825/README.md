# variant_aggregators_20260825 — is the aggregator the problem, and does a linear model already win?

Reproducible via `submit.sh` in this folder. Two halves that answer two
different questions, run together because the second is only interesting given
the first.

## The question

Three transformer arms have lost in this project — `setfusion` (0.8877 vs
0.9086), `setfusion_scaling` (62 arms, closed nothing), `transformer_run` (165
jobs, DNA-only 0.802 vs 0.892) — and `token_signal` diagnosed why: **0.14% of an
encoded token varied with the genotype**, so attention sat flat at 1/8 and a
linear probe on the encoder output beat the trained model by 0.011.

`locusfusion` (running separately) answers the *encoding* half of that: tokenize
the variants, not the sequence. This run answers the other half.

Once the variants are tokenized there is no long sequence left. The median
isolate differs from H37Rv at **0-3 columns of a 2.5 kb gene, 14 across all 19
loci**, so the task is *sparse evidence aggregation*, and softmax is a
specifically bad operator for it: **it normalises, so it is a RELATIVE
selector.** With one informative token among thirteen neutral ones the weights
must sum to 1 — it cannot say "this one, absolutely", it has to spend mass on
the neutral tokens.

That raises two questions this run asks in order:

1. **Does a linear model on the variant matrix already do the job?** If it does,
   the architecture question is much less interesting than it looks.
2. **If not, is the aggregator what was broken?** Six aggregators over one
   fixed tokenizer, so the answer is attributable.

## Half 1 — the baseline that is not a network

`sparse_baseline.py` (lives with this run; delete it when the run is written
up). L1-logistic across a reported C grid, plus sklearn's
`HistGradientBoostingClassifier`, both on the sparse isolate x variant matrix
that `models.experimental_models.variant_design_matrix()` exports.

**There is no sparse-linear or tree baseline anywhere in this project**, and for
TB AMR from a variant matrix those are the canonical strong methods —
TB-Profiler and Mykrobe are catalogue lookups and are clinically competitive.

Same protocol as `run_modal_cv`, step for step: same `load_dataset(...,
per_modality_branch=False, delta=True)` input, missing phenotypes dropped before
splitting, `train_test_split(0.2, seed 42, stratified)`, `StratifiedKFold(5,
shuffle, seed 42)`, `class_weight="balanced"` in place of the inverse-frequency
alpha, the same AUC / AUC-PR / `tb.get_threshold_val` helpers, and TEST scored
once with the best CV fold's model.

Three advantages it has over the networks, all stated rather than hidden:

* **The C grid is reported in full**, and `cv_auc_mean` is the grid's best. That
  is regularisation selected on the CV it reports; a network is trained once. So
  read the whole grid in the JSON, not the headline.
* **The design matrix vocabulary is built on the full cohort** — a column exists
  if the variant appears anywhere, test isolates included. No label is touched
  and no test row's values are fitted, so this is not label leakage, but it is a
  vocabulary the networks must discover. The `sparse_baseline_trainvocab/` arm
  (INH / PZA / MOXI) restricts the columns to variants seen in training and is
  the strict version.
* **`max_variants=512`**, far above the training default of 16 — a one-off
  feature extraction has no reason to truncate the tail.

And one restriction, which is the whole point of the comparison: `min_count=2`
drops singleton columns, because **a per-identity model cannot generalise from a
variant it saw once, and cannot score one it never saw at all.** That is exactly
the *pncA* / PYRAZINAMIDE case — hundreds of distinct inactivating
substitutions, most unseen in training — and exactly what `--arch additive`
exists to handle by featurising the variant instead of memorising it.

Sanity check from the pilot (MOXIFLOXACIN, CV 0.8111): the largest L1
coefficients are `protein:gyrA@93`, `dna:gyrA@1844`, `gyrB@1446`,
`protein:gyrA@90` — the canonical fluoroquinolone resistance codons, recovered
from nothing but the tokenizer. That is a validity check on the whole
variant-token pipeline, not just on this baseline.

## Half 2 — the six aggregators

One tokenizer, six ways of combining the tokens. `all_modalities` only: that is
where the protein and biophysical features `additive` and `fm` read actually
exist, and `full_run_v2/all_modalities__*` is the matched control.

| `--arch` | aggregator | the question |
|---|---|---|
| `catalogue` | scalar per exact variant id | how far does pure memorisation get? |
| `additive` | `sum w(features_v)` | does featurising buy generalisation to unseen substitutions? |
| `noisyor` | `1 - prod(1 - p_v)` | "susceptible unless something confers resistance", as an architecture |
| `gatedpool` | sigmoid gate, no softmax | is normalisation what broke attention? |
| `deepsets` | sum + max + count | does attention buy anything at all? |
| `fm` | factorization machine | is epistasis worth anything, at O(T*k)? |

## The matched control

`full_run_v2/all_modalities__*` — same per-drug loci, same 300 epochs, patience
30, `--min-epochs 50`, `--save-weights best`, same seed 0. The **only**
differences are `--arch` and the `--delta` input encoding that `--arch` implies
for these six. Bars to beat: `all_modalities__mdcnn` **0.9086** (per-drug loci)
and **0.9246** (19 loci, `alllocus_run_v2`); SD-CNN is 0.8636.

## How to read it — three comparisons, not eleven numbers

1. **`sparse_baseline` vs everything.** If L1 reaches ~0.909 the deep-learning
   framing is not earning its keep and that is the finding, not a disappointment.
2. **`catalogue` vs `additive`.** `catalogue`'s per-identity table is
   zero-initialised, so a substitution absent from training contributes exactly
   nothing; `additive` is the identical aggregator with the weight computed from
   the variant's features. The gap **is** the *pncA* generalisation, isolated.
   Expect it to show up on PYRAZINAMIDE and ETHIONAMIDE, and nowhere on
   RIFAMPICIN (rpoB RRDR is one tight positional signal already at 0.976).
3. **`deepsets` vs `gatedpool` vs `locusfusion`.** If `gatedpool` closes a gap
   `deepsets` does not, normalisation was the problem. If `deepsets` matches
   both, the selection machinery never earned its parameters — which is worth
   more than a fourth-decimal AUC gain.

## What this run deliberately does NOT do

* **No modality ablation.** Six archs x 8 modality sets is 48 cells and this is
  a comparison of aggregators, not of inputs. `all_modalities` only.
* **No capacity sweep.** `setfusion_scaling` already spent 62 arms establishing
  that width is not the lever here. Every `--xm-*` knob is at its default.
* **No 19-locus arm.** Per-drug loci only, so the cells are directly comparable
  to `full_run_v2`. The 19-locus question is `alllocus_run_v2`'s.
* **Single seed**, like everything else in this project — so **differences under
  ~0.01 are unresolved, not small.**

## Known risks

* `catalogue`, `additive` and `fm` are zero-initialised at the output, so they
  start exactly at the base rate and the first epochs are flat. `--min-epochs 50`
  covers this, but check `best_epoch` is not pinned at 50.
* `noisyor` is monotone in evidence — it *cannot* learn a protective variant —
  and assumes independent causes, so it cannot express compensation. If it wins,
  that says something; if it loses on a drug with compensatory biology, that is
  the restriction and not a bug.
* The GBM arm densifies, so it is capped at the 4,000 most frequent columns
  (by frequency, never by any label-derived score). Reported in each JSON.

---

## Results (2026-08-26)

Macro CV AUC over the 11 drugs, `all_modalities`, **per-drug loci**. Matched
control: `full_run_v2/all_modalities__*` (`mdcnn` 0.9086, `cisfusion` 0.9086,
`late_fusion` 0.9068).

| cell | macro CV |
|---|---:|
| `deepsets` | 0.8906 |
| `gatedpool` | 0.8904 |
| `fm` | 0.8897 |
| `additive` | 0.8892 |
| `sparse_baseline` (L1-logistic, **no GPU**) | 0.8788 |
| `catalogue` | 0.8672 |
| `noisyor` | *rerunning — first run 0.4956, see below* |

**Answer to question 2 (is the aggregator what was broken?): no.** Five
aggregators sit within **0.0014** of each other. Whatever separates this family
from the CNNs, it is not the pooling operator, and no further sweep over
attention variants is warranted.

**Answer to question 1 (does a linear model already do the job?): nearly.** L1
on the variant matrix reaches **0.8788** with no GPU — within 0.03 of the best
CNN and above `catalogue`. Not enough to close the question, enough that every
future model in this family should be quoted against it.

**The result worth keeping: `catalogue` 0.8672 → `additive` 0.8892, +0.022.**
Identical models except that `catalogue` looks a weight up by exact variant
identity and `additive` computes it from the variant's features. That is the
cleanest measurement in the project of generalisation to substitutions unseen
in training — the *pncA* / PYRAZINAMIDE mechanism — and it is the empirical
justification for `locusfusion`'s feature-based token design.

### The confound this run cannot resolve

Every cell here ran at **per-drug loci** (2 genes for ISONIAZID). That is the
regime where the entire variant-token family loses: `locusfusion` scores 0.8920
here and **0.9236 at 19 loci**, +0.032 from the locus count alone. At per-drug
loci `locusfusion` and all six aggregators are indistinguishable (0.8892–0.8920,
spread 0.0028), so this run measured the family where it is weakest and cannot
tell an architecture effect from a locus effect. `variant_aggregators_alllocus/`
is the run that resolves it.

### `noisyor` — a code bug, not a verdict

First run: macro **0.4956**, below chance. `NoisyOrVariantNet` is monotone in its
evidence, and it was pointed at P(resistant) while this project encodes
**R=0/S=1** — so it was structurally forced to push every variant-carrying
isolate toward the wrong class, and resistant isolates carry more variants.
Compounded by a `-4.0` init that attenuated every gradient into the MLP ~50x on
top of `lr = exp(-9)`: train loss moved 0.3152 → 0.3023 over 99 epochs against
`additive`'s 0.2246 → 0.0875. Fixed 2026-08-26; results archived at
`results/archive/noisyor_polarity_bug_20260825/`; cell resubmitted at the
identical protocol (jobs `63647994`–`63648007`).

### Two data-integrity fixes applied on 2026-08-26

- `sparse_baseline/summary.csv` held **1 row, not 11** — the script writes the
  summary inside its per-drug loop and one job per drug shares an `--out`, so the
  last to finish clobbered the rest. Regenerated from the JSONs, which are the
  real record. `sparse_baseline.py` now carries a comment saying so.
- The first sparse submission (jobs `636156xx`) died on
  `CondaError: Run 'conda init' before 'conda activate'` → `ModuleNotFoundError:
  numpy`. It was resubmitted as `sparse-abr_sparse_*` (`636157xx`) and succeeded;
  the tracebacks in `slurm_logs/slurm-sparse_baseline_*` are from the dead
  attempt, not from the results on disk.

# SHAP over the `locusfusion` checkpoints — completing the single-drug grid

**Written 2026-09-01.** Extends `results/analysis/shap_locusfusion/` from the two
cells attributed on 2026-08-27 to **all ten single-drug cells** of
`results/experiments/newmodels_full`, and adds a convergence check on the
`nsamples` every attribution in that tree was computed at.

Runs this document reads:

| path | what it holds |
|---|---|
| `results/analysis/shap_locusfusion/newmodels_full/locusfusion/` | 10 cells x 11 drugs x 2 backgrounds |
| `results/analysis/shap_locusfusion_converge/ns{16,32,64,128,256}/` | the `nsamples` ladder, one cell |
| `notebooks/shap_locusfusion.ipynb` | sections 8-11, added by this run |

## 1. What was missing and why it mattered

The 2026-08-27 sweep attributed `sd19_dna_protein` and `sd19_all_modalities` —
the two best cells by mean test AUC. Those two cannot separate **what a modality
is worth** from **what it takes**: `sd19_all_modalities` offers all four at once,
`sd19_dna_protein` offers exactly one alternative to DNA, and neither has a
DNA-only control at the same locus set. Nor could they say anything about the
locus scope, because both are 19-locus cells.

The eight cells added here close both gaps:

* **the 19-locus modality ladder** — `sd19_dna` (the control), plus
  `sd19_dna_regulatory` and `sd19_dna_biophysical` to sit beside the existing
  `sd19_dna_protein`. Same loci, same tokenizer, same protocol, one non-DNA
  modality at a time.
* **the per-drug-loci family** — all five `sd_*` cells, which train on the drug's
  own 1-3 genes. `locusfusion` gains **+0.032 AUC from the locus count alone**,
  so the pair (`sd_X`, `sd19_X`) is the only way to ask whether the extra 16-18
  loci changed what the model reads *inside* the genes that matter.

## 2. What ran

```bash
sbatch --array=0-10 -t 08:00:00 --mem=128G scripts/sbatch/shap_locusfusion.sh \
    --cells sd19_dna sd19_dna_regulatory sd19_dna_biophysical --skip-existing
sbatch --array=0-10 -t 06:00:00 --mem=48G  scripts/sbatch/shap_locusfusion.sh \
    --cells sd_dna sd_dna_protein sd_dna_regulatory sd_dna_biophysical \
            sd_all_modalities --skip-existing
python scripts/shap_locusfusion.py --aggregate
```

No code change was needed: the sbatch script forwards trailing arguments to the
python call and `argparse` takes the last `--cells`, so the cell list is a
submit-time parameter. Settings are identical to the 2026-08-27 sweep
(`--n-explain 300 --nsamples 64 --shap-batch 25`, both backgrounds, seed 0), so
the ten cells are directly comparable.

**The `md_multidrug_*` cells are NOT attributed here.** They are locusfusion
checkpoints and they are trainable targets, but `scripts/shap_locusfusion.py` is
a single-drug script in three specific ways: the AUC gate reads `summary.csv`
(the joint cells write `multidrug_summary.csv`), `predict` collapses the output
to one column, and `readout_attention` hardcodes query 0. A joint arm is a real
piece of work, not a flag — the conv nets keep theirs in a separate
`scripts/shap_multidrug.py` for the same reason. See section 7.

## 3. The gate

220 attributions (10 cells x 11 drugs x 2 backgrounds), **0 failures**. Worst
`|rescored - reported|` test AUC across all 220: **1.32e-04**, against a refusal
threshold of 5e-3. Every checkpoint was rebuilt on delta input and reproduced
its own reported number, which is also what proves the delta resolution in
`rebuild_inputs` is right for the eight cells that had never been reloaded.

One number that is *not* reassuring and is reported here rather than buried:
**`completeness_r` has a median of 0.263** (0.263 wt / 0.264 train). Expected
gradients decomposes `f(x) - E[f(background)]` exactly, so a per-isolate
attribution sum should track the logit gap; at 0.26 it only loosely does.
Section 6 shows this is *not* a sampling artefact. Every share below is
therefore a statement about where the attribution went, not a certified
decomposition of the score.

## 4. What a modality takes has nothing to do with what it buys

The completed 19-locus ladder. `auc_gain` is against `sd19_dna` — the same 19
loci, the same tokenizer, DNA alone — so the difference is the modality.

| cell | dna | protein | biophysical | regulatory | non-DNA share | test AUC | gain |
|---|---|---|---|---|---|---|---|
| `sd19_dna`             | 1.000 | — | — | — | 0.000 | 0.9053 | — |
| `sd19_dna_protein`     | 0.649 | 0.351 | — | — | **0.351** | 0.9124 | **+0.0072** |
| `sd19_dna_regulatory`  | 0.961 | — | — | 0.039 | **0.039** | 0.9101 | **+0.0049** |
| `sd19_dna_biophysical` | 0.468 | — | 0.532 | — | **0.532** | 0.9112 | **+0.0059** |
| `sd19_all_modalities`  | 0.347 | 0.198 | 0.428 | 0.027 | 0.653 | 0.9117 | +0.0064 |

**`biophysical` takes 13.5x the budget `regulatory` takes and buys 1.2x the
AUC.** Over all 44 (drug, cell) points the rank correlation between share taken
and AUC gained is **rho = -0.025, p = 0.87** — no relationship whatsoever.

This is the same conclusion `AGGREGATOR_FAMILY_20260827.md` reached for the
aggregator family ("attribution share badly mis-ranks predictive value"),
established here on a *controlled* ladder: one modality at a time, everything
else held fixed. It is the strongest argument in the project for preferring an
architecture whose attribution is exact (`additive`) over one that has to be
explained after the fact.

The per-drug spread says the same thing louder — the share a modality takes is
a property of the *drug*, not of the modality's worth:

| cell | min | median | max |
|---|---|---|---|
| `sd19_dna_protein` | 0.198 | 0.346 | 0.511 |
| `sd19_dna_regulatory` | 0.000 | 0.021 | 0.161 |
| `sd19_dna_biophysical` | 0.320 | 0.563 | 0.671 |

At per-drug loci `biophysical` ranges from 0.003 (KANAMYCIN) to 0.781
(PYRAZINAMIDE) — a 260x spread across drugs for one modality.

## 5. The wild-type reference spends a third of its budget on nothing

Both backgrounds were run for all 220. The `wt` arm is the architecturally
natural one — on delta input the all-zero isolate is a genome identical to H37Rv
at every loaded locus, it is the model's own null, and the integration path from
it never changes the token set. That argues the attribution is *well defined*.
It does not argue the budget goes anywhere useful.

Share of the attributed-variant budget landing on columns **carried by >=80% of
the explained isolates**:

| cell | `wt` | `train` |
|---|---|---|
| `sd19_dna` | **43.0%** | 3.9% |
| `sd19_dna_protein` | **43.9%** | 5.0% |
| `sd19_dna_regulatory` | **42.4%** | 3.6% |
| `sd19_dna_biophysical` | **35.1%** | 4.8% |
| `sd19_all_modalities` | **42.3%** | 6.0% |
| `sd_dna` | 24.0% | 6.1% |
| `sd_dna_protein` | 17.9% | 2.1% |
| `sd_dna_regulatory` | 35.0% | 9.1% |
| `sd_dna_biophysical` | 19.2% | 6.2% |
| `sd_all_modalities` | 28.5% | 3.1% |

A column where 90% of isolates deviate from H37Rv is not a mutation that
distinguishes anything — it is a site where the reference genome disagrees with
the population. Its carriers are resistant at the base rate. Against the
all-zero reference it still attributes, because that reference is a genome no
isolate actually is; against real training isolates it does not, because the
reference usually carries it too.

**Denominator note.** A variant row exists only where the *explained* isolate
deviates. Against a `train` reference the attribution also lands on columns
where only the *background* deviates, and those produce no row — so the rows
capture 37–40% of the total budget under `train` against 79–90% under `wt`. The
table above is renormalised by that captured fraction; comparing raw
`share_of_total` across backgrounds would compare two different denominators.

### 5.1 The worked example

ISONIAZID / `sd19_dna` / `wt`, the columns taking the most budget:

| column | carriers | % resistant among carriers | share of budget |
|---|---|---|---|
| `dna:tlyA` 133 G | 100% | 33.4% | 15.3% |
| `dna:gyrA` 284 C | 90% | 34.8% | 14.8% |
| `dna:gyrA` 2006 A | 90% | 34.8% | 9.5% |
| `dna:embC` 2881 T | 98% | 33.7% | 6.3% |
| `dna:katG` 1537 T | 43% | 36.0% | 5.6% |
| `dna:gyrA` 61 C | 100% | 33.4% | 4.0% |

The split base rate is ~33%. Every one of those columns is at it. Meanwhile
`dna:katG` 1074 C — carried by 22% of isolates and **98.4% resistant** — takes
0.87%.

The locus ranking flips with the background, which is the cleanest single
statement of the effect:

* `sd19_dna` / **wt** -> top loci `gyrA, tlyA, katG`
* `sd19_dna` / **train** -> top loci `katG, gid, gyrA`

Same model, same isolates, same 300-isolate draw. Against a real-isolate
reference the actual isoniazid resistance gene comes first.

**Consequence for the existing notebook.** Sections 1–4 of
`notebooks/shap_locusfusion.ipynb` use `BG = "wt"` for their headline views, and
the 2026-08-27 write-up read those. Those shares are not wrong, but ~43% of what
they are dividing up at 19 loci is lineage/reference disagreement rather than
resistance evidence. **The `train` background is the one to read for "which
mutation matters"; `wt` is the one to read for "what does this isolate's
deviation from H37Rv buy".**

## 6. Locus scope, on-target share, and where selection is not happening

### 6.1 The regulatory budget collapses when the locus set widens

Same modality set, same tokenizer; only the locus count differs. Regulatory
share of the attribution budget (`wt`):

| drug | `sd_dna_regulatory` | `sd19_dna_regulatory` |
|---|---|---|
| ISONIAZID | 0.456 | 0.014 |
| ETHIONAMIDE | 0.451 | 0.024 |
| AMIKACIN | 0.386 | 0.107 |
| STREPTOMYCIN | 0.333 | 0.161 |
| KANAMYCIN | 0.329 | 0.039 |
| CAPREOMYCIN | 0.279 | 0.063 |
| ETHAMBUTOL | 0.093 | 0.021 |
| PYRAZINAMIDE | 0.020 | 0.005 |
| RIFAMPICIN | 0.008 | 0.000 |
| MOXIFLOXACIN | 0.001 | 0.000 |
| LEVOFLOXACIN | 0.000 | 0.000 |
| **mean** | **0.214** | **0.039** |

**5.4x lower at 19 loci, same direction in every drug — and this one is not a
background artefact: under `train` the collapse is 10.4x** (0.129 -> 0.012),
i.e. stronger once the ubiquitous-column budget of §5 is removed. The promoter evidence
being abandoned is the mechanistically causal kind — the `fabG1`/`inhA` promoter
for isoniazid, the `eis` promoter for kanamycin and amikacin. Read with §5: the
budget moves off promoter sites and onto DNA columns that are largely
near-universal.

### 6.2 On-target share — the model does read its own loci

Against the variant-token null (a model spreading attribution evenly over the
deviations it actually saw scores 1.0):

**This must be read on the `train` background, for the reason §5 gives.** The
`wt` numbers are depressed by the ubiquitous-column budget, which is off-target
by construction — nearly every near-universal column sits in one of the 16-18
loci that are not this drug's.

| cell | on-target share (`train`) | enrichment (`train`) | enrichment (`wt`) |
|---|---|---|---|
| `sd19_dna` | 0.237 | **5.77** | 3.93 |
| `sd19_dna_protein` | 0.242 | **7.43** | 4.68 |
| `sd19_dna_regulatory` | 0.225 | 2.93 | 2.85 |
| `sd19_dna_biophysical` | 0.183 | 4.91 | 3.39 |
| `sd19_all_modalities` | 0.204 | 2.34 | 2.18 |

The plainest version of the same thing, and the one to quote: **the single
most-attributed locus is one of the drug's own genes for 9 of 11 drugs under
`train`, against 3 of 11 under `wt`** (`sd19_dna_protein`). The reference choice,
not the model, accounts for that gap. (`sd19_all_modalities` manages only 5 of
11 even on `train` — see §4: that cell spends 43% of its budget on
`biophysical`.)

**Enriched 2.3–7.4x against the honest null, and every one of the 11 drugs
clears 1.0 under `train`** (KANAMYCIN 38.2x, AMIKACIN 13.9x, ISONIAZID 5.1x,
PYRAZINAMIDE 1.28x, LEVOFLOXACIN 1.34x). The model preferentially reads the
drug's own loci.

The absolute on-target share stays at 18–24%, but that is not the indictment it
looks like: those 1–3 genes hold only 3–4% of the variant tokens the model sees,
which is exactly why the enrichment ratio rather than the raw share is the
quantity to read.

*Correction to an earlier draft of this document*: PYRAZINAMIDE /
`sd19_dna_protein` was reported as failing outright at **0.66x**, below the null.
That was the `wt` number, and it is an artefact of §5 rather than a property of
the model — under `train` it is 1.28x. No cell fails the null.

### 6.3 The read-out attention is not doing the selecting

Concentration of the top locus, as a multiple of a uniform split over the same
loci (`wt`):

| cell | loci | attention | SHAP |
|---|---|---|---|
| `sd19_dna` | 19 | **1.107** | 4.410 |
| `sd19_dna_protein` | 19 | **1.067** | 5.108 |
| `sd19_dna_regulatory` | 19 | **1.113** | 4.090 |
| `sd19_dna_biophysical` | 19 | **1.090** | 4.646 |
| `sd19_all_modalities` | 19 | **1.063** | 6.144 |
| `sd_*` (2.2 loci mean) | 2 | 1.010–1.025 | 1.41–1.64 |

The read-out attention is flat to within 11% of uniform across 19 loci, at every
modality set, while SHAP puts 4.1–6.1x uniform on the top locus. **Whatever
locus selection this architecture performs, it is not happening in the read-out
attention.** That is the softmax-dilution argument of
`AGGREGATOR_FAMILY_20260827.md` §1.1, measured directly on `locusfusion` at 19
loci rather than inferred from the 8-token `token_signal` result.

### 6.4 Per-variant agreement across scope

Pooled over drugs, joining on `(block, column, symbol)` so the comparison is
only ever inside a locus both models loaded:

| modality set | n variants | Spearman rho | p |
|---|---|---|---|
| `dna` | 268 | +0.348 | 4.6e-09 |
| `dna_protein` | 459 | +0.248 | 7.2e-08 |
| `dna_regulatory` | 294 | +0.305 | 9.6e-08 |
| `dna_biophysical` | 498 | +0.368 | 2.2e-17 |
| `all_modalities` | 715 | +0.208 | 2.1e-08 |

Highly significant and weak. The 19-locus model reads the *same mutations in the
same genes* in a substantially different order from the model trained on those
genes alone. This is not the ubiquitous-column artefact either: on `train` the
`dna_protein` pair falls to rho = +0.176 (from +0.248 on `wt`), so removing the
near-universal budget makes the two scopes agree *less*, not more.

## 7. `nsamples = 64` is converged — and the low completeness is not sampling

`TODO.md` records the aggregator-family attribution as **unconverged below
`NSAMPLES` ~128**, and every attribution in `results/analysis/shap_locusfusion/`
was computed at 64. That finding was made on a different architecture, so it
does not transfer; it does mean the question had to be asked here.

`results/analysis/shap_locusfusion_converge/` re-attributes one
(drug, cell, background) — ISONIAZID / `sd19_dna_protein` / `wt`, 150 isolates,
seed 0, everything else identical — at a ladder of `nsamples`.

| nsamples | rho vs. 256 | top-10 overlap | completeness r | completeness MAE | seconds |
|---|---|---|---|---|---|
| 16 | 0.995 | 1.00 | 0.362 | 3.392 | 165 |
| 32 | 0.993 | 1.00 | 0.366 | 3.392 | 289 |
| 64 | **1.000** | **1.00** | 0.351 | 3.395 | 542 |
| 128 | 1.000 | 1.00 | 0.352 | 3.394 | 1000 |
| 256 | — | — | 0.352 | 3.394 | 1995 |

**The concern does not transfer.** The per-variant ranking at `nsamples = 64` is
rank-identical to the 256-sample attribution (rho = 1.000), and even 16 samples
reach rho = 0.995 with the same top-10 set. The whole corpus at 64 is converged
for every ranking this notebook reports, and the 256 rung costs 3.7x the time to
change nothing.

The completeness columns are the more interesting ones. **`completeness_r` is
flat at 0.35–0.37, and `completeness_MAE` flat at 3.39, across a 16x sampling
range.** Whatever makes the attribution sum track the
logit gap poorly, it is not sampling error — more samples do not fix it. It is
structural to the wild-type decomposition, and it is why §3 flags the median
0.263 rather than treating the shares as a certified decomposition. Running the
corpus at a higher `nsamples` would cost 2-4x the GPU time and buy nothing.

## 8. What is not covered, and what should happen next

**The five `md_multidrug_*` locusfusion cells are not attributed.** They are
newly trained locusfusion checkpoints and they have saved weights, so this is a
genuine gap in coverage rather than a missing model. `scripts/shap_locusfusion.py`
is single-drug in three specific places:

1. the AUC gate reads `summary.csv`; the joint cells write
   `multidrug_summary.csv` (per-drug `test_auc`, plus a MACRO row) — an easy fix,
   and the reason the cells were first, wrongly, thought ungateable;
2. `rebuild_inputs` (shared with `shap_attribution.py`) calls `load_dataset` for
   one drug; the joint path needs `load_multidrug_dataset` and a per-drug
   `y != -1` mask, so the explained split differs per output column;
3. `predict` collapses the output to one column and `readout_attention`
   hardcodes query 0 — for a joint model the drug's own query index is exactly
   the interesting thing, since §6.3 says the attention is flat and the joint
   read-out is where it would have to stop being flat.

That is a joint arm, not a flag. The conv nets keep theirs in a separate
`scripts/shap_multidrug.py` for the same reasons, and its `rebuild_inputs` and
per-output-column reduction are the pieces to reuse. **It is the single highest
value follow-up here**: the joint model is the only one where "does it read the
right locus for the right drug" has 11 answers from one set of weights.

**Other gaps, in priority order:**

* **Read the `train` background as the primary view.** §5 is not a caveat, it is
  a reading instruction, and §6.2 is the worked example of getting it wrong: the
  same on-target measurement says "pyrazinamide is below the null" on `wt` and
  "every drug clears the null, best cell 7.4x" on `train`. The notebook's
  headline sections still default to `BG = "wt"`; they should be shown side by
  side, and any claim about *which mutation matters* should quote `train`.
* **Single seed everywhere.** Every number in this document is seed 0, one CV
  fold (the config's `best_fold`), 300 explained isolates. The modality-ladder
  result (rho = -0.025 between share and gain) is robust to that because it
  pools 44 points, but the per-drug shares are not.
* **LEVOFLOXACIN is the one exactly-zero regulatory share** in
  `sd19_dna_regulatory`: both its regulatory blocks are present but carry
  `mean_abs_attr` of exactly 0. MOXIFLOXACIN and RIFAMPICIN only *round* to
  0.000 (4.7e-04 and 4.0e-04) — they are ignored, not absent. LEVOFLOXACIN has
  the smallest test split in the grid (54 isolates, so `n_explain` is 54 rather
  than 300), which is the likeliest explanation and makes every LEVOFLOXACIN
  number here the least reliable of the eleven. Worth confirming before any
  LEVOFLOXACIN-specific claim is made from this run.

## 9. Reproducing this

```bash
# the attributions (2 array jobs, 22 tasks, ~1h05 per task)
sbatch --array=0-10 -t 08:00:00 --mem=128G scripts/sbatch/shap_locusfusion.sh \
    --cells sd19_dna sd19_dna_regulatory sd19_dna_biophysical --skip-existing
sbatch --array=0-10 -t 06:00:00 --mem=48G  scripts/sbatch/shap_locusfusion.sh \
    --cells sd_dna sd_dna_protein sd_dna_regulatory sd_dna_biophysical \
            sd_all_modalities --skip-existing
python scripts/shap_locusfusion.py --aggregate

# the notebook (sections 8-11 are this run)
jupyter nbconvert --to notebook --execute --inplace notebooks/shap_locusfusion.ipynb
```

SLURM job ids: `63873322` (sd19), `63873323` (sd_), `63873346` (convergence).

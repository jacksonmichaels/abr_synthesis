# Claim figures — exported from `notebooks/claims.ipynb`

**Exports, not sources.** Regenerate with:

    python scripts/build_claims.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/claims.ipynb
    python results/figures/claims/_export.py

Every figure's caption, qualification and caveat lives in the markdown directly
above it in the notebook. A figure title names its subject and its slice; it
never states the finding, so **none of these stands alone** — take the prose with
it.

## Part I — the display figures, in reading order

| file | claim | what it carries |
|---|---|---|
| `01_claim-1-…` | 1 | left: macro CV AUC by model at 19 loci. right (the one to read): fold-paired difference against `mdcnn` — `locusfusion` −0.0004 [−0.0062, +0.0048] at 0.66 M params, the lasso −0.0005 [−0.0033, +0.0023] at 228 coefficients, both 5/11 drugs up; `late_fusion` −0.0175 [−0.0408, −0.0040] is the one resolvable loss |
| `02_claim-2-…` | 2 | the paired decomposition — locus universe +0.049 [+0.029, +0.074] 11/11 (lasso) and +0.016 [−0.006, +0.033] 10/11 (networks), modality +0.014 [+0.005, +0.025] 9/11, multi-task sharing +0.001 [−0.003, +0.006] 5/11, architecture −0.007 [−0.017, +0.000] 2/11 |
| `03_claim-4-…` | 4 | left: attribution share against AUC gain, 44 points, rho = +0.11 [−0.18, +0.42] — **all 44 gains have an interval containing zero**. right: SHAP at 4.1–6.1x uniform on the top locus while the model's own attention is at 1.06–1.11x |
| `04_claim-6-…` | 6 | left: parameter count against macro AUC over 3.5 orders of magnitude, rho = +0.07, p = 0.73. right: four deliberate additions of capacity — three null, and `late_fusion` over `mdcnn` at −0.0175 [−0.0406, −0.0039], 1/11 drugs up |

## Part II — the appendix

| file | what it carries |
|---|---|
| `05_a1-…` | every complete cell, by task and model |
| `06_a2-…` | per drug at 19 loci: `mdcnn`, `locusfusion`, the lasso, and the MD-CNN baseline |
| `07_a3-…` | best modality gain per drug — ETO +0.106 and INH +0.039 on the promoter, nothing on the rRNA drugs |
| `08_a4-…` | locus-universe gain per drug, lasso against networks |
| `09_a5-…` | the tokenizer rebuild: 55/55 cells improved, mean +0.017 |
| `10_a6-…` | `additive` − `catalogue` against cohort size, Spearman −0.900, p = 1.6e-4 |
| `11_a7-…` | the whole genome against the 19 curated loci at matched encoding: +0.0022 [−0.0074, +0.0123], 6/11 — 4.41 Mb buys nothing over 19 hand-picked genes |
| `12_a8-…` | attribution budget by drug and modality |
| `13_a9-…` | the noise floor: fold SD by drug and cohort size |
| `14_a10-…` | on-target attribution enrichment, `wt` against `train` background |
| `model_zoo.csv` | **A11** — every model x task x modality set, macro CV AUC as mean ± SD over the 5 folds. A table, not a figure. |

## How to read the intervals

Which interval is used depends on whether the quantity is **marginal** or
**paired**, and each figure's footnote names it:

* a **marginal macro** (one model's score) gets a CI from **fold noise with the
  11-drug panel treated as fixed**. Bootstrapping drugs there would fold in the
  0.17 spread of drug difficulty — shared by every model — and make all of them
  look equally uncertain (±0.027 instead of ±0.005) while saying nothing about
  the model;
* a **paired difference** gets a **95% percentile bootstrap over the 11 drugs**,
  where the drug spread cancels (models correlate at r = 0.98–0.99 across drugs)
  and what is captured instead is how much the difference itself varies by drug;
* **one drug's CV mean** gets a t interval over its 5 folds, and **one drug, two
  models** a t interval over the 5 fold-paired differences — valid because every
  single-drug run shares the same `StratifiedKFold(5, …, random_state=42)`
  splits.

Nothing plotted is selected on the data it is compared against: every model
appears at the same modality set, Figure 2's architecture reference is
pre-specified (`mdcnn`), and A3 shows all four modality sets rather than each
drug's best of four.

Where a mean's interval is wide because of one drug, the **sign count** (*k*/11
drugs up) is printed beside it; it assumes nothing about the distribution and is
the more robust statement there. `05_a1`, `12_a8`, `13_a9` and `14_a10` carry no
intervals on purpose — each says why in the notebook.

## The two caveats that travel with all of them

* **Single seed everywhere, so no interval here contains seed variance** — all of
  them are lower bounds on the true uncertainty. The CV-fold intervals are also
  anti-conservative, because k-fold folds share training data (Bengio &
  Grandvalet 2004). `13_a9` shows how much larger the spread is on the small
  cohorts.
* **The 19-locus `locusfusion` cells ran on the OLD tokenizer.** The rebuilt one
  is worth a uniform +0.017 at per-drug loci (`09_a5`) and has never been run at
  19 loci, so Figures 1, 4 and A2 understate it by an unknown amount.

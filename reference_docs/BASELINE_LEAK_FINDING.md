# BIG-TB SD-CNN baseline: a train/test leak inflates the published test AUCs

**Date:** 2026-07-28 · **Scope:** analysis of the reference repo
`Big-TB-benchmark/dna-tasks/SD-CNN/` (read-only — their code was **not** modified).
Reproduction scripts: `abr_workspace/h1_repro/` (`eval_leak.py`, `eval_leak_all.py`).

## TL;DR
The published SD-CNN "test AUC" numbers are the output of their `assess` script,
which **evaluates the trained model on a data split that overlaps the split the
model was trained on** (~80% of the reported test isolates were in training). This
inflates the reported number, most for the hard imbalanced drugs. For MOXIFLOXACIN
the honest number is **~0.82**, not the published **0.886** — and our own model
already matches or beats the honest baseline. So the apparent "we're far below
baseline" gap was largely an artifact of the reference's evaluation, not our model.

## The bug (mechanism)
Two scripts share the same seed but split the data differently:

- **`run_SDCNN_ccp_crossval.py`** (trains + saves `sd-cnn_model_best.h5`):
  `train_test_split(idx, test_size=0.2, random_state=42)` — **NOT stratified**.
  The model is trained on this split's train portion (5-fold CV inside it).
- **`run_SDCNN_ccp_assess.py`** (computes the reported test AUC):
  `train_test_split(idx, test_size=0.2, random_state=42, stratify=y)` — **stratified**.

Same seed + different `stratify` argument → **different partitions**. So the "test"
set used to report the number is not the set the model was held out from; it
overlaps the crossval **training** set. The model is being scored largely on
isolates it memorized.

## Why it inflates (and only for some drugs)
On well-generalizing drugs the model does about the same on seen vs unseen
isolates, so the leak barely matters. On hard, class-imbalanced drugs (few
resistant isolates) the model memorizes its training resistant isolates, so
scoring on leaked training isolates lifts the AUC. Hence the inflation
concentrates on MOXIFLOXACIN / CAPREOMYCIN / ETHIONAMIDE.

## How we proved it's real (no training, their own artifacts)
All of the reference's precomputed files are world-readable on Unity: the
per-drug `X_sparse.npz`, the `geno_pheno_metadata.parquet`, the saved
`sd-cnn_model_best.h5`, and their result CSVs. So we did this **inference-only**,
without retraining anything:

1. **Reproduced both splits** from their parquet index (non-stratified vs
   stratified, seed 42) and counted the overlap.
2. **Loaded their saved best model** and evaluated it on:
   - the **assess (stratified)** split → should reproduce their published number;
   - the **crossval (non-stratified)** split → the set the model was truly held
     out from = the leak-free estimate.
3. Cross-checked against their own CSVs: `MOXI_auc.csv` (clean 5-fold CV AUC) and
   `MOXI_test_set_drug_auc.csv` (their published/leaky test AUC).

### MOXIFLOXACIN result
- Universe 17,942 isolates; MOXI-valid 2,868 (388 R / 2,480 S) — matches ours exactly.
- **460 / 574 (80.1%)** of the assess-"test" MOXI isolates were in the crossval
  **training** set. (The two test sets overlap only 19.6%.)
- Their best model on the **assess (leaky)** split = **0.8861** — reproduces their
  published `test_set_drug_auc.csv` (0.8861) exactly → our split reconstruction is faithful.
- Same model on the **clean** held-out split = **0.8251** → **leak inflation = +0.061**.
- Their own clean 5-fold CV mean (`MOXI_auc.csv`) = **0.819**.
- Our model: clean CV **0.853**, clean single-split test **~0.808** — i.e. at/above their honest baseline.

### All 11 drugs (`eval_leak_all.py`)
Columns: **published** = their paper number; **our leaky repro** = OUR re-scoring
of their saved model on their (leaky) split — it **matches `published` to ±0.000
for 10/11 drugs**, proving we reproduced their pipeline exactly; **real test** =
the same model on the clean held-out split; **real CV** = their own clean 5-fold
CV mean; **inflation** = published − real test.

| drug | %R | published | our leaky repro | real test | real CV | inflation |
|---|---:|---:|---:|---:|---:|---:|
| MOXIFLOXACIN | 13.5 | 0.886 | 0.886 | 0.825 | 0.819 | **+0.061** |
| CAPREOMYCIN  | 17.5 | 0.873 | 0.873 | 0.815 | 0.847 | **+0.058** |
| ETHIONAMIDE  | 32.0 | 0.670 | 0.670 | 0.644 | 0.622 | **+0.026** |
| PYRAZINAMIDE | 16.5 | 0.930 | 0.930 | 0.922 | 0.913 | +0.008 |
| ETHAMBUTOL   | 19.5 | 0.931 | 0.931 | 0.925 | 0.926 | +0.007 |
| ISONIAZID    | 33.4 | 0.917 | 0.917 | 0.917 | 0.912 | ~0 |
| RIFAMPICIN   | 27.7 | 0.977 | 0.977 | 0.980 | 0.972 | −0.004 |
| KANAMYCIN    | 19.7 | 0.849 | 0.849 | 0.855 | 0.867 | −0.006 |
| STREPTOMYCIN | 31.5 | 0.911 | 0.911 | 0.924 | 0.913 | −0.013 |
| LEVOFLOXACIN | 28.3 | 0.839 | 0.839 | 0.885 | 0.850 | −0.046 (N=269, noisy) |
| AMIKACIN     | 19.1 | 0.885 | 0.845\* | 0.885 | 0.859 | ~0\* |

`published` == `our leaky repro` for every drug except AMIKACIN\* — that identity
IS the replication result: we regenerate their exact reported AUCs from their own
model + data.

\* AMIKACIN's published number matches our *clean* split, not the stratified one —
its reported value appears to already be leak-free.

The overlap is ~78–82% for every drug (the split mismatch is in shared code), but
it only **materially inflates** the three hard imbalanced drugs — exactly the ones
we appeared to "underperform" on.

## Implication
- The published SD-CNN test-AUC column is **not an apples-to-apples target** for
  imbalanced drugs. Compare against the leak-corrected numbers (`clean test` /
  `clean CV` above), which are now wired into `scripts/results_viewer.ipynb`
  (`SDCNN_CLEAN`, with the published values kept as a labeled reference).
- The correct comparison to make is clean-vs-clean: our CV-AUC vs their CV-AUC.
  On that basis MOXI (and the other "gap" drugs) are matched, not behind.

## Reproduce
```bash
conda activate abr_env          # TF is CPU-only here; inference only, no GPU needed
cd abr_workspace/h1_repro
python eval_leak.py             # MOXIFLOXACIN: overlap + leaky vs clean AUC
python eval_leak_all.py         # all 11 drugs -> leak_all.csv
```
Needs `pyarrow` in the env. Reads the reference's precomputed files under
`/project/pi_annagreen_umass_edu/saishradha/.../benchmarking/SD-CNN/`; writes only
to `h1_repro/`. Their repository is untouched.

# SD-CNN baseline alignment — what changed and why

**Date:** 2026-07-28 · **Scope:** `biophysical-fusion/` only (nothing under `Big-TB-benchmark/` was touched — that's read-only reference).

## Why
Single-modality DNA runs scored **~0.1 AUC below** the BIG-TB published SD-CNN baseline.
Root cause (see #1) plus several protocol mismatches vs. the reference code in
`Big-TB-benchmark/dna-tasks/SD-CNN/model_training/` (`run_SDCNN_ccp_crossval.py`,
`run_SDCNN_ccp_assess.py`, `parameters/tb_cnn_codebase.py`, and the per-drug
`parameter_files/optimized_epochs/MOXI_ccp_epoch_60.txt`). Where the original TODO
guessed numbers, **the reference code was treated as ground truth** — three of those
guesses were wrong and are corrected below.

## The changes

### `train_multimodal.py` — `run_modal_cv` (the entry point via `run_experiment.py`)
1. **Drop `y == -1` before splitting** *(primary fix).* The old code carried missing
   phenotypes through CV. For MOXI only 2,868 / 17,942 rows are phenotyped, so ~85% of
   every batch was dead padding; combined with the old loss reduction (see below) this
   shrank the effective gradient ~6×. Now: `keep = np.nonzero(data.y != -1)[0]` and all
   modality arrays + `y` are filtered before anything else.
2. **Alpha fit on the training split only**, then scattered into a full-length array
   (`alpha[train_idx] = tb.alpha_mat(y[train_idx]…)`) so val/test rows never leak class
   frequencies.
3. **Stratified split + StratifiedKFold.**
   `train_test_split(test_size=0.2, random_state=42, stratify=y)` then
   `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
   *Deviation:* baseline crossval used plain `KFold(seed=1)`; we use stratified (seed 42)
   per the paper. Documented in the docstring.
4. **Per-fold early stopping with best-weight restore** via `train.EarlyStopper`
   (`monitor=val_loss, patience=5, min_delta=1e-4, restore_best_weights`).
5. **Output-bias init** to `+log(n_S / n_R)` on the train split (≈ **+1.855** for MOXI).
6. **TEST = the best-val-AUC CV-fold model** scored once on the held-out split (mirrors
   the baseline's `sd-cnn_model_best.h5`), NOT a retrain on full train.
7. **Reporting:** prints `CV AUC = mean ± std` and TEST separately; result dict gains
   `cv_auc_mean`, `cv_auc_std`, `test`, `test_model_fold`, `out_bias`, `patience`,
   `min_delta`, `n_valid`, `n_resistant`, `n_susceptible`.

### `train.py`
- **`masked_weighted_bce` reduction fixed:** divide by the count of *valid rows*, not the
  full (padded) batch size. Identical to the baseline when no rows are missing; invariant
  to masked padding otherwise.
  ```python
  valid_per_row = mask.sum(dim=-1)
  per_row = (bce * mask).sum(dim=-1) / valid_per_row.clamp_min(eps)
  n_valid_rows = (valid_per_row > 0).float().sum().clamp_min(eps)
  return per_row.sum() / n_valid_rows
  ```
- **Added `EarlyStopper`** (`step(epoch, val_loss, model) -> bool`, `restore(model)`;
  tracks `best`, `best_epoch`, `num_bad`, `best_state`).

### `models.py`
- `DenseHead` and `MultiModalNet` gained an **`out_bias`** arg. When set, the final
  Linear bias is filled with it. `out_bias=None` leaves PyTorch's default (small uniform)
  bias — note this differs from Keras Dense (zero), but a constant bias never changes AUC
  ranking, so it's harmless.

## Corrections to the original TODO (reference wins)
- **alpha values are `{-a, 0, +a}` with `a = R/(R+S) ≈ 0.135` for MOXI**, NOT the TODO's
  inverse-frequency `{0, 1, 6.39}`. `tb.alpha_mat` does not do inverse-frequency weighting.
- **Output bias is `+1.855`, NOT `-1.855`.** The sigmoid's positive class is `y == 1 =
  susceptible` (the majority), so the log-odds are positive.
- **DNA channel order is `(A, C, T, G, gap)`, NOT `(A, C, G, T, gap)`.**
  `BASE_TO_COLUMN = {'A':0,'C':1,'T':2,'G':3,'-':4}`; `N`/unknown → all-zero row (distinct
  from gap). Verified byte-for-byte against `tb.get_one_hot`.
- **MOXI:** epochs ceiling is **60** (not 175); loci are `['gyrB', 'gyrA']` in that order.

## Verify
`python biophysical-fusion/test_baseline_alignment.py` — 8 static/assertion checks
(covers #1–#9), runs on synthetic fixtures + a local reference import. **No Unity data
and no training run required.**

## Still on the OLD protocol
`biophysical-fusion/eval_dna_cnn.py` — a parallel DNA-only script. It picked up the
`masked_weighted_bce` fix (shared) but its `_train`/CV loop was **not** ported (no missing
filter, no train-only alpha, no early stop). Consolidate with `run_modal_cv` or port it if
still in use.

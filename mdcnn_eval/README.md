# MD-CNN (BIG-TB multi-drug baseline) — reproduction run

Reproduces the BIG-TB **MD-CNN**: one conv-conv-pool CNN over 19 loci with a
13-output sigmoid head (one per antibiotic), trained with a masked, class-weighted
BCE so isolates missing a phenotype for a drug contribute nothing to that output.

* **Stage 1** `run_MDCNN_ccp_crossval.py` — 5-fold CV on the 80% train split,
  150 epochs/fold → `auc.csv`, `cv_split_{0..4}_auc.csv`, `history_cv_split*.csv`,
  and 5 saved models.
* **Stage 2** `run_MDCNN_ccp_eval.py` — loads the `cv_split_4` model, picks a
  per-drug threshold on the train split, scores the held-out 20% → `test_set_auc.csv`.

## Which code this is

`model_training/` is copied **verbatim** from the authors' working tree,
`/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/MTB-CNN/model_training/`
— the code that actually produced the published numbers.

The public repo copy (`Big-TB-benchmark/dna-tasks/MDCNN/`) is the same model but
**cannot run on the real data as shipped**. Differences, all in the authors' favour:

| | public repo | authors' tree (used here) |
|---|---|---|
| `locus_order` | 12 operon names (`gyrBA`, `rpoBC`, `embCAB`, …) | 19 per-gene names matching the real `aligned/*.fasta`; the operon names glob to zero files |
| phenotype index col | `index_col="Isolate"` | `index_col="New_ID"` — the FASTA record IDs are `SAMN…`/`SAMEA…` accessions, not the `Isolate` column |
| alpha matrix | indexed by fold into the **unfiltered** matrix | `alpha_matrix = alpha_matrix[indices_with_R_phenotype]` first — without this the labels are misaligned against `X` |
| param files | lack `pkl_file`, `random_seed`, `test_size` | present (script `KeyError`s without them) |

Repo-side quirks left **unpatched** (this is their baseline, not ours to improve):
`compute_drug_auc_table` computes `spec`/`sens` from a 2-D `binary_prediction`
summed across all 13 drugs, so those two columns are meaningless (they exceed 1 —
see their own `test_set_auc_cv_4_best.csv`). **`AUC` and `AUC_PR` are unaffected**
and are the columns to compare against.

## Inputs (read-only, reused verbatim)

Everything under `/project/…/MTB-CNN/model_training/pickle_files/` — the parquet
metadata, one-hot HDF5, sparse X train/test, and alpha matrix — is reused as-is so
this run sees byte-identical inputs to theirs. Every code path that would rewrite
them is guarded by `os.path.isfile` and short-circuits; `pi_annagreen` stays untouched.

* 17,942 isolates × 19 loci, longest locus `rpoC` = 4,066 bp → X is `n × 5 × 4066 × 19`
* `train_test_split(test_size=0.2, random_state=42)` → 14,353 train / 3,589 test
  (alpha matrix has exactly 14,353 rows, confirming the split matches)
* 5-fold `KFold(shuffle=True, random_state=1)` inside the train split

**No train/test leak here.** Crossval and eval derive the split with the *same*
call — unlike SD-CNN, where crossval splits without `stratify` and assess splits
with it, overlapping the two sets (~80% for MOXI). MD-CNN AUCs are honest numbers.

## Running

```bash
sbatch scripts/_sbatch_mdcnn_full.sh
```

CPU-only, 20 cores, 400 GB, `-p cpu`. This is not a downgrade: the authors' own
run (job 49009464) logged *"Could not find cuda drivers … GPU will not be used"*
and did all 5×150 epochs in **4h55m** at **226 GB peak RSS**. Their TF 2.14 env
ships cuDNN 9 while TF 2.14 needs `libcudnn.so.8`, so it never had a usable GPU.
Env is theirs too: `/work/pi_annagreen_umass_edu/saishradha/miniconda3/envs/cnn`
(py3.9, TF/keras 2.14.0, numpy 1.25.2, sparse 0.14.0, sklearn 1.3.1).

**The 4h55m figure does not transfer to this cluster.** Our run took ~45 min to
load the data and **3h31m–3h35m per fold**, so a full five-fold run plus stage 2
needs ~19 h, not 16. Use `-t 24:00:00` if you ever rerun all five.

## Resuming a partial run

Job `62593892` (`-t 16:00:00`) completed folds 0–3 and was cancelled by the wall
clock 48 epochs into fold 4, so it never wrote the aggregate `auc.csv` and never
reached stage 2. Rather than pay for four folds again:

```bash
sbatch scripts/_sbatch_mdcnn_fold4_eval.sh     # fold 4, then merge, then eval
```

Three stages: fold 4 only, then `auc.csv` assembly, then the held-out test set.
`-t 12:00:00`, roughly 2× the ~6 h it should need.

How the resume works, and why it is faithful:

- `run_MDCNN_ccp_crossval.py` already carried the authors' own resume hook — a
  hardcoded `start_cv_fold = 0` guarding a `continue` at the top of the fold
  loop. **The one change we made to that script** is to read it from the
  parameter file (`kwargs.get("start_cv_fold", 0)`), so an unchanged parameter
  file behaves byte-identically and `mdcnn_crossval_fold4.txt` — which differs
  from `mdcnn_crossval.txt` in that single key — runs fold 4 alone.
- `KFold(5, shuffle=True, random_state=1)` is deterministic, and `X` is derived
  from the same read-only parquet/sparse inputs through the same seed-42
  `train_test_split` and the same `indices_with_R_phenotype` filter. So fold 4's
  train/val partition here is the one the killed job would have used.
- Keras weight init is **not** seeded in this script, so fold 4's initialisation
  differs from whatever the killed job drew. That is equally true of the
  authors' own reruns, and our folds 0–3 land within 0.007 AUC of theirs under
  the same unseeded init — well inside their run-to-run spread.
- A resumed run starts from an empty `results` frame, so its `cv_split_4_auc.csv`
  (and the `auc.csv` it writes on exit) hold **fold 4 alone**.
  `scripts/merge_cv_auc.py` concatenates `cv_split_3_auc.csv` (cumulative folds
  0–3) with it and renumbers the index, reproducing the 65-row file a single
  uninterrupted run would have written. It refuses to write anything other than
  5 folds × 13 drugs.

## Where the reproduction stands

Folds 0–3 match the authors' published run closely — this is the replication
result, measured against their own `auc.csv` restricted to the same four folds
and our 11 shared drugs:

| | macro AUC, folds 0–3 |
|---|---|
| ours | 0.9205 |
| theirs | 0.9212 |
| **Δ** | **−0.0007** |

Largest per-drug gap is AMIKACIN at −0.0069; nine of eleven drugs are within
0.003. Fold 4 and the stage-2 test numbers are what the resume job above adds.

## Comparison target

Their published run: `/project/…/MTB-CNN/training_output/results_ccp_filter12_epoch150_sbatch_18_Nov/`
(`auc.csv` for CV, `test_set_auc_cv_4_best.csv` for test). Same params —
`filter_size 12`, `N_epochs 150`, `random_seed 1`, `test_size 0.2`.

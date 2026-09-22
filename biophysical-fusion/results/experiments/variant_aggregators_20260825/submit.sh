#!/bin/bash
# variant_aggregators_20260825 — reproduce this run.
#
# Two halves, both writing under results/experiments/variant_aggregators_20260825/:
#
#   1. the sparse baseline that is NOT a network — one CPU job per drug
#   2. the six variant-set aggregators — one GPU job per (arch x drug), 66 total
#
# Matched control: full_run_v2/all_modalities__* (same loci, same 300 epochs /
# patience 30 / --min-epochs 50 / --save-weights best, same seed). The ONLY
# differences are --arch and, for the variant-token archs, the --delta input
# encoding that --arch implies.
set -euo pipefail
cd "$(dirname "$0")/../../.."
RUN=variant_aggregators_20260825

# ---------------------------------------------------------------------------
# 1. sparse baseline: L1-logistic + gradient boosting on the variant matrix.
#    CPU only. Run this FIRST and read it before anything else in this folder —
#    if it reaches ~0.909 the rest of the run changes meaning.
# ---------------------------------------------------------------------------
# NB `sbatch --wrap` runs under /bin/sh, where `source` is not a builtin, so the
# job dies before conda activates. Hence a real script with a bash shebang.
for DRUG in ISONIAZID RIFAMPICIN ETHAMBUTOL PYRAZINAMIDE STREPTOMYCIN \
            KANAMYCIN AMIKACIN CAPREOMYCIN LEVOFLOXACIN MOXIFLOXACIN ETHIONAMIDE; do
  sbatch --job-name="abr_sparse_${DRUG}" \
    "results/experiments/${RUN}/sbatch_sparse.sh" "$DRUG"
done

# strict arm: the feature vocabulary itself restricted to variants seen in the
# TRAINING split, so not even the column set is derived from test isolates
for DRUG in ISONIAZID PYRAZINAMIDE MOXIFLOXACIN; do
  sbatch --job-name="abr_sparse_strict_${DRUG}" \
    "results/experiments/${RUN}/sbatch_sparse.sh" "$DRUG" --fit-vocab-on-train
done

# ---------------------------------------------------------------------------
# 2. the six aggregators, at the full_run_v2 method. --delta and per-locus
#    blocks are implied by --arch, so they are not passed here.
# ---------------------------------------------------------------------------
python scripts/sbatch_all_runs.py \
  --experiments all_modalities__catalogue all_modalities__additive \
                all_modalities__noisyor   all_modalities__gatedpool \
                all_modalities__deepsets  all_modalities__fm \
  --drugs all \
  --epochs 300 --patience 30 --min-epochs 50 \
  --run-prefix "${RUN}/" \
  --mem 48G --time 12:00:00

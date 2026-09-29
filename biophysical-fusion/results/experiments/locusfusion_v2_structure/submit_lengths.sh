#!/bin/bash
# locusfusion_v2 + structure at LONGER closeness decay lengths, exp(-d/L) with
# L = 20 and 30 A (the first arm, submit.sh, used the 10 A default and moved
# nothing: macro CV AUC -0.0021). At 10 A only 22% of real INH token pairs fall
# inside 0.05-0.95; at 30 A all of them do (notebooks/tokenization_v2.ipynb §6.2).
# Single-drug, per-drug loci, the same protocol, seeds and folds as submit.sh and
# the no-structure control locusfusion_v2/sd_all_modalities__locusfusion_v2.
# 2 lengths x 11 drugs = 22 jobs.
#
# Run from the project root:  bash results/experiments/locusfusion_v2_structure/submit_lengths.sh
# Dry run:                    DRY=1 bash results/experiments/locusfusion_v2_structure/submit_lengths.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"

echo "### locusfusion_v2 + structure, L = 20 and 30 A, single-drug, per-drug loci: 22 jobs"
python scripts/sbatch_all_runs.py $DRY_FLAG \
    --experiments all_modalities_structure_L20__locusfusion_v2 \
                  all_modalities_structure_L30__locusfusion_v2 --drugs all \
    --epochs 300 --patience 30 --min-epochs 50 \
    --save-weights best \
    --run-prefix "locusfusion_v2_structure/sd_" \
    --mem 64G --cpus 4 --gpus 1 --time 24:00:00 --constraint vram23
[ -z "$DRY_FLAG" ] && echo && echo "submitted. Monitor: squeue -u \$USER" || true

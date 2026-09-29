#!/bin/bash
# locusfusion_v2 WITH vs WITHOUT structure (AlphaFold C-alpha closeness between a
# locus's variant tokens, 16 channels, exp(-d/10 A); tentative -- see
# notebooks/tokenization_v2.ipynb §6). Single-drug, per-drug loci, 11 jobs.
#
# Only the WITH arm is submitted here. The WITHOUT arm is the matched control
# already run at exactly this protocol and code:
#   results/experiments/locusfusion_v2/sd_all_modalities__locusfusion_v2
# (jobs 64819342-64819354, started 2026-09-24 14:28-14:36, after the last model
# code change at 14:21; trained tok_proj 128x41 = the current base layout). Same
# flags, seed 0, cv-seed 42, so both arms see identical folds.
#
# Run from the project root:  bash results/experiments/locusfusion_v2_structure/submit.sh
# Dry run:                    DRY=1 bash results/experiments/locusfusion_v2_structure/submit.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"

echo "### locusfusion_v2 + structure, single-drug, per-drug loci: 11 jobs"
python scripts/sbatch_all_runs.py $DRY_FLAG \
    --experiments all_modalities_structure__locusfusion_v2 --drugs all \
    --epochs 300 --patience 30 --min-epochs 50 \
    --save-weights best \
    --run-prefix "locusfusion_v2_structure/sd_" \
    --mem 64G --cpus 4 --gpus 1 --time 24:00:00 --constraint vram23
[ -z "$DRY_FLAG" ] && echo && echo "submitted. Monitor: squeue -u \$USER" || true

#!/bin/bash
# locusfusion_v2 canaries — two short jobs that prove the full grid will run
# before 115 jobs are spent finding out. NOT results: 3 epochs, no weights kept.
#
# Run from the project root:  bash results/experiments/locusfusion_v2/submit_canary.sh
# Dry run:                    DRY=1 bash results/experiments/locusfusion_v2/submit_canary.sh
#
#   1. single-drug, all_modalities, ISONIAZID at all 19 loci — the heaviest
#      single-drug input, and the only arm that loads fabG1 next to inhA's
#      promoter window (a base shared ACROSS loci), on the sd19 allocation.
#   2. multi-drug, all_modalities, union of per-drug loci — the path that could
#      not be exercised interactively (the loader alone OOMs a 16 GB session), on
#      the md allocation.
#
# Pass: both logs end in "Done:", each wrote its {DRUG|multidrug}__*.json, and
# neither printed "no genome span" (every block placed in the genome).
set -euo pipefail
cd "$(dirname "$0")/../../.."
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"

SHORT=(--epochs 3 --patience 30 --min-epochs 0 --n-splits 2 --save-weights none)
LOCI=(--loci eis embA embB embC ethA ethR fabG1 gid gyrA gyrB inhA katG pncA
      rpoB rpoC rpsL rrl rrs tlyA)

echo "### canary 1: single-drug, ISONIAZID, all 19 loci"
python scripts/sbatch_all_runs.py $DRY_FLAG \
    --experiments all_modalities__locusfusion_v2 --drugs ISONIAZID \
    "${LOCI[@]}" "${SHORT[@]}" \
    --run-prefix "locusfusion_v2/canary_sd19_" \
    --mem 128G --cpus 4 --gpus 1 --time 04:00:00 --constraint vram23

echo "### canary 2: multi-drug, union of per-drug loci"
python scripts/sbatch_all_runs.py $DRY_FLAG --multidrug \
    --experiments all_modalities__locusfusion_v2 \
    "${SHORT[@]}" \
    --run-prefix "locusfusion_v2/canary_" \
    --mem 128G --cpus 6 --gpus 1 --time 06:00:00 --constraint vram23

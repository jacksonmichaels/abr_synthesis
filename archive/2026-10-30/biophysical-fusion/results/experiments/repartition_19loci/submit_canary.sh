#!/bin/bash
# repartition_19loci canary -- one drug, one new partition, both models, 3 epochs.
# Proves --cv-seed reaches the jobs before 88 of them are spent. NOT results.
#
# Run from the project root:  bash results/experiments/repartition_19loci/submit_canary.sh
# Dry run:                    DRY=1 bash .../submit_canary.sh
#
# Pass: both logs end in "Done:", both JSONs record cv_seed 101, their five
# `val_ids_sha1` fold fingerprints match each other (paired) and match the
# fingerprints StratifiedKFold(seed 101) gives on ISONIAZID (partition moved as
# asked), and the test split is unchanged.
set -euo pipefail
cd "$(dirname "$0")/../../.."
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"

LOCI=(--loci eis embA embB embC ethA ethR fabG1 gid gyrA gyrB inhA katG pncA
      rpoB rpoC rpsL rrl rrs tlyA)
SHORT=(--epochs 3 --patience 30 --min-epochs 0 --seed 0 --save-weights none)

for exp in all_modalities__locusfusion_v2 all_modalities__mdcnn; do
    echo "### canary: $exp, ISONIAZID, partition 101"
    python scripts/sbatch_all_runs.py $DRY_FLAG \
        --experiments "$exp" --drugs ISONIAZID \
        "${LOCI[@]}" "${SHORT[@]}" --cv-seed 101 \
        --run-prefix "repartition_19loci/canary_cv101_" \
        --mem 128G --cpus 4 --gpus 1 --time 04:00:00 --constraint vram23
done

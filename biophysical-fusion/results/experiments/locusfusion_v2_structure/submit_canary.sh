#!/bin/bash
# locusfusion_v2 + structure canaries -- prove that training with the 16
# AlphaFold C-alpha proximity channels runs end to end. NOT results: 3 epochs,
# 2 folds. Weights ARE kept (under this folder, not the shared weights volume) so
# the checkpoint -> config.json -> rebuild path, which carries the fold on the
# structure blocks' meta, is exercised too.
#
# Run from the project root:  bash results/experiments/locusfusion_v2_structure/submit_canary.sh
# Dry run:                    DRY=1 bash results/experiments/locusfusion_v2_structure/submit_canary.sh
#
#   1. single-drug, ISONIAZID at all 19 loci: every coding locus gets a fold, and
#      rrs / rrl (no fold) exercise the fully masked path.
#   2. multi-drug, union of per-drug loci: the md allocation path.
#
# Pass: both logs end in "Done:" with no "FAILED", each wrote its
# {DRUG|multidrug}__*.json with "structure" in the modality tag, and a model
# rebuilt from the saved config gives the saved model's logits.
set -euo pipefail
cd "$(dirname "$0")/../../.."
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"

SHORT=(--epochs 3 --patience 30 --min-epochs 0 --n-splits 2
       --save-weights best --weights-dir results/experiments/locusfusion_v2_structure/canary_weights)
LOCI=(--loci eis embA embB embC ethA ethR fabG1 gid gyrA gyrB inhA katG pncA
      rpoB rpoC rpsL rrl rrs tlyA)

echo "### canary 1: single-drug, ISONIAZID, all 19 loci, + structure"
python scripts/sbatch_all_runs.py $DRY_FLAG \
    --experiments all_modalities_structure__locusfusion_v2 --drugs ISONIAZID \
    "${LOCI[@]}" "${SHORT[@]}" \
    --run-prefix "locusfusion_v2_structure/canary_sd19_" \
    --mem 128G --cpus 4 --gpus 1 --time 04:00:00 --constraint vram23

echo "### canary 2: multi-drug, union of per-drug loci, + structure"
python scripts/sbatch_all_runs.py $DRY_FLAG --multidrug \
    --experiments all_modalities_structure__locusfusion_v2 \
    "${SHORT[@]}" \
    --run-prefix "locusfusion_v2_structure/canary_" \
    --mem 128G --cpus 6 --gpus 1 --time 06:00:00 --constraint vram23

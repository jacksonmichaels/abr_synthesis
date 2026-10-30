#!/bin/bash
# repartition_19loci -- locusfusion_v2 vs mdcnn at all 19 loci, all_modalities,
# on four more CV partitions (repeated cross-validation).
#
# Run from the project root:  bash results/experiments/repartition_19loci/submit.sh
# Dry run:                    DRY=1 bash results/experiments/repartition_19loci/submit.sh
# Subset of partitions:       CV_SEEDS="101 102" bash .../submit.sh
# Canary first:               bash results/experiments/repartition_19loci/submit_canary.sh
#
# WHY THIS EXISTS. seeds_19loci varied only the TRAINING seed on one fixed CV
# partition (StratifiedKFold seed 42), so it measured training noise (small: 0.7%
# of the test's variance) and never measured PARTITION noise -- how much the
# numbers move when different isolates land in different folds. Every §13 test is
# on that one partition and prices partition noise with the Nadeau-Bengio formula,
# unchecked. This run measures it, and tests the headline pair on 25 folds.
#
# DESIGN.
#   * Only --cv-seed varies (101 102 103 104; partition 42 already exists). The
#     held-out TEST split stays seed 42, and the training seed stays 0, so
#     seeds_19loci (training varies, partition fixed) and this run (partition
#     varies, training fixed) decompose the two noise sources.
#   * Both models share every partition, so (drug, partition, fold) is paired; each
#     result records `val_ids_sha1` per fold and compare.py checks it.
#   * Partition 42 is read from runs that already exist at seed 0:
#       locusfusion_v2  results/experiments/locusfusion_v2/sd19_all_modalities__locusfusion_v2
#       mdcnn           results/experiments/seeds_19loci/s0_all_modalities__mdcnn
#   * Protocol is byte-identical to those: 300 epochs / patience 30 /
#     --min-epochs 50 / --save-weights best, the same explicit 19 loci, and each
#     model's own control allocation.
#
# Cost: 4 partitions x 2 models x 11 drugs = 88 jobs.
#
# Excluded from results_frame via DROP_RUNS; read with compare.py.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # project root
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"
CV_SEEDS=(${CV_SEEDS:-101 102 103 104})

LOCI=(--loci eis embA embB embC ethA ethR fabG1 gid gyrA gyrB inhA katG pncA
      rpoB rpoC rpsL rrl rrs tlyA)
TRAIN=(--epochs 300 --patience 30 --min-epochs 50 --seed 0)
KEEP=(--save-weights best)

for c in "${CV_SEEDS[@]}"; do
    echo "### partition $c: locusfusion_v2 (11 jobs)"
    python scripts/sbatch_all_runs.py $DRY_FLAG \
        --experiments all_modalities__locusfusion_v2 --drugs all \
        "${LOCI[@]}" "${TRAIN[@]}" "${KEEP[@]}" --cv-seed "$c" \
        --run-prefix "repartition_19loci/cv${c}_" \
        --mem 128G --cpus 4 --gpus 1 --time 36:00:00 --constraint vram23
    echo "### partition $c: mdcnn (11 jobs)"
    python scripts/sbatch_all_runs.py $DRY_FLAG \
        --experiments all_modalities__mdcnn --drugs all \
        "${LOCI[@]}" "${TRAIN[@]}" "${KEEP[@]}" --cv-seed "$c" \
        --run-prefix "repartition_19loci/cv${c}_" \
        --mem 128G --cpus 4 --gpus 1 --time 36:00:00 --constraint vram23
done

[ -z "$DRY_FLAG" ] && echo && echo "submitted. Monitor: squeue -u \$USER" || true
echo "Results: results/experiments/repartition_19loci/cv<seed>_all_modalities__{locusfusion_v2,mdcnn}/"

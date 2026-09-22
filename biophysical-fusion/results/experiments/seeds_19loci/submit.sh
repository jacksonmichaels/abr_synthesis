#!/bin/bash
# seeds_19loci -- locusfusion vs mdcnn at all 19 loci, all_modalities, 5 seeds.
#
# Run from the project root:  bash results/experiments/seeds_19loci/submit.sh
# Dry run:                    DRY=1 bash results/experiments/seeds_19loci/submit.sh
# Subset of seeds:            SEEDS="1000 2000" bash .../submit.sh
#
# WHY THIS EXISTS. results.ipynb §17 tests locusfusion − mdcnn at 19 loci on the
# folds of ONE seed: +0.0117 over five modality sets, 10/11 drugs, raw p = 0.004,
# Holm p = 0.060 -- unresolved. The corrected t prices fold noise only; seed noise
# (init + data order) is not in it. This run measures it.
#
# DESIGN.
#   * Only --seed varies. The CV partition is StratifiedKFold(seed 42) and the test
#     split is seed 42 regardless of --seed, so fold i is the same isolates across
#     every seed and both models: (drug, seed, fold) stays fully paired.
#   * Each fold trains with seed + fold, so seeds are spaced by 1000 -- adjacent
#     integers would reuse init seeds across folds.
#   * Seed 0 is RERUN, not reused: models/ and training/ carry uncommitted edits
#     since locusfusion_fixed and alllocus_run_v2 ran, so all five seeds are on one
#     code state. Seed 0 here vs. those runs is the code-drift check.
#   * Protocol is byte-identical to the controls: 300 epochs / patience 30 /
#     --min-epochs 50 / --save-weights best, the same explicit 19 loci. Resources
#     match each control's own submit script.
#
# Cost: 5 seeds x 2 models x 11 drugs = 110 jobs, ~130 GPU-h (seed-0 runtimes:
# locusfusion 13.8 GPU-h, mdcnn 12.6 GPU-h per seed; slowest job ~5 h).
#
# Excluded from results_frame via DROP_RUNS; read with compare.py.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # project root
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"
SEEDS=(${SEEDS:-0 1000 2000 3000 4000})

LOCI=(--loci eis embA embB embC ethA ethR fabG1 gid gyrA gyrB inhA katG pncA
      rpoB rpoC rpsL rrl rrs tlyA)
TRAIN=(--epochs 300 --patience 30 --min-epochs 50)
KEEP=(--save-weights best)

for s in "${SEEDS[@]}"; do
    echo "### seed $s: locusfusion (11 jobs)"
    python scripts/sbatch_all_runs.py $DRY_FLAG \
        --experiments all_modalities__locusfusion --drugs all \
        "${LOCI[@]}" "${TRAIN[@]}" "${KEEP[@]}" --seed "$s" \
        --run-prefix "seeds_19loci/s${s}_" \
        --mem 64G --cpus 4 --gpus 1 --time 12:00:00 --constraint vram23
    echo "### seed $s: mdcnn (11 jobs)"
    python scripts/sbatch_all_runs.py $DRY_FLAG \
        --experiments all_modalities__mdcnn --drugs all \
        "${LOCI[@]}" "${TRAIN[@]}" "${KEEP[@]}" --seed "$s" \
        --run-prefix "seeds_19loci/s${s}_" \
        --mem 128G --cpus 4 --gpus 1 --time 36:00:00 --constraint vram23
done

echo
echo "submitted. Monitor: squeue -u \$USER"
echo "Results: results/experiments/seeds_19loci/s<seed>_all_modalities__{locusfusion,mdcnn}/"

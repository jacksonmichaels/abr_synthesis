#!/bin/bash
# full_run_v2 — the full architecture x modality sweep again, trained longer,
# early-stopped less aggressively, and WITH SAVED WEIGHTS.
#
# Run from the project root:  bash results/experiments/full_run_v2/submit.sh
# Dry run:                    DRY=1 bash .../submit.sh
#
# Why this exists: full_run (2026-08-04, 240 jobs) checkpointed nothing, so
# every model behind its numbers is gone. This re-runs the same grid on the same
# inputs so the project has a baseline whose models actually exist.
#
# 4 architectures x 5 modality sets = 20 cells, x 11 drugs single-drug (220 jobs)
# + 1 joint job per cell (20 jobs) = 240 jobs.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # project root
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"

# Training settings — the ONLY things that differ from full_run. Inputs are
# untouched (no --all-regulatory, no --extra-loci), so this stays a baseline
# replacement rather than a new experiment.
#   --epochs 300     full_run's 150 was a binding cap for the joint cells
#                    (best_epoch 120-148 on 40% of late_fusion folds)
#   --patience 30    full_run's 15 fired spuriously: all_modalities__mdcnn joint
#                    fold 1 stopped at epoch 36 scoring 0.8609 while its sibling
#                    folds reached 0.918
#   --min-epochs 50  warmup. SetFusionNet starts near-degenerate, sits at a flat
#                    loss ~12 epochs, and patience fired before it escaped —
#                    that is what made full_run's setfusion row (0.76-0.80) an
#                    artifact rather than a verdict. Best-weight restore still
#                    runs during warmup, so this can only help, never hurt, and
#                    it is therefore safe to apply to every architecture.
TRAIN=(--epochs 300 --patience 30 --min-epochs 50)

# --save-weights best = the fold scored on TEST, i.e. the model the reported
# numbers come from. 5.1 GB across 240 runs. Change to 'all' for every CV fold
# (25.5 GB) if you need to re-derive fold-level results — but note the shared
# volume was at 95% (1.1 TB free) when this was written.
KEEP=(--save-weights best)

echo "### single-drug: 20 cells x 11 drugs = 220 jobs"
python scripts/sbatch_all_runs.py $DRY_FLAG \
    --experiments all --drugs all \
    "${TRAIN[@]}" "${KEEP[@]}" \
    --run-prefix "full_run_v2/" \
    --mem 48G --cpus 4 --gpus 1 --time 16:00:00

echo
echo "### joint: 1 job per cell = 20 jobs"
python scripts/sbatch_all_runs.py $DRY_FLAG \
    --multidrug --experiments all \
    "${TRAIN[@]}" "${KEEP[@]}" \
    --run-prefix "full_run_v2/" \
    --mem 64G --cpus 6 --gpus 1 --time 48:00:00

echo
echo "240 jobs submitted. Monitor: squeue -u \$USER"
echo "Results: results/experiments/full_run_v2/"
echo "Weights: /project/pi_mfiterau_umass_edu/abr_model_weights/full_run_v2/"

#!/bin/bash
# shap_followup — the leave-one-out arm the SHAP attribution asked for.
#
# Run from the project root:  bash results/experiments/shap_followup/submit.sh
# Dry run:                    DRY=1 bash results/experiments/shap_followup/submit.sh
#
# 3 modality sets x 11 drugs = 33 single-drug jobs, mdcnn only. See README.md
# for what each set tests. Training settings are IDENTICAL to full_run_v2 so the
# numbers drop straight into the same table — this is an added arm of that
# ablation ladder, not a new experiment.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # project root
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"

# identical to full_run_v2/submit.sh — do not "improve" these or the comparison
# against dna / all_modalities stops being like-for-like
TRAIN=(--epochs 300 --patience 30 --min-epochs 50)
KEEP=(--save-weights best)

echo "### leave-one-out: 3 cells x 11 drugs = 33 jobs"
python scripts/sbatch_all_runs.py $DRY_FLAG \
    --experiments no_dna__mdcnn no_regulatory__mdcnn regulatory_only__mdcnn \
    --drugs all \
    "${TRAIN[@]}" "${KEEP[@]}" \
    --run-prefix "shap_followup/" \
    --mem 48G --cpus 4 --gpus 1 --time 16:00:00

echo
echo "33 jobs submitted. Monitor: squeue -u \$USER"
echo "Results: results/experiments/shap_followup/"
echo "Weights: /project/pi_mfiterau_umass_edu/abr_model_weights/shap_followup/"

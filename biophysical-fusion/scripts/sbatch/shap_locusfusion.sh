#!/bin/bash
#SBATCH --job-name=abr_shap_lf
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram23
#SBATCH -p gpu
#SBATCH -t 06:00:00
#SBATCH --array=0-10
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/shap_lf-%A_%a.out

# SHAP attribution for the trained locusfusion checkpoints, one array task per
# drug. Each task attributes the two best cells of newmodels_full --
# sd19_dna_protein (best mean test AUC, 0.9124) and sd19_all_modalities (0.9117,
# and the only cell where all four modalities can be compared) -- under both
# reference distributions, so 4 attributions per task.
#
#   sbatch scripts/sbatch/shap_locusfusion.sh            # all 11 drugs
#   sbatch --array=4 scripts/sbatch/shap_locusfusion.sh  # ISONIAZID only
#   sbatch scripts/sbatch/shap_locusfusion.sh --cells sd19_dna_protein
#
# Extra args land on the python call, so --n-explain / --nsamples / --cells /
# --background are all overridable at submit time.
#
# --mem: the load is the peak, not the SHAP. 19 loci x 17.9k isolates x every
# modality is ~31 GB of float32 and the `y != -1` slice briefly doubles it;
# this is the same 128G the sd19 training jobs asked for. The GPU side is small
# (expected gradients holds `nsamples` interpolated copies, ~120 MB), which is
# why vram23 is enough.
# -t 06:00:00: measured cost is dominated by the two dataset loads per task,
# not by the attribution.
set -euo pipefail
echo "Job ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} on $(hostname)"
source ~/.bashrc
conda activate abr_env
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion

# index -> drug, alphabetical and pinned here rather than read from
# bigtb_ref.tb.DRUG_TO_LOCI, whose order is not alphabetical and could change:
# a task id has to mean the same drug when this is resubmitted.
DRUGS=(AMIKACIN CAPREOMYCIN ETHAMBUTOL ETHIONAMIDE ISONIAZID KANAMYCIN
       LEVOFLOXACIN MOXIFLOXACIN PYRAZINAMIDE RIFAMPICIN STREPTOMYCIN)
DRUG="${DRUGS[${SLURM_ARRAY_TASK_ID:-0}]}"
echo "drug: $DRUG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/shap_locusfusion.py \
    --run newmodels_full --arch locusfusion \
    --cells sd19_dna_protein sd19_all_modalities \
    --drugs "$DRUG" \
    --background wt train \
    --n-explain 300 --nsamples 64 --shap-batch 25 \
    --device cuda "$@"

echo "Done: SHAP locusfusion $DRUG"

#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH --job-name=abr_shap_lf_v3_split
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram23
#SBATCH -p gpu
#SBATCH -t 08:00:00
#SBATCH --array=0-10%11
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/shap_lf_v3_split-%A_%a.out

# LocusFusion v3 SHAP with the biophysical attribution split into the residue-
# properties slot and the D_BIO (change-from-H37Rv) slot, by path decomposition
# of the same expected gradients (scripts/shap_locusfusion_v3_token_split.py).
# Cell sd19_all_modalities_unk, training background, one array task per drug.
# Resources mirror scripts/sbatch/shap_locusfusion_v3.sh.
#
#   sbatch --array=6 -t 02:00:00 scripts/sbatch/shap_locusfusion_v3_token_split.sh   # LEVOFLOXACIN smoke
#   sbatch --array=0-5,7-10%10 scripts/sbatch/shap_locusfusion_v3_token_split.sh     # the other 10
#   python scripts/shap_locusfusion_v3_token_split.py --aggregate                    # afterwards
#
# index -> drug: 0 AMIKACIN 1 CAPREOMYCIN 2 ETHAMBUTOL 3 ETHIONAMIDE 4 ISONIAZID
#   5 KANAMYCIN 6 LEVOFLOXACIN 7 MOXIFLOXACIN 8 PYRAZINAMIDE 9 RIFAMPICIN 10 STREPTOMYCIN
set -euo pipefail
echo "Job ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} on $(hostname) at $(date)"
source ~/.bashrc
conda activate abr_env
module list 2>&1 || true
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
which python && python --version

DATA=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot
export ABR_GENOTYPE_DIR=$DATA/genomic_data/aligned
export ABR_PHENOTYPE_CSV=$DATA/phenotype_data/master_resistance_table.csv
export ABR_GENBANK=$DATA/genbank/genome.gbff
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DRUGS=(AMIKACIN CAPREOMYCIN ETHAMBUTOL ETHIONAMIDE ISONIAZID KANAMYCIN
       LEVOFLOXACIN MOXIFLOXACIN PYRAZINAMIDE RIFAMPICIN STREPTOMYCIN)
DRUG="${DRUGS[${SLURM_ARRAY_TASK_ID:-6}]}"
echo "drug: $DRUG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/shap_locusfusion_v3_token_split.py \
    --run locusfusion_v3 --arch locusfusion_v2 \
    --cell sd19_all_modalities_unk \
    --drugs "$DRUG" \
    --background train \
    --n-explain 300 --nsamples 64 \
    --device cuda "$@"

echo "Done: SHAP token split $DRUG at $(date)"

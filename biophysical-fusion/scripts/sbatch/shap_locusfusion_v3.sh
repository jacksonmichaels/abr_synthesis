#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH --job-name=abr_shap_lf_v3
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram23
#SBATCH -p gpu
#SBATCH -t 08:00:00
#SBATCH --array=0-10%6
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/shap_lf_v3-%A_%a.out

# SHAP attribution for the LocusFusion v3 sd19 checkpoints (locusfusion_v3/
# sd19_*_unk__locusfusion_v2), one array task per drug, all five 19-locus cells
# per task off ONE shared data load. Code is imported from the frozen
# v3_snapshot by the python script; data from data_snapshot via ABR_* below.
#
#   sbatch --array=6 -t 02:00:00 scripts/sbatch/shap_locusfusion_v3.sh   # LEVOFLOXACIN smoke
#   sbatch scripts/sbatch/shap_locusfusion_v3.sh --skip-existing         # full grid
#
# Extra args land on the python call (--cells / --background / --n-explain ...).
# --mem: the v3 sd19 training jobs peaked at 42-80 GB (LEVO..INH); the load is
# the peak and is sliced to test+background without the `y != -1` copy.
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

python -u scripts/shap_locusfusion_v3.py \
    --run locusfusion_v3 --arch locusfusion_v2 \
    --cells sd19_all_modalities_unk sd19_dna_regulatory_unk sd19_dna_unk \
            sd19_dna_protein_unk sd19_dna_biophysical_unk \
    --drugs "$DRUG" \
    --background train \
    --n-explain 300 --nsamples 64 --shap-batch 25 \
    --device cuda "$@"

echo "Done: SHAP locusfusion v3 $DRUG at $(date)"

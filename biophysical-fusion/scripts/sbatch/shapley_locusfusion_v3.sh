#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH --job-name=abr_shapley_lf_v3
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram23
#SBATCH -p gpu
#SBATCH -t 03:00:00
#SBATCH --array=0-10%6
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/shapley_lf_v3-%A_%a.out

# Exact modality / called-uncalled Shapley + permutation locus Shapley for the
# LocusFusion v3 sd19 checkpoints, one array task per drug, all five cells off
# ONE shared data load (as scripts/sbatch/shap_locusfusion_v3.sh). Same 300
# explained isolates as the EG study.
#
#   sbatch --array=6 -t 01:00:00 scripts/sbatch/shapley_locusfusion_v3.sh \
#       --cells sd19_all_modalities_unk --perm-check 32 64 128 256        # LEVO smoke
#   sbatch scripts/sbatch/shapley_locusfusion_v3.sh --skip-existing       # full grid
#
# --mem: the EG tasks peaked at 46-53 GB (the shared load); no 100-row
# background is held here.
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

python -u scripts/shapley_locusfusion_v3.py \
    --run locusfusion_v3 --arch locusfusion_v2 \
    --cells sd19_all_modalities_unk sd19_dna_regulatory_unk sd19_dna_unk \
            sd19_dna_protein_unk sd19_dna_biophysical_unk \
    --drugs "$DRUG" --n-explain 300 --device cuda "$@"

echo "Done: Shapley locusfusion v3 $DRUG at $(date)"

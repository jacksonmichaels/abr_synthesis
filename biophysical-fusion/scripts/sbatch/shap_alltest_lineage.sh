#!/bin/bash
#SBATCH --job-name=abr_shap_lineage_all
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram23
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --array=0-54%8
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/shap_alltest_lineage-%A_%a.out

# SHAP (expected gradients, 100-isolate training background, 64 samples) on EVERY
# lineage-split test isolate, LocusFusion v3 sd19 checkpoints retrained on
# lineage-grouped splits (--split lineage). One array task per drug x cell
# (11 x 5 = 55), index = drug * 5 + cell, as scripts/sbatch/shap_alltest_v3.sh.
# Writes to results/analysis/shap_locusfusion_lineage_alltest/lineage_split/
# locusfusion_v2/ only.
#
#   sbatch --array=30 -t 02:00:00 scripts/sbatch/shap_alltest_lineage.sh  # smoke: LEVOFLOXACIN x all_modalities
#   sbatch scripts/sbatch/shap_alltest_lineage.sh --skip-existing          # full grid
#
# Resources, from the random-split all-test array (65716236): 99.6 GPU-h total,
# longest RIF all_modalities 7.8 h, peak RSS 88 GB (INH all_modalities) against
# a 96G request. Lineage test sets are larger (RIF 4,105 vs 3,517, EMB 3,891 vs
# 3,023), so scaled by test size the longest task is ~9.1 h and the total
# ~115 GPU-h; the per-isolate part of RSS grows with it, so 128G (what the
# lineage training used) instead of 96G, and 12 h instead of 10 h.
set -euo pipefail
echo "Job ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} on $(hostname) at $(date)"
source "$(conda info --base)/etc/profile.d/conda.sh"
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
CELLS=(sd19_all_modalities_unk sd19_dna_regulatory_unk sd19_dna_unk
       sd19_dna_protein_unk sd19_dna_biophysical_unk)
I=${SLURM_ARRAY_TASK_ID:-30}
DRUG="${DRUGS[$((I / 5))]}"
CELL="${CELLS[$((I % 5))]}"
echo "drug: $DRUG  cell: $CELL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/shap_locusfusion_v3.py --split lineage \
    --run lineage_split --arch locusfusion_v2 \
    --cells "$CELL" --drugs "$DRUG" --background train \
    --n-explain 100000 --nsamples 64 --shap-batch 25 \
    --out results/analysis/shap_locusfusion_lineage_alltest \
    --device cuda "$@"

echo "Done: SHAP all-test lineage $DRUG $CELL at $(date)"

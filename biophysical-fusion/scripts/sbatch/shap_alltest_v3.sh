#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH --job-name=abr_shap_lf_v3_all
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram23
#SBATCH -p gpu
#SBATCH -t 10:00:00
#SBATCH --array=0-54%8
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/shap_alltest_v3-%A_%a.out

# SHAP (expected gradients) on EVERY test isolate, LocusFusion v3 sd19 cells.
# One array task per drug x cell (11 x 5 = 55): at all test isolates the
# largest cells (RIF / INH all_modalities, ~3,500 isolates) take ~5 h, so cells
# are split across tasks instead of sharing one load. 10 h walltime covers the
# largest with margin. Writes to results/analysis/shap_locusfusion_v3_alltest/
# (the 300-isolate run is untouched).
#
#   sbatch --array=15 -t 03:00:00 scripts/sbatch/shap_alltest_v3.sh    # smoke: ETHIONAMIDE x all_modalities
#   sbatch scripts/sbatch/shap_alltest_v3.sh --skip-existing           # full grid
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
CELLS=(sd19_all_modalities_unk sd19_dna_regulatory_unk sd19_dna_unk
       sd19_dna_protein_unk sd19_dna_biophysical_unk)
I=${SLURM_ARRAY_TASK_ID:-15}
DRUG="${DRUGS[$((I / 5))]}"
CELL="${CELLS[$((I % 5))]}"
echo "drug: $DRUG  cell: $CELL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/shap_locusfusion_v3.py \
    --run locusfusion_v3 --arch locusfusion_v2 \
    --cells "$CELL" --drugs "$DRUG" --background train \
    --n-explain 100000 --nsamples 64 --shap-batch 25 \
    --out results/analysis/shap_locusfusion_v3_alltest \
    --device cuda "$@"

echo "Done: SHAP all-test $DRUG $CELL at $(date)"

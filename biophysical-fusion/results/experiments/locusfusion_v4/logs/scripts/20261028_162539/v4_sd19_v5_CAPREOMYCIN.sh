#!/bin/bash
#SBATCH --job-name=v4_sd19_v5_CAPREOMYCIN
#SBATCH --output=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/locusfusion_v4/logs/%x-%j.out
#SBATCH --error=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/locusfusion_v4/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=vram23
set -euo pipefail
echo "host $(hostname)  start $(date)"
module list 2>&1 || true
source ~/.bashrc
conda activate abr_env
which python && python --version
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export ABR_GENOTYPE_DIR=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/genomic_data/aligned
export ABR_PHENOTYPE_CSV=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/phenotype_data/master_resistance_table.csv
export ABR_GENBANK=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/genbank/genome.gbff
cd ../v4_snapshot/20261028_162532/biophysical-fusion
python -u -m locusfusion_v4.train --drugs CAPREOMYCIN --split-dir /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/lineage_split/splits --keep-residue --drop-synonymous --out /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/locusfusion_v4/sd19_v5 --token-cache /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/token_cache_v4 --weights-dir /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/model_weights_v4
echo "done $(date)"

#!/bin/bash
#SBATCH --job-name=v4_md_nodist
#SBATCH --output=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/locusfusion_v4/logs/%x-%j.out
#SBATCH --error=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/locusfusion_v4/logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
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
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/v4_snapshot/20261027_200637/biophysical-fusion
python -u -m locusfusion_v4.train --drugs all --split-file /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/locusfusion_v4/splits/MULTIDRUG.csv --aug-frac 0.5 --aug-lambda 0.5 --no-distance --out /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/locusfusion_v4/md_nodist --token-cache /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/token_cache_v4 --weights-dir /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/model_weights_v4
echo "done $(date)"

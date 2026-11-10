#!/bin/bash
#SBATCH --job-name=gf_size6776_AMIKACIN
#SBATCH --output=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/genomefusion/logs/%x-%j.out
#SBATCH --error=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/genomefusion/logs/%x-%j.err
#SBATCH --partition=gpu,gpu-preempt-highvram
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --constraint=vram23
#SBATCH --requeue
#SBATCH --exclude=gpu058
set -euo pipefail
echo "host $(hostname)  start $(date)"
module list 2>&1 || true
source ~/.bashrc
conda activate abr_env
which python && python --version
nvidia-smi || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export ABR_GENOTYPE_DIR=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/genomic_data/aligned
export ABR_PHENOTYPE_CSV=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/phenotype_data/master_resistance_table.csv
export ABR_GENBANK=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/genbank/genome.gbff
export ABR_WORKSPACE=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis
export ABR_GF_REFS=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/genomefusion_refs
export ABR_GF_CACHE=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/genomefusion_cache
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/gf_snapshot/20261105_024429/biophysical-fusion
python -u -m genomefusion.train --drugs AMIKACIN --locus-set size:6776 --ladder-seed 0 --max-hours 44 --batch-size 32 --out /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/genomefusion/size6776 --split-dir /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/lineage_split/splits --tokens /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/genomefusion_tokens/tokens.npz --weights-dir /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/model_weights_gf --partition-cache /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/genomefusion_cache_v2
echo "done $(date)"

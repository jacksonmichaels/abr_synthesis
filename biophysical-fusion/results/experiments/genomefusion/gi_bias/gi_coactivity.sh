#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH --job-name=gi_coactivity
#SBATCH --output=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/genomefusion/gi_bias/logs/%x-%j.out
#SBATCH --error=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/genomefusion/gi_bias/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
set -euo pipefail
echo "host $(hostname)  start $(date)"
module list 2>&1 || true
set +u; source ~/.bashrc; conda activate abr_env; set -u
which python && python --version
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export ABR_WORKSPACE=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis
export ABR_GENOTYPE_DIR=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/genomic_data/aligned
export ABR_PHENOTYPE_CSV=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/phenotype_data/master_resistance_table.csv
export ABR_GENBANK=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/genbank/genome.gbff
export ABR_GF_REFS=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/genomefusion_refs
export ABR_GF_CACHE=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/genomefusion_cache_v2
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python -u scripts/gi_coactivity.py \
  --xlsx /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/gene_interaction/41586_2026_11212_MOESM5_ESM.xlsx \
  --split-dir /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/lineage_split/splits \
  --out /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/genomefusion/gi_bias/coactivity
echo "done $(date)"

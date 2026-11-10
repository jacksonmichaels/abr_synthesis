#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -p cpu-preempt
#SBATCH -t 01:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/missingness_check-%j.out
#SBATCH --job-name=abr_missingness
# Uncalled-base (N) batch-vs-biology check for locusfusion_v3; see scripts/missingness_check.py.
set -euo pipefail
source ~/.bashrc
conda activate abr_env
export ABR_GENOTYPE_DIR=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/genomic_data/aligned ABR_PHENOTYPE_CSV=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/phenotype_data/master_resistance_table.csv ABR_GENBANK=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/genbank/genome.gbff
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python -u scripts/missingness_check.py

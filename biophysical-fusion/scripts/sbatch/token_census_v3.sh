#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH -p cpu-preempt
#SBATCH -t 01:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/token_census_v3-%j.out
#SBATCH --job-name=abr_token_census_v3
# Variant-token counts per (isolate, locus) before the 16-token cap; see scripts/token_census_v3.py.
set -euo pipefail
source ~/.bashrc
conda activate abr_env
export ABR_GENOTYPE_DIR=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/genomic_data/aligned ABR_PHENOTYPE_CSV=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/phenotype_data/master_resistance_table.csv ABR_GENBANK=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/genbank/genome.gbff
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python -u scripts/token_census_v3.py

#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -p cpu-preempt
#SBATCH -t 01:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/token_census_v1-%j.out
#SBATCH --job-name=abr_token_census_v1
# Variant-token counts per (isolate, locus) before the 16-token cap (v1, per stream); see scripts/token_census_v1.py.
set -euo pipefail
source ~/.bashrc
conda activate abr_env
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python -u scripts/token_census_v1.py

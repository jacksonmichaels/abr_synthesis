#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -p cpu-preempt
#SBATCH -t 01:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/token_census-%j.out
#SBATCH --job-name=abr_token_census
# Variant-token counts per (isolate, locus) before the 16-token cap; see scripts/token_census.py.
set -euo pipefail
source ~/.bashrc
conda activate abr_env
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python -u scripts/token_census.py

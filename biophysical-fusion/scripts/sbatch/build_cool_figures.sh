#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -p cpu-preempt
#SBATCH -t 01:30:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/build_cool_figures-%j.out
#SBATCH --job-name=abr_cool_figures
# Rebuild and execute notebooks/cool_figures.ipynb (loads real data one gene at a time).
set -euo pipefail
source ~/.bashrc
conda activate abr_env
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python scripts/build_cool_figures.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=5400 notebooks/cool_figures.ipynb
echo "Done: cool_figures"

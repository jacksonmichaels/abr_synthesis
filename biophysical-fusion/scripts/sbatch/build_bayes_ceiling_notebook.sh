#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -p cpu
#SBATCH -t 00:30:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/build_bayes_ceiling-%j.out
#SBATCH --job-name=abr_bayes_ceiling_nb
# Executes notebooks/bayes_ceiling.ipynb (built by scripts/build_bayes_ceiling.py).
set -euo pipefail
hostname; date
source ~/.bashrc
conda activate abr_env
which python && python --version
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python scripts/build_bayes_ceiling.py
jupyter nbconvert --to notebook --execute --inplace notebooks/bayes_ceiling.ipynb
date

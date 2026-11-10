#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -p cpu
#SBATCH -t 01:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/bayes_ceiling-%j.out
#SBATCH --job-name=abr_bayes_ceiling
# DNA-only AUC ceiling (genotype-group oracle), single + multi-drug lineage splits,
# per-drug loci + 19 loci, with and without promoter windows; see scripts/bayes_ceiling.py. Writes results/analysis/bayes_ceiling/.
set -euo pipefail
hostname; date
source ~/.bashrc
conda activate abr_env
which python && python --version
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python -u scripts/bayes_ceiling.py
date

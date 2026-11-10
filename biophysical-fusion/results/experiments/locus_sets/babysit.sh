#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -p cpu
#SBATCH -t 7-00:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/locus_sets/babysit-%j.log
#SBATCH --job-name=abr_locus_sets_babysit
# Supervises the locus_sets sweep; see babysit.py. Resumes from state.json.
#   sbatch results/experiments/locus_sets/babysit.sh
set -euo pipefail
echo "host $(hostname)  $(date)"
source ~/.bashrc
conda activate abr_env
which python && python --version
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/locus_sets
python -u babysit.py

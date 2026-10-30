#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -p cpu
#SBATCH -t 6-00:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/locusfusion_v3/babysit-%j.log
#SBATCH --job-name=abr_v3_babysit
# Supervises the locusfusion_v3 sweep; see babysit.py. Resumes from state.json.
#   sbatch results/experiments/locusfusion_v3/babysit.sh
set -euo pipefail
source ~/.bashrc
conda activate abr_env
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/locusfusion_v3
python -u babysit.py

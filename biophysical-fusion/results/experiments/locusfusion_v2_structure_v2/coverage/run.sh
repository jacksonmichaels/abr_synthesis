#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -p cpu
#SBATCH -t 03:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/structure_coverage-%j.out
#SBATCH --job-name=abr_struct_cov
set -euo pipefail
ls /project/pi_annagreen_umass_edu/ >/dev/null || { echo "FAILED: /project not mounted"; exit 1; }
source ~/.bashrc; conda activate abr_env
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python results/experiments/locusfusion_v2_structure_v2/coverage/coverage.py

#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -p cpu-preempt
#SBATCH -t 01:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/fetch_af_structures-%j.out
#SBATCH --job-name=abr_fetch_af

# Re-run scripts/fetch_af_structures.py (C-alpha, heavy atoms, PAE) on a node
# that has /project mounted -- it reads the H37Rv GenBank record from there.
set -euo pipefail
echo "Job: fetch AlphaFold structures  (host $(hostname))"
ls /project/pi_annagreen_umass_edu/ >/dev/null || { echo "FAILED: /project not mounted"; exit 1; }
source ~/.bashrc
conda activate abr_env
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python scripts/fetch_af_structures.py "$@"
echo "Done: fetch AlphaFold structures"

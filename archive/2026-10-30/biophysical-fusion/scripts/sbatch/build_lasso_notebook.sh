#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -p cpu
#SBATCH -t 01:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/build_lasso_notebook-%j.out
#SBATCH --job-name=abr_lasso_notebook

# Rebuild and execute notebooks/lasso.ipynb (scripts/build_lasso.py). Reads only
# result JSON/CSVs plus results_frame; no training. Re-run whenever more of the
# penalty sweep (§5.3) lands.
set -euo pipefail
echo "Job: lasso notebook  (host $(hostname))"
source ~/.bashrc
conda activate abr_env
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python scripts/build_lasso.py
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 notebooks/lasso.ipynb
echo "Done: lasso notebook"

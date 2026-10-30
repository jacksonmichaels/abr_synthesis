#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -p cpu
#SBATCH --qos=short
#SBATCH -t 00:15:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/build_shap_alltest_notebook-%j.out
#SBATCH --job-name=abr_shap_alltest_nb
# Rebuild and execute notebooks/shap_shapley_v3_alltest.ipynb (figures -> results/figures/shap_v3_alltest/).
# Safe to re-run while the SHAP / Shapley arrays are still landing.
set -euo pipefail
hostname; date
source ~/.bashrc
conda activate abr_env
module list 2>&1 || true
which python && python --version
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python scripts/build_shap_alltest_notebook.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/shap_shapley_v3_alltest.ipynb
echo "Done: shap_alltest notebook"

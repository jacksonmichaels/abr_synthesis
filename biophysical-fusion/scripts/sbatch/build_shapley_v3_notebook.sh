#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -p cpu
#SBATCH --qos=short
#SBATCH -t 00:15:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/build_shapley_v3_notebook-%j.out
#SBATCH --job-name=abr_shapley_nb
# Rebuild and execute notebooks/paper/shapley_locusfusion_v3.ipynb (figures -> results/figures/shapley_v3/).
set -euo pipefail
hostname; date
source ~/.bashrc
conda activate abr_env
which python && python --version
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python scripts/build_shapley_v3_notebook.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/paper/shapley_locusfusion_v3.ipynb
echo "Done: shapley_v3 notebook"

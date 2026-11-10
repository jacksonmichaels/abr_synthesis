#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -p cpu
#SBATCH --qos=short
#SBATCH -t 00:15:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/build_display_figures-%j.out
#SBATCH --job-name=abr_display_figures
# Rebuild and execute notebooks/paper/display_figures.ipynb (manuscript figures -> results/figures/manuscript/).
set -euo pipefail
hostname; date
source ~/.bashrc
conda activate abr_env
which python && python --version
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export ABR_MDCNN_RUN_DIR=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/mdcnn_baseline
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python scripts/build_display_figures.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 notebooks/paper/display_figures.ipynb
echo "Done: display_figures"

#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -p cpu
#SBATCH -t 01:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/build_data_fixes-%j.out
#SBATCH --job-name=abr_data_fixes_nb
# Precomputes results/analysis/data_fixes/ (scripts/data_fixes_prep.py; skipped when
# present unless PREP=1) and executes notebooks/analysis/data_fixes.ipynb
# (built by scripts/build_data_fixes.py). ONLY=prep stops after the prep.
set -euo pipefail
hostname; date
source ~/.bashrc
conda activate abr_env
which python && python --version
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
if [[ "${PREP:-0}" == 1 || ! -f results/analysis/data_fixes/alignment_scan.csv ]]; then
    python -u scripts/data_fixes_prep.py
fi
[[ "${ONLY:-}" == prep ]] && { date; exit 0; }
python scripts/build_data_fixes.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 \
    notebooks/analysis/data_fixes.ipynb
date

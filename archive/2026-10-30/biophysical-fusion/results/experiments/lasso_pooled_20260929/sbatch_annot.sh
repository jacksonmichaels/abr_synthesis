#!/bin/bash
# Build ~/abr_workspace/wholegenome_annot.npz (snpEff effect per cohort key) once,
# so the 11 drug jobs don't race to build it. Parses all 17,942 variant TSVs.
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -p cpu
#SBATCH -t 03:00:00
#SBATCH -o slurm_logs/slurm-pooled-annot-%j.out
set -euo pipefail
source ~/.bashrc
conda activate abr_env
python -u results/experiments/lasso_wholegenome_20260903/lasso_wholegenome.py \
    --build-annotation-only --procs 8 --out results/experiments/lasso_pooled_20260929

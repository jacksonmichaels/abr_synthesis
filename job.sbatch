#!/bin/bash
# Generic job wrapper: runs its arguments from the repo root in abr_env.
# Override resources on the command line, e.g.
#   sbatch --job-name=lf_PZA -p gpu --gres=gpu:1 --constraint=vram23 --mem=32G -t 12:00:00 \
#       job.sbatch python -m locusfusion.train --drugs PYRAZINAMIDE
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH --job-name=abr
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
set -euo pipefail
echo "host $(hostname)  $(date)  job ${SLURM_JOB_ID:-}"
cd "${SLURM_SUBMIT_DIR:-.}"
source ~/.bashrc
conda activate abr_env
module list 2>&1 || true
which python && python --version
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "+ $*"
"$@"
echo "done $(date)"

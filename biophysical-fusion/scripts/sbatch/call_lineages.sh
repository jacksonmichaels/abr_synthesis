#!/bin/bash
#SBATCH --job-name=abr_call_lineages
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -p cpu
#SBATCH -t 01:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/call_lineages-%j.out
# Lineage calls from the TB-Profiler barcode over the cached genome-wide PASS calls.
# Reads only ~/abr_workspace/wholegenome_cache.npz (2.7 GB) and data_snapshot; never /project.
set -euo pipefail
hostname; date
source ~/.bashrc
conda activate abr_env
which python && python --version
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python -u scripts/call_lineages.py "$@"
echo "Done: call_lineages at $(date)"

#!/bin/bash
# All three window arms for one drug. Usage:
#   sbatch sbatch_windows.sh DRUG
#
# A real script rather than `sbatch --wrap`, for the reason the parent run's
# launcher gives: --wrap runs under /bin/sh, where `source ~/.bashrc` is not a
# builtin, so conda never activates and the job dies on ModuleNotFoundError.
#
# One job per drug, not per (drug, arm): the two delta arms take 1-4 s each and
# the whole cost is the onehot arm, but all three share a 2.6 GB cache load and
# one design build, so splitting them would pay that three times to parallelise
# five seconds of work.
#
# Sized against the measured worst cell, ETHAMBUTOL / onehot-window: 16.9 k
# isolates x 5,718 columns at 1,566 nonzeros per isolate, 984 s for the 40 fits
# (8 C values x 5 folds). The 2.6 GB cache is loaded whole and the drug's row
# slice is taken from it, so memory is dominated by the cache, not the design.
# 64 G and 12 h leave the same headroom the parent run used.
#
# -p cpu, NOT cpu-preem: this run was first attempted inside an interactive
# allocation on the preemptible partition, which is exactly what these jobs
# exist to avoid.
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -p cpu
#SBATCH -t 12:00:00
#SBATCH -o slurm_logs/slurm-windows-%x-%j.out

set -euo pipefail
DRUG="$1"
RUN=results/experiments/lasso_windows_20260903
PARENT=results/experiments/lasso_wholegenome_20260903/all_drugs

source ~/.bashrc
conda activate abr_env
# No cd: submit.sh passes --chdir=<project root> and a hand-submitted job
# inherits the submission directory, which is the project root by convention.

echo "lasso top-50 windows: $DRUG (host $(hostname))"
python -u "${RUN}/lasso_windows.py" \
    --drugs "$DRUG" \
    --wholegenome "$PARENT" \
    --out "${RUN}/all_drugs"
echo "Done: $DRUG"

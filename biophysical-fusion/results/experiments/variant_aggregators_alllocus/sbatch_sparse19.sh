#!/bin/bash
# One sparse-baseline job at the 19 curated loci. Usage: sbatch sbatch_sparse19.sh DRUG
#
# Its own script rather than reusing variant_aggregators_20260825/sbatch_sparse.sh:
# that one hardcodes its own RUN and OUT, and editing it would change what that
# run's submit.sh reproduces. The python it calls is shared.
#
# A real script rather than `sbatch --wrap`: --wrap runs under /bin/sh, where
# `source ~/.bashrc` is not a builtin, so conda never activates and the job dies
# on `ModuleNotFoundError: numpy` three seconds in -- which is exactly how the
# first submission of the per-drug-loci arm failed.
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p cpu
#SBATCH -t 16:00:00
#SBATCH -o slurm_logs/slurm-sparse19-%x-%j.out

set -euo pipefail
DRUG="$1"; shift

source ~/.bashrc
conda activate abr_env
cd "${SLURM_SUBMIT_DIR:-.}"

echo "sparse baseline @19 loci: $DRUG (host $(hostname))"
python -u results/experiments/variant_aggregators_20260825/sparse_baseline.py \
    --drugs "$DRUG" \
    --loci eis embA embB embC ethA ethR fabG1 gid gyrA gyrB inhA katG pncA \
           rpoB rpoC rpsL rrl rrs tlyA \
    --out results/experiments/variant_aggregators_alllocus/sparse_baseline "$@"
echo "Done: $DRUG"

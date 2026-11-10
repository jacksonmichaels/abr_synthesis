#!/bin/bash
# One sparse-baseline job. Usage: sbatch sbatch_sparse.sh DRUG [--fit-vocab-on-train]
#
# A real script rather than `sbatch --wrap`: --wrap runs the command under /bin/sh,
# where `source ~/.bashrc` is not a builtin, so conda never activates and the job
# dies on `ModuleNotFoundError: numpy` three seconds in. The generated scripts in
# scripts/sbatch_all_runs.py have always had the bash shebang for this reason.
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -p cpu
#SBATCH -t 08:00:00
#SBATCH -o slurm_logs/slurm-sparse-%x-%j.out

set -euo pipefail
DRUG="$1"; shift
RUN=variant_aggregators_20260825
OUT="sparse_baseline"
if [[ "${1:-}" == "--fit-vocab-on-train" ]]; then OUT="sparse_baseline_trainvocab"; fi

source ~/.bashrc
conda activate abr_env
cd "${SLURM_SUBMIT_DIR:-.}"

echo "sparse baseline: $DRUG $* (host $(hostname))"
python -u "results/experiments/${RUN}/sparse_baseline.py" \
    --drugs "$DRUG" "$@" \
    --out "results/experiments/${RUN}/${OUT}"
echo "Done: $DRUG"

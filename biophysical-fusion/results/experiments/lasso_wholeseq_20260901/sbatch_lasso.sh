#!/bin/bash
# One whole-sequence lasso cell. Usage:
#   sbatch sbatch_lasso.sh DRUG ENCODING LOCUS_SET
#     ENCODING   = onehot | delta
#     LOCUS_SET  = perdrug | all
#
# A real script rather than `sbatch --wrap`: --wrap runs the command under
# /bin/sh, where `source ~/.bashrc` is not a builtin, so conda never activates
# and the job dies on `ModuleNotFoundError: numpy` three seconds in. Same reason
# the sparse baseline's launcher is a file.
#
# CPU-only by construction — there is no network here. The memory is for the
# `onehot`/`all` arm, whose design matrix is ~40 k columns x 17.9 k isolates at
# ~16 k nonzeros per isolate; every other arm is far smaller and simply does not
# use the allocation.
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -p cpu
#SBATCH -t 12:00:00
#SBATCH -o slurm_logs/slurm-lasso-%x-%j.out

set -euo pipefail
DRUG="$1"; ENC="$2"; LOCI="$3"
RUN=results/experiments/lasso_wholeseq_20260901

source ~/.bashrc
conda activate abr_env
cd "${SLURM_SUBMIT_DIR:-.}"

echo "lasso whole-sequence: $DRUG enc=$ENC loci=$LOCI (host $(hostname))"
python -u "${RUN}/lasso_wholeseq.py" \
    --drugs "$DRUG" \
    --encoding "$ENC" \
    --locus-set "$LOCI" \
    --out "${RUN}/${ENC}_${LOCI}"
echo "Done: $DRUG $ENC $LOCI"

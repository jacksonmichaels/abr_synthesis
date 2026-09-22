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
# CPU-only by construction — there is no network here. The allocation is sized
# against the measured worst cell, ISONIAZID / onehot / all 19 loci: 17,436
# isolates x 33,218 columns at 16,130 nonzeros per isolate, 11.3 GB peak RSS and
# 23 s per liblinear fit, so 40 fits (8 C values x 5 folds) plus the ~60 s build
# sits well inside 12 h. Every other cell is smaller and does not use the room.
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
# No cd: submit.sh passes --chdir=<project root>, and a hand-submitted job
# inherits the submission directory, which is the project root by convention.
# A `cd "$SLURM_SUBMIT_DIR"` here would actively UNDO --chdir and send the job
# back to whichever directory the grid happened to be launched from.

echo "lasso whole-sequence: $DRUG enc=$ENC loci=$LOCI (host $(hostname))"
python -u "${RUN}/lasso_wholeseq.py" \
    --drugs "$DRUG" \
    --encoding "$ENC" \
    --locus-set "$LOCI" \
    --out "${RUN}/${ENC}_${LOCI}"
echo "Done: $DRUG $ENC $LOCI"

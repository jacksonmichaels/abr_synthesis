#!/bin/bash
# One (drug, l1_ratio) cell of the whole-genome penalty sweep. Usage:
#   sbatch sbatch_sweep.sh DRUG L1_RATIO
# A real script, not --wrap: under /bin/sh `source ~/.bashrc` fails and conda
# never activates (see lasso_windows_20260903/sbatch_windows.sh).
# 64G as the windows run: memory is the 2.6 GB cache loaded whole, not the design.
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -p cpu
#SBATCH -t 24:00:00
#SBATCH -o slurm_logs/slurm-lassosweep-%x-%j.out

set -euo pipefail
DRUG="$1"; R="$2"
HERE=results/experiments/lasso_wholegenome_sweep_20260929
# 18 C values: the 09-03 grid's 8 plus 2 below, 2 above, and 6 filling the
# 0.03-3 band where every drug's optimum sat (all 11 picked 0.1 or 0.3).
CGRID=(0.0001 0.0003 0.001 0.003 0.01 0.03 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1 2 3 10 30)

source ~/.bashrc
conda activate abr_env
echo "whole-genome sweep: $DRUG l1_ratio=$R (host $(hostname))"
python -u results/experiments/lasso_wholegenome_20260903/lasso_wholegenome.py \
    --drugs "$DRUG" --l1-ratio "$R" --c-grid "${CGRID[@]}" --procs 2 \
    --out "${HERE}/l1r${R}"
echo "Done: $DRUG l1_ratio=$R"

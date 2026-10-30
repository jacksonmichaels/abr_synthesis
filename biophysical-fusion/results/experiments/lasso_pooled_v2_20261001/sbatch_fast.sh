#!/bin/bash
# The liblinear arms for one drug, in sequence. Usage: sbatch sbatch_fast.sh DRUG
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -p cpu
#SBATCH -t 12:00:00
#SBATCH -o slurm_logs/slurm-pooledv2-%x-%j.out
set -euo pipefail
DRUG="$1"
HERE=results/experiments/lasso_pooled_v2_20261001
S=results/experiments/lasso_wholegenome_20260903/lasso_wholegenome.py
CGRID=(0.0001 0.0003 0.001 0.003 0.01 0.03 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1 2 3 10 30)
source ~/.bashrc
conda activate abr_env
echo "pooled v2: $DRUG (host $(hostname))"
run() { local out="$1"; shift; python -u "$S" --drugs "$DRUG" --procs 2 --out "$HERE/$out" "$@"; }
run both            --pool both --c-grid "${CGRID[@]}"
run both_fmc3       --pool both --c-grid "${CGRID[@]}" --fold-min-count 3
run both_ps2        --pool both --c-grid "${CGRID[@]}" --pooled-scale 2
run both_ps4        --pool both --c-grid "${CGRID[@]}" --pooled-scale 4
run both_fmc3_ps2   --pool both --c-grid "${CGRID[@]}" --fold-min-count 3 --pooled-scale 2
# like-for-like encoding test: 19 loci, delta + site only, the parent's 8-value
# grid, against lasso_wholeseq_20260901/onehot_all
run all19_site      --pool site --restrict-loci all19
echo "Done: $DRUG"

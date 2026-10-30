#!/bin/bash
# Elastic net (l1_ratio 0.75, saga) on the fixed pooled design. Usage: sbatch sbatch_enet.sh DRUG
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -p cpu
#SBATCH -t 3-00:00:00
#SBATCH -o slurm_logs/slurm-pooledv2-%x-%j.out
set -euo pipefail
DRUG="$1"
HERE=results/experiments/lasso_pooled_v2_20261001
CGRID=(0.0001 0.0003 0.001 0.003 0.01 0.03 0.05 0.1 0.15 0.2 0.3 0.5 0.7 1 2 3 10 30)
source ~/.bashrc
conda activate abr_env
echo "pooled v2 enet: $DRUG (host $(hostname))"
python -u results/experiments/lasso_wholegenome_20260903/lasso_wholegenome.py \
    --drugs "$DRUG" --procs 2 --pool both --l1-ratio 0.75 --c-grid "${CGRID[@]}" \
    --out "$HERE/both_l1r0.75"
echo "Done: $DRUG"

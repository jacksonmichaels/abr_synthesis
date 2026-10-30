#!/bin/bash
# All four pooled arms for one drug. Usage: sbatch sbatch_pooled.sh DRUG
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -p cpu
#SBATCH -t 08:00:00
#SBATCH -o slurm_logs/slurm-pooled-%x-%j.out
set -euo pipefail
DRUG="$1"
HERE=results/experiments/lasso_pooled_20260929
S=results/experiments/lasso_wholegenome_20260903/lasso_wholegenome.py
source ~/.bashrc
conda activate abr_env
echo "pooled lasso: $DRUG (host $(hostname))"
for P in site gene both; do
  python -u "$S" --drugs "$DRUG" --pool "$P" --procs 2 --out "${HERE}/${P}"
done
python -u "$S" --drugs "$DRUG" --pool both --restrict-loci all19 --procs 2 \
    --out "${HERE}/all19_both"
echo "Done: $DRUG"

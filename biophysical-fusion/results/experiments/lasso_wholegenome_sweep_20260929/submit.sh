#!/bin/bash
# Whole-genome logistic regression: sweep of the penalty's norm mix and strength
# (2026-09-29). Parent: lasso_wholegenome_20260903 (pure L1, 8-value C grid,
# macro CV 0.9174 / test 0.9170).
#
#   l1_ratio  1 (L1), 0.75, 0.5, 0.25 (elastic net, saga), 0 (L2/ridge, liblinear)
#   C         18 values, 1e-4 .. 30 (sbatch_sweep.sh), 5 folds each
#   -> 5 ratios x 11 drugs = 55 CPU jobs, results in l1r<ratio>/, one summary.csv each.
#
# Same split (80/20, seed 42), folds, class weighting, cohort cache and metrics as
# the parent; --l1-ratio 1 calls the parent's own fit.
#
# A 2-job canary (LEVOFLOXACIN at 0.5 and 0 -- the two new code paths) goes first;
# the grid waits on it with afterok, so a broken path costs two small jobs.
#
#   bash results/experiments/lasso_wholegenome_sweep_20260929/submit.sh
#   DRY=1 bash ...   prints the sbatch lines
set -euo pipefail
cd "$(dirname "$0")/../../.."
HERE=results/experiments/lasso_wholegenome_sweep_20260929
RATIOS=(1 0.75 0.5 0.25 0)
DRUGS=(AMIKACIN CAPREOMYCIN ETHAMBUTOL ETHIONAMIDE ISONIAZID KANAMYCIN
       LEVOFLOXACIN MOXIFLOXACIN PYRAZINAMIDE RIFAMPICIN STREPTOMYCIN)
SB=(sbatch --parsable --chdir="$PWD")
run() { if [ "${DRY:-0}" = "1" ]; then echo "$*" >&2; echo 0; else "$@"; fi; }

mkdir -p slurm_logs
c1=$(run "${SB[@]}" --job-name="lassosweep-canary-LEVO-0.5" "$HERE/sbatch_sweep.sh" LEVOFLOXACIN 0.5)
c2=$(run "${SB[@]}" --job-name="lassosweep-canary-LEVO-0" "$HERE/sbatch_sweep.sh" LEVOFLOXACIN 0)
echo "canary: $c1 $c2"
n=0
for r in "${RATIOS[@]}"; do
  for d in "${DRUGS[@]}"; do
    # the canary cells ARE grid cells; don't run them twice
    if [ "$d" = LEVOFLOXACIN ] && { [ "$r" = 0.5 ] || [ "$r" = 0 ]; }; then continue; fi
    run "${SB[@]}" --dependency="afterok:$c1:$c2" --job-name="lassosweep-${d}-${r}" \
        "$HERE/sbatch_sweep.sh" "$d" "$r" > /dev/null
    n=$((n + 1))
  done
done
echo "grid: $n jobs held on the canary"

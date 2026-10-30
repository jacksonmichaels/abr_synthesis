#!/bin/bash
# The whole run: 11 drugs x 3 arms, one job per drug. Run from anywhere.
#
#   bash results/experiments/lasso_windows_20260903/submit.sh
#   bash results/experiments/lasso_windows_20260903/submit.sh RIFAMPICIN ISONIAZID
#
# With no arguments it submits all 11 drugs. Named drugs submit just those,
# which is what a rerun of a preempted or failed cell needs.
#
# Every job writes into the same all_drugs/ directory and each calls
# _write_summary at the end, so the eleven jobs race on summary.csv. That race
# is benign -- each write is a full rescan of the directory, so the last job to
# finish leaves a correct file -- but if a job is killed mid-write the fix is to
# regenerate rather than to rerun:
#
#   python -c "import sys; sys.path.insert(0, 'results/experiments/lasso_wholeseq_20260901'); \
#              import lasso_wholeseq as ws, pathlib; \
#              ws._write_summary(pathlib.Path('results/experiments/lasso_windows_20260903/all_drugs'))"
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/../../.." && pwd)"
RUN=results/experiments/lasso_windows_20260903
DRUGS=(AMIKACIN CAPREOMYCIN ETHAMBUTOL ETHIONAMIDE ISONIAZID KANAMYCIN
       LEVOFLOXACIN MOXIFLOXACIN PYRAZINAMIDE RIFAMPICIN STREPTOMYCIN)
if [ "$#" -gt 0 ]; then DRUGS=("$@"); fi

mkdir -p "${PROJECT_DIR}/slurm_logs"
for drug in "${DRUGS[@]}"; do
  sbatch --chdir="$PROJECT_DIR" \
         --job-name="windows-${drug}" \
         "${PROJECT_DIR}/${RUN}/sbatch_windows.sh" "$drug"
done

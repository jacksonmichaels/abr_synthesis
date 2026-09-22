#!/bin/bash
# The whole grid: 2 encodings x 2 locus universes x 11 drugs = 44 jobs.
# Run from the project root (biophysical-fusion/).
#
#   bash results/experiments/lasso_wholeseq_20260901/submit.sh
#
# One job per cell rather than one per arm: the cells are wildly uneven
# (LEVOFLOXACIN is 269 isolates and 94 columns, ISONIAZID on all 19 loci is
# 17.4 k x ~40 k), so batching them per arm would put every arm behind its
# slowest drug for no gain.
set -euo pipefail

# The project root is derived from this script's own location, and handed to
# every job as --chdir, so it does not matter which directory the grid is
# submitted from — the jobs always read the code and write the results under
# the checkout this file lives in.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/../../.." && pwd)"
RUN=results/experiments/lasso_wholeseq_20260901
DRUGS=(AMIKACIN CAPREOMYCIN ETHAMBUTOL ETHIONAMIDE ISONIAZID KANAMYCIN
       LEVOFLOXACIN MOXIFLOXACIN PYRAZINAMIDE RIFAMPICIN STREPTOMYCIN)

mkdir -p "${PROJECT_DIR}/slurm_logs"
for enc in onehot delta; do
  for loci in perdrug all; do
    for drug in "${DRUGS[@]}"; do
      sbatch --chdir="$PROJECT_DIR" \
             --job-name="lasso-${enc}-${loci}-${drug}" \
             "${PROJECT_DIR}/${RUN}/sbatch_lasso.sh" "$drug" "$enc" "$loci"
    done
  done
done

#!/bin/bash
# Pooled whole-genome lasso, round 2 (2026-10-01). Parent: lasso_pooled_20260929
# (pool both: macro CV 0.9265). Changes, all in lasso_wholegenome.py:
#   * site: span fix -- an indel no longer marks its unchanged anchor base
#   * 18-value C grid (1e-4..30); round 1 hit the 8-value grid's edge for LEVO/ISON
#   * --fold-min-count 3: per fold, drop columns with < 3 TRAINING carriers
#     (57% of variable positions have one carrier in the whole cohort)
#   * --pooled-scale 2 / 4: pooled columns x s = their L1 penalty / s
#   * elastic net 0.75 (saga, 3-day limit)
#   * all19_site: 19 loci, delta + site only -- the like-for-like test against
#     onehot_all (0.9241), same 8-value grid
#
# Arms -> folders: both, both_fmc3, both_ps2, both_ps4, both_fmc3_ps2, all19_site
# (sbatch_fast.sh, one job per drug) and both_l1r0.75 (sbatch_enet.sh).
# Canary: LEVOFLOXACIN fast job first; everything else waits on it (afterok).
#   bash results/experiments/lasso_pooled_v2_20261001/submit.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
HERE=results/experiments/lasso_pooled_v2_20261001
mkdir -p slurm_logs
SB=(sbatch --parsable --chdir="$PWD")
c=$("${SB[@]}" --job-name=pooledv2-LEVOFLOXACIN "$HERE/sbatch_fast.sh" LEVOFLOXACIN 2>/dev/null)
echo "canary: $c"
ids=()
for d in AMIKACIN CAPREOMYCIN ETHAMBUTOL ETHIONAMIDE ISONIAZID KANAMYCIN \
         MOXIFLOXACIN PYRAZINAMIDE RIFAMPICIN STREPTOMYCIN; do
  ids+=($("${SB[@]}" --dependency=afterok:$c --job-name="pooledv2-$d" "$HERE/sbatch_fast.sh" "$d" 2>/dev/null))
done
for d in AMIKACIN CAPREOMYCIN ETHAMBUTOL ETHIONAMIDE ISONIAZID KANAMYCIN \
         LEVOFLOXACIN MOXIFLOXACIN PYRAZINAMIDE RIFAMPICIN STREPTOMYCIN; do
  ids+=($("${SB[@]}" --dependency=afterok:$c --job-name="pooledv2-enet-$d" "$HERE/sbatch_enet.sh" "$d" 2>/dev/null))
done
echo "held on canary: ${ids[*]}"

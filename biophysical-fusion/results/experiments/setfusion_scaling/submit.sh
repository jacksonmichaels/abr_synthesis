#!/bin/bash
# setfusion_scaling — capacity sweep for SetFusionNet. See README.md in this folder.
#
# Run from anywhere:   bash results/experiments/setfusion_scaling/submit.sh <stage>
# Dry run (default):   DRY=1 bash .../submit.sh all
#
# Stages, in the order they are meant to be run. Each is a superset of nothing —
# they partition the sweep, so running all four submits every cell exactly once.
#
#   joint       62 jobs   every arm, both modality sets, ONE job per arm.
#                         The joint gap (0.79 vs 0.92) is what the sweep is
#                         about, so this stage alone answers the main question.
#   single-ab   264 jobs  axes A + B single-drug (arm x drug x modality set)
#   single-cr   220 jobs  axes C + R single-drug
#   single-x    132 jobs  axis X single-drug (D is joint-only)
#   all         678 jobs  everything, in that order
#
# Staging is deliberate: the single-drug side is 616 of the 678 jobs and answers
# the secondary question. Look at the joint stage before spending it.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # project root
STAGE="${1:-all}"
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry-run"
SWEEP=(python scripts/sweep_setfusion_scaling.py $DRY_FLAG)

case "$STAGE" in
  joint)      "${SWEEP[@]}" --scope joint ;;
  single-ab)  "${SWEEP[@]}" --scope single --axes A B ;;
  single-cr)  "${SWEEP[@]}" --scope single --axes C R ;;
  single-x)   "${SWEEP[@]}" --scope single --axes X ;;
  all)
    "${SWEEP[@]}" --scope joint
    "${SWEEP[@]}" --scope single --axes A B
    "${SWEEP[@]}" --scope single --axes C R
    "${SWEEP[@]}" --scope single --axes X
    ;;
  params)     python scripts/sweep_setfusion_scaling.py --params --write-csv ;;
  *)
    echo "unknown stage '$STAGE'"
    echo "usage: bash $0 {joint|single-ab|single-cr|single-x|all|params}"
    exit 2 ;;
esac

echo
echo "Monitor:  squeue -u \$USER   |   cancel all:  scancel -u \$USER"
echo "Weights:  /project/pi_mfiterau_umass_edu/abr_model_weights/setfusion_scaling/"

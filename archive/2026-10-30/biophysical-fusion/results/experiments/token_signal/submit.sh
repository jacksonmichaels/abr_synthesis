#!/bin/bash
# token_signal — 4 arms x 2 drugs on all_modalities__setfusion.
#
# Run from anywhere:  bash results/experiments/token_signal/submit.sh
# Dry run:            DRY=1 bash results/experiments/token_signal/submit.sh
#
# Every arm changes exactly ONE thing against a0_control, so each margin is
# attributable. Training settings are full_run_v2's, so a0 reproduces that cell
# and is the control the other three are read against.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # project root
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"

COMMON=(--experiments all_modalities__setfusion
        --drugs ISONIAZID MOXIFLOXACIN
        --epochs 300 --patience 30 --min-epochs 50
        --save-weights best --mem 64G --cpus 6 --gpus 1 --time 16:00:00)

sub () {  # sub <arm> [extra flags...]
    local arm="$1"; shift
    echo "=== submitting $arm"
    python scripts/sbatch_all_runs.py "${COMMON[@]}" $DRY_FLAG \
        --run-prefix "token_signal/${arm}_" "$@"
}

# a0: the control. Same settings as full_run_v2's cell — also confirms the new
#     flags left the defaults bit-identical.
sub a0_control

# a1: strip the locus-constant from each token so the genotype is at unit scale.
#     Targets the uniform attention and the redundant locus embedding directly.
sub a1_tokennorm  --token-norm keyed

# a2: attack the same ratio one stage upstream, at the input.
sub a2_delta      --delta

# a3: both. Expected to matter most if the sparsity a2 creates needs a1's
#     normalisation downstream to be usable at all.
sub a3_both       --token-norm keyed --delta

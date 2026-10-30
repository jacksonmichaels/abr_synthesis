#!/bin/bash
# joint_capacity — does the joint dense head have enough capacity for 11 tasks?
#
# Run from the project root:  bash results/experiments/joint_capacity/submit.sh
# Dry run:                    DRY=1 bash .../submit.sh
#
# Cells: multidrug dna_protein, late_fusion + cisfusion — the best joint
# modality set in full_run (cisfusion 0.9228, late_fusion 0.9184).
#
# EVERY arm runs at epochs=150, matching full_run exactly, so full_run's own
# cells are the controls and the ONLY difference is head capacity. Do not
# "improve" this by also raising the epoch cap: that is joint_convergence's
# variable, and changing both makes neither attributable.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # project root
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"

COMMON=(--multidrug --experiments dna_protein__late_fusion dna_protein__cisfusion
        --epochs 150 --save-weights all --mem 64G --cpus 6 --gpus 1
        --time 24:00:00)

sub () {  # sub <arm> [extra flags...]
    local arm="$1"; shift
    echo "=== submitting $arm"
    python scripts/sbatch_all_runs.py "${COMMON[@]}" $DRY_FLAG \
        --run-prefix "joint_capacity/${arm}_" "$@"
}

# b1: is the 256-d shared vector the bottleneck? All 11 drugs are read off it by
#     one linear layer; single-drug models get the same 256 for one task.
sub b1_hidden512   --hidden 512

# b2: or is it the absence of ANY per-drug capacity? Each drug gets its own
#     256 -> 64 -> 1 branch off the shared trunk. ~180k params against 46M.
sub b2_perdrug64   --per-drug-hidden 64

# b3: regularization alone — 46M params, 17.9k isolates, currently nothing.
sub b3_reg         --dropout 0.3 --weight-decay 1e-4

# b4: all three. Only interpretable against b1/b2/b3, which is why they exist.
sub b4_all         --hidden 512 --per-drug-hidden 64 --dropout 0.3 --weight-decay 1e-4

echo
echo "8 jobs submitted. Monitor: squeue -u \$USER"
echo "Weights: /project/pi_mfiterau_umass_edu/abr_model_weights/joint_capacity/"

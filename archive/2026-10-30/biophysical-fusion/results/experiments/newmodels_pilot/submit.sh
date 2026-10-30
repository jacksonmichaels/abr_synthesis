#!/bin/bash
# newmodels_pilot — does either NEW model earn a full sweep?
#
# Run from the project root:  bash results/experiments/newmodels_pilot/submit.sh
# Dry run:                    DRY=1 bash .../submit.sh
# One arm only:               ARM=locusfusion bash .../submit.sh   (or ARM=branched)
#
# Two models have never been trained on real data:
#   locusfusion — built 2026-08-25, tests pass, ZERO training runs
#   branched    — the Learn-to-Branch head added 2026-08-25 (models.BranchedHead)
# A full sweep of both is ~130 jobs and several hundred GPU-hours. This pilot is
# 7 jobs and answers whether that is worth spending, on the two cheapest cells
# that are still directly comparable to an existing control.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # project root
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"
ARM="${ARM:-both}"

# --- the full_run_v2 method, unchanged, so full_run_v2 IS the control --------
TRAIN=(--epochs 300 --patience 30 --min-epochs 50)
KEEP=(--save-weights best)

# ============================================================ locusfusion ====
# Single-drug, all modalities. Two drugs, chosen not for convenience:
#   ISONIAZID    17.4k isolates, the drug the project's core question is about,
#                and the cell token_signal's fix was measured on. Control:
#                full_run_v2/all_modalities__mdcnn = 0.9645.
#   MOXIFLOXACIN 2.9k isolates, the cell every token_signal diagnostic was run
#                on, so its collinearity numbers are directly comparable.
#                Control: full_run_v2/all_modalities__mdcnn = 0.8680.
# all_modalities is the right cell for a FIRST locusfusion run: stage 1 fuses a
# locus's modalities before the loci are fused, so a DNA-only run cannot show
# what the architecture is for.
if [ "$ARM" = "locusfusion" ] || [ "$ARM" = "both" ]; then
echo "### locusfusion: 1 cell x 2 drugs = 2 jobs"
python scripts/sbatch_all_runs.py $DRY_FLAG \
    --experiments all_modalities__locusfusion \
    --drugs ISONIAZID MOXIFLOXACIN \
    "${TRAIN[@]}" "${KEEP[@]}" \
    --run-prefix "newmodels_pilot/" \
    --mem 64G --cpus 4 --gpus 1 --time 24:00:00 --constraint vram23
fi

# =============================================================== branched ====
# Multi-drug dna__mdcnn — the cheapest joint cell (2.23 h at 300 epochs in
# full_run_v2, 3.87M params) and the one the sharing measurement was made on.
# Control: full_run_v2/multidrug_dna__mdcnn, macro CV 0.9179.
#
# TWO controls, and they are not the same thing:
#   c0_dense      the joint baseline, reproduced here at identical settings
#   c1_perdrug64  DenseHead(per_drug_hidden=64) — the SAME per-drug capacity
#                 with no routing. This is the bar branching must clear;
#                 joint_capacity's b2_perdrug64 already measured it at -0.001,
#                 so beating plain DenseHead alone proves nothing.
# Then three group counts. G=2 is the smallest split that can express anything;
# G=4 is the paper's default; G=11 would be one group per drug (= no sharing)
# and is deliberately not run.
if [ "$ARM" = "branched" ] || [ "$ARM" = "both" ]; then
BR_COMMON=(--real --device cuda --modalities dna --arch mdcnn --drugs all
           --batch-size 128 --n-splits 5 --seed 0 "${TRAIN[@]}" "${KEEP[@]}")
submit_branched () {   # $1 = arm name, rest = extra flags
    local name="$1"; shift
    local cmd="python scripts/run_multidrug.py ${BR_COMMON[*]} $* --run-name newmodels_pilot/${name}"
    if [ -n "$DRY_FLAG" ]; then echo "$cmd"; return; fi
    sbatch --job-name="pilot_${name}" --mem=64G --cpus-per-task=4 --gres=gpu:1 \
           --time=16:00:00 --constraint=vram23 \
           --output="slurm_logs/slurm-pilot_${name}-%j.out" \
           --wrap="source ~/.bashrc && conda activate abr_env && cd $PWD && \
                   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $cmd"
}
echo "### branched: 5 joint jobs (2 controls + 3 group counts)"
submit_branched c0_dense
submit_branched c1_perdrug64 --per-drug-hidden 64
submit_branched b2_branch_g2 --head branched --branch-groups 2
submit_branched b3_branch_g3 --head branched --branch-groups 3
submit_branched b4_branch_g4 --head branched --branch-groups 4
fi

echo
echo "submitted. Monitor: squeue -u \$USER"

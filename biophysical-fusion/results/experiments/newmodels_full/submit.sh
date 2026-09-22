#!/bin/bash
# newmodels_full — the two new models across the project's full grid.
#
# Run from the project root:  bash results/experiments/newmodels_full/submit.sh
# Dry run:                    DRY=1 bash .../submit.sh
# One task only:              TASK=sd|sd19|md bash .../submit.sh
#
# DO NOT SUBMIT BEFORE ../newmodels_pilot/ HAS REPORTED. The resources below are
# placeholders taken from alllocus_run_v2, and locusfusion's cost per epoch is
# unmeasured -- its input is O(variants) rather than O(sequence length), so it
# could be much cheaper or much more expensive than the conv nets. The pilot is
# what turns --mem/--time into measurements instead of guesses.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # project root
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"
TASK="${TASK:-all}"
# The branched arm is OFF by default -- see "branched: HELD" below. BRANCHED=1
# to run it anyway.
BRANCHED="${BRANCHED:-0}"

# --- the full_run_v2 method, so full_run_v2 / alllocus_run_v2 are the controls
TRAIN=(--epochs 300 --patience 30 --min-epochs 50)
KEEP=(--save-weights best)

# The five modality sets: DNA alone, DNA + each other modality, and all four.
MODSETS=(dna dna_protein dna_biophysical dna_regulatory all_modalities)

# The 19 curated loci, pinned explicitly (byte-identical to alllocus_run_v2's
# list) rather than depending on what is on disk later.
LOCI=(--loci eis embA embB embC ethA ethR fabG1 gid gyrA gyrB inhA katG pncA
      rpoB rpoC rpsL rrl rrs tlyA)

LF_CELLS=(); for m in "${MODSETS[@]}"; do LF_CELLS+=("${m}__locusfusion"); done

# ===================== locusfusion: all three tasks ==========================
# The three architectures already in full_run_v2 / alllocus_run_v2 cover all
# three tasks x all five modality sets, so locusfusion is the only model that
# needs the whole grid run.

# --- task 1: single-drug, per-drug loci (control: full_run_v2) ---------------
if [ "$TASK" = "sd" ] || [ "$TASK" = "all" ]; then
echo "### locusfusion, single-drug, per-drug loci: 5 cells x 11 drugs = 55 jobs"
python scripts/sbatch_all_runs.py $DRY_FLAG \
    --experiments "${LF_CELLS[@]}" --drugs all \
    "${TRAIN[@]}" "${KEEP[@]}" \
    --run-prefix "newmodels_full/sd_" \
    --mem 64G --cpus 4 --gpus 1 --time 24:00:00 --constraint vram23
fi

# --- task 2: single-drug, all 19 loci (control: alllocus_run_v2) -------------
if [ "$TASK" = "sd19" ] || [ "$TASK" = "all" ]; then
echo "### locusfusion, single-drug, all 19 loci: 5 cells x 11 drugs = 55 jobs"
python scripts/sbatch_all_runs.py $DRY_FLAG \
    --experiments "${LF_CELLS[@]}" --drugs all \
    "${LOCI[@]}" "${TRAIN[@]}" "${KEEP[@]}" \
    --run-prefix "newmodels_full/sd19_" \
    --mem 128G --cpus 4 --gpus 1 --time 36:00:00 --constraint vram23
fi

# --- task 3: multi-drug (control: full_run_v2 multidrug_* cells) -------------
# --multidrug submits ONE job per experiment (all 11 drugs in one model).
if [ "$TASK" = "md" ] || [ "$TASK" = "all" ]; then
echo "### locusfusion, multi-drug: 5 cells = 5 jobs"
python scripts/sbatch_all_runs.py $DRY_FLAG --multidrug \
    --experiments "${LF_CELLS[@]}" \
    "${TRAIN[@]}" "${KEEP[@]}" \
    --run-prefix "newmodels_full/md_" \
    --mem 128G --cpus 6 --gpus 1 --time 36:00:00 --constraint vram23

# ===================== branched: HELD, not submitted =========================
# BranchedHead is a MULTI-TASK head, so it has a multi-drug arm only: with one
# output there are no tasks to group, make_head returns a plain DenseHead, and
# --head branched is a documented no-op.
#
# It is HELD because the routing does not learn. Probed on a synthetic task
# built to be maximally easy for it -- 8 drugs, 4 loci, drugs 0-3 a function of
# locus 0 and 4-7 of locus 1, so the true grouping is a 4/4 split -- across six
# configurations x 3 seeds, scored by Adjusted Rand Index against that grouping
# (1.0 = recovered, 0.0 = chance):
#
#   arm                            ARI     mean |grad theta|
#   as-built (soft, tau 5)        +0.22          9.1e-04
#   + symmetry-break init         -0.08          8.2e-04
#   + theta lr x10                +0.14          6.8e-04
#   HARD routing                  -0.03          6.1e-04
#   HARD + theta lr x10           +0.05          6.0e-04
#   HARD + lr x10 + tau 1         -0.00          1.1e-03
#
# Every arm is at chance. Straight-through hard routing -- the fix that should
# have worked, since it makes a group node's gradient come only from the drugs
# routed to it -- did not move it either. The per-drug MLPs fit the task
# (loss 0.07-0.13) whichever group they route through, so theta has no signal to
# descend and stays where it started. That is the same "mechanism present but
# inert" failure token_signal diagnosed in setfusion.
#
# Consistent with the source paper, which is the point worth keeping: Branched
# CALM-Net beat unbranched CALM-Net by 0.003 F1 in the WARM-START setting
# (0.605 vs 0.602). Its entire gain was cold-start. This grid is warm-start, so
# there was never much to find here -- the cold-start arm (leave-one-drug-out,
# where the generic head is what predicts) is the experiment that would test the
# paper's actual claim, and it is not this sweep.
#
# BRANCHED=1 runs it anyway.
if [ "$BRANCHED" = "1" ]; then
echo "### branched, multi-drug: 5 modality sets x 3 base archs = 15 jobs"
for m in "${MODSETS[@]}"; do
  case "$m" in
    dna)             MODS=(dna) ;;
    dna_protein)     MODS=(dna protein) ;;
    dna_biophysical) MODS=(dna biophysical) ;;
    dna_regulatory)  MODS=(dna regulatory) ;;
    all_modalities)  MODS=(dna protein biophysical regulatory) ;;
  esac
  for a in mdcnn cisfusion late_fusion; do
    cmd="python scripts/run_multidrug.py --real --device cuda \
--modalities ${MODS[*]} --arch $a --drugs all --batch-size 128 --n-splits 5 --seed 0 \
${TRAIN[*]} ${KEEP[*]} --head branched --branch-groups 4 \
--run-name newmodels_full/md_${m}__${a}_branched"
    if [ -n "$DRY_FLAG" ]; then echo "$cmd"; continue; fi
    sbatch --job-name="br_${m}_${a}" --mem=128G --cpus-per-task=6 --gres=gpu:1 \
           --time=36:00:00 --constraint=vram23 \
           --output="slurm_logs/slurm-br_${m}_${a}-%j.out" \
           --wrap="source ~/.bashrc && conda activate abr_env && cd $PWD && \
                   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $cmd"
  done
done
else
echo "### branched: HELD (routing is inert -- see the note above). BRANCHED=1 to override."
fi
fi

echo
echo "submitted. Monitor: squeue -u \$USER"

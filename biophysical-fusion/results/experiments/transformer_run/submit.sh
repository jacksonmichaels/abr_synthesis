#!/bin/bash
# transformer_run — the full_run_v2 grid with a TRANSFORMER encoder in place of
# the CNN, at matched parameter count.
#
# Run from the project root:  bash results/experiments/transformer_run/submit.sh
# Dry run:                    DRY=1 bash .../submit.sh
#
# 3 architectures x 5 modality sets = 15 cells x 11 drugs = 165 jobs.
# full_run_v2 is the matched control: same loci, same schedule, same weights
# policy. The ONLY thing that changes is which encoder each branch/trunk uses.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # project root
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"

# --- the full_run_v2 method, unchanged --------------------------------------
TRAIN=(--epochs 300 --patience 30 --min-epochs 50)
KEEP=(--save-weights best)

# --- resources ---------------------------------------------------------------
# Raised over full_run_v2's single-drug 48G / 16 h. Attention is O(n_tokens^2)
# and the longest DNA block here is ETHAMBUTOL's embC+embA+embB (~10.1 kb ->
# ~1,120 tokens at patch 9), so both GPU memory and time per epoch are well
# above the conv trunk's. vram23 keeps the 11 GiB cards out; the driver also
# exports PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
RES=(--mem 64G --cpus 4 --gpus 1 --time 36:00:00 --constraint vram23)

# --- matched capacity, per architecture --------------------------------------
# A transformer branch is NOT parameter-comparable to a CNN branch at the
# defaults. CNNEncoder flattens, so its width scales with sequence length
# (32 * L/9 -> ~12k features on a 3.4 kb block, ~3.1M params in the dense head
# alone); TransformerEncoder mean-pools to exactly d_model=64 whatever the
# length. All-CNN vs all-transformer differ ~30x unless the knobs are raised.
#
# Shape is fixed at the standard transformer proportions -- nhead 4, layers 4,
# dim_ff = 4 * d_model, patch 9 (unchanged, so the token count matches the CNN's
# pooling factor) -- and d_model alone is solved per architecture to match THAT
# architecture's own CNN median in full_run_v2. Per-architecture, not global,
# because the comparison being made is CNN-vs-transformer *within* an
# architecture; matching across them would mis-size all three.
#
#   arch          CNN median   d_model  dim_ff   transformer median   ratio
#   late_fusion    4,450,561      208     832         4,482,977       1.05
#   cisfusion      4,517,377      160     640         4,145,825       1.00
#   mdcnn          2,491,329      176     704         3,226,737       0.99
#
# ("ratio" is the median of the per-cell-per-drug ratio, which is the honest
# figure; the median-of-medians in the column before it differs slightly.)
#
# The residual is structural and documented in README.md: per-cell ratios run
# 0.59-1.34 because CNN cost tracks total sequence length while transformer cost
# tracks BLOCK COUNT. No single config removes that.
MODS=(dna dna_protein dna_biophysical dna_regulatory all_modalities)

submit_arch () {   # $1 arch  $2 d_model  $3 dim_ff
    local arch=$1 d=$2 ff=$3 cells=()
    for m in "${MODS[@]}"; do cells+=("${m}__${arch}"); done
    echo "### ${arch}: 5 cells x 11 drugs = 55 jobs  (d_model=${d}, dim_ff=${ff})"
    python scripts/sbatch_all_runs.py $DRY_FLAG \
        --experiments "${cells[@]}" --drugs all \
        --default-encoder transformer \
        --tf-d-model "$d" --tf-nhead 4 --tf-layers 4 --tf-dim-ff "$ff" \
        "${TRAIN[@]}" "${KEEP[@]}" \
        --run-prefix "transformer_run/" \
        "${RES[@]}"
    echo
}

submit_arch late_fusion 208 832
submit_arch cisfusion   160 640
submit_arch mdcnn       176 704

echo "165 jobs submitted. Monitor: squeue -u \$USER"
echo "Results: results/experiments/transformer_run/"
echo "Weights: /project/pi_mfiterau_umass_edu/abr_model_weights/transformer_run/"
echo
echo "setfusion is NOT in this grid, on purpose: its fusion stage is ALREADY a"
echo "transformer, and its per-block encoder has its own --enc-* capacity knobs."
echo "Running it here would either duplicate full_run_v2 or confound two axes."

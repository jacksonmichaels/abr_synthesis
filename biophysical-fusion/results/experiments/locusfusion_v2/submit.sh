#!/bin/bash
# locusfusion_v2 — locusfusion with the tokenizer rebuilt.
#
# Run from the project root:  bash results/experiments/locusfusion_v2/submit.sh
# Dry run:                    DRY=1 bash .../submit.sh
#
# TWO changes since newmodels_full, and nothing else:
#
#   1. The coordinate is right. A nucleotide token used to land at column/3
#      minus a LEARNED per-locus scalar, which put katG S315 at 357.3 against
#      its own protein token's 314 (rpoB S450: 501.7 vs 449; pncA S65: 126.7 vs
#      64), and that scalar read [-0.0107, +0.0081] off the trained ISONIAZID
#      checkpoint — it never moved off its zero init. It is now computed from
#      the CDS annotation and the H37Rv gap pattern: exact, zero parameters, and
#      the same number the WHO catalogue names a mutation by.
#   2. The token is a symbol, not a 42-float slot vector. alt id + ref id +
#      codon phase + coordinate, over a 35-symbol vocabulary. The reference base
#      is new information; everything dropped was duplicated or derivable. A
#      side effect: an N call, or a premature stop, is now a variant instead of
#      an all-zero column indistinguishable from a match.
#
# CONTROLS, both already run, both at this exact protocol:
#   results/experiments/newmodels_full/sd_*__locusfusion   macro CV 0.8920
#   results/experiments/full_run_v2/*__mdcnn               macro CV 0.9086
# Same 300 epochs / patience 30 / --min-epochs 50 / --save-weights best / seed,
# same per-drug loci, same five modality sets. --arch is not even different: the
# only thing that changed is the tokenizer underneath it.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # project root
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"

TRAIN=(--epochs 300 --patience 30 --min-epochs 50)
KEEP=(--save-weights best)
MODSETS=(dna dna_protein dna_biophysical dna_regulatory all_modalities)

CELLS=(); for m in "${MODSETS[@]}"; do CELLS+=("${m}__locusfusion"); done

# Resources copied from newmodels_full's sd arm rather than re-derived. The new
# input is 5-20x leaner in host memory (int8 symbol ids where there used to be a
# float32 one-hot), so 64G is now generous — but matching the control's
# allocation keeps the only difference between the two runs the one being
# tested.
echo "### locusfusion_v2, single-drug, per-drug loci: 5 cells x 11 drugs = 55 jobs"
python scripts/sbatch_all_runs.py $DRY_FLAG \
    --experiments "${CELLS[@]}" --drugs all \
    "${TRAIN[@]}" "${KEEP[@]}" \
    --run-prefix "locusfusion_v2/sd_" \
    --mem 64G --cpus 4 --gpus 1 --time 24:00:00 --constraint vram23

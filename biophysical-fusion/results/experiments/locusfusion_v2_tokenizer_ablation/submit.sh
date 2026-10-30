#!/bin/bash
# 2x2 ablation of the 2026-09-24 48-channel locusfusion_v2 draft against the final
# 41-channel tokenizer (2026-10-01). The draft beat the final by ~0.018 macro CV
# (dna 0.8999 vs 0.8822, p = 3e-9). It differed in two features the user removed
# on 09-24; each is now an opt-in flag (models.locusfusion_v2.token_layout):
#   --lf-unknown-symbols  N channel in dna/regulatory, '-'/'X' in protein; an
#                         uncalled base makes a token                 (+4 channels)
#   --lf-ref-bio          the H37Rv residue's 3 properties             (+3)
#
#   cells (dna, single-drug, per-drug loci, 11 drugs):
#     neither   locusfusion_v2/sd41_dna__locusfusion_v2            (already run)
#     unk       -> locusfusion_v2/sd41_dna_unk__locusfusion_v2      ARMS=unk
#     refbio    -> locusfusion_v2/sd41_dna_refbio__locusfusion_v2   ARMS=refbio
#     both      -> locusfusion_v2/sd41_dna_unk_refbio__locusfusion_v2  ARMS=both (optional;
#                  the draft's own sd_dna is "both" on the DRAFT code, which differs
#                  in the deleted-codon fill rule)
#   canary: ARMS=canary (LEVOFLOXACIN, both flags, exercises every new path)
#
# Same protocol as every v2 run: 300 / patience 30 / min 50 / best weights / seed 0.
#   ARMS="canary" bash results/experiments/locusfusion_v2_tokenizer_ablation/submit.sh
#   ARMS="unk refbio" bash ...      DRY=1 for a dry run
set -euo pipefail
cd "$(dirname "$0")/../../.."
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"
ARMS="${ARMS:-unk refbio}"
TRAIN=(--epochs 300 --patience 30 --min-epochs 50 --save-weights best)
RES=(--mem 64G --cpus 4 --gpus 1 --time 24:00:00 --constraint vram23)
for arm in $ARMS; do
  case "$arm" in
    canary) EXP=dna_unk_refbio__locusfusion_v2; DRUGS=(LEVOFLOXACIN); PFX=locusfusion_v2_tokenizer_ablation/canary_ ;;
    unk)    EXP=dna_unk__locusfusion_v2;        DRUGS=(all); PFX=locusfusion_v2/sd41_ ;;
    refbio) EXP=dna_refbio__locusfusion_v2;     DRUGS=(all); PFX=locusfusion_v2/sd41_ ;;
    both)   EXP=dna_unk_refbio__locusfusion_v2; DRUGS=(all); PFX=locusfusion_v2/sd41_ ;;
    *) echo "unknown arm '$arm' (canary unk refbio both)"; exit 1 ;;
  esac
  echo "### $arm: $EXP"
  python scripts/sbatch_all_runs.py $DRY_FLAG --experiments "$EXP" --drugs "${DRUGS[@]}" \
      "${TRAIN[@]}" --run-prefix "$PFX" "${RES[@]}"
done

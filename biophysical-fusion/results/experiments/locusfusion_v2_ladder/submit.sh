#!/bin/bash
# locusfusion_v2: which added modality causes the all-modalities regression, and
# does structure help on v2's best base (2026-09-29). 66 single-drug jobs, per-drug
# loci, the protocol of locusfusion_v2/submit.sh (300 / patience 30 / min 50 /
# best weights / seed 0 / cv-seed 42), so every arm sees the controls' folds.
#
#   controls   dna, dna_protein, dna_biophysical on the FINAL tokenizer 33 jobs
#              -> locusfusion_v2/sd41_*. The sd_ runs of these three were
#              submitted 2026-09-24 14:13, mid-edit, and trained on the 48-channel
#              DRAFT (uncalled base = own channel: dna specs 6, protein 23):
#              sd_dna and sd_dna_protein all 11 drugs, sd_dna_biophysical 4 of 8
#              (3 died at import). sd_dna_regulatory, sd_all_modalities and every
#              structure / who_t12 arm are on the final code (dna 5, protein 21).
#              The draft folders are kept as they are, not overwritten.
#   ladder     dna_protein_regulatory, dna_protein_biophysical          22 jobs
#              Control: locusfusion_v2/sd41_dna_protein__locusfusion_v2.
#   structure  dna_protein + structure, 30 A                            11 jobs
#              Same control.
#
# Code since the controls trained (09-24 14:21) only adds opt-in knobs
# (--locus-set default bigtb, --lf-struct-length default 10 A).
#
# Run from the project root:  bash results/experiments/locusfusion_v2_ladder/submit.sh
# Dry run:                    DRY=1 bash results/experiments/locusfusion_v2_ladder/submit.sh
# One arm only:               ARMS="controls" bash ...   (any of: controls ladder structure)
set -euo pipefail
cd "$(dirname "$0")/../../.."
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"
ARMS="${ARMS:-controls ladder structure}"

TRAIN=(--epochs 300 --patience 30 --min-epochs 50 --save-weights best)
RES=(--mem 64G --cpus 4 --gpus 1 --time 24:00:00 --constraint vram23)

for arm in $ARMS; do
  case "$arm" in
    controls)
      echo "### dna / dna_protein / dna_biophysical, final tokenizer: 3 cells x 11 drugs = 33 jobs"
      python scripts/sbatch_all_runs.py $DRY_FLAG \
          --experiments dna__locusfusion_v2 dna_protein__locusfusion_v2 \
                        dna_biophysical__locusfusion_v2 --drugs all \
          "${TRAIN[@]}" --run-prefix "locusfusion_v2/sd41_" "${RES[@]}" ;;
    ladder)
      echo "### dna_protein + regulatory / + biophysical: 2 cells x 11 drugs = 22 jobs"
      python scripts/sbatch_all_runs.py $DRY_FLAG \
          --experiments dna_protein_regulatory__locusfusion_v2 \
                        dna_protein_biophysical__locusfusion_v2 --drugs all \
          "${TRAIN[@]}" --run-prefix "locusfusion_v2/sd_" "${RES[@]}" ;;
    structure)
      echo "### dna_protein + structure (30 A): 11 jobs"
      python scripts/sbatch_all_runs.py $DRY_FLAG \
          --experiments dna_protein_structure_L30__locusfusion_v2 --drugs all \
          "${TRAIN[@]}" --run-prefix "locusfusion_v2_structure/sd_" "${RES[@]}" ;;
    *) echo "unknown arm '$arm' (use: controls ladder structure)"; exit 1 ;;
  esac
done

[ -z "$DRY_FLAG" ] && echo && echo "submitted. Monitor: squeue -u \$USER" || true

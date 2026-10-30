#!/bin/bash
# locusfusion_v2 structure sweep v2 (2026-10-01): heavy atoms, AlphaFold PAE
# gate, and -1 for invalid pairs. Single-drug, per-drug loci, dna+protein base,
# 30 A decay, the protocol of locusfusion_v2_ladder/submit.sh (300 / patience 30 /
# min 50 / best weights / seed 0 / cv-seed 42), so every arm sees the controls' folds.
#
#   L30_neg1             C-alpha, invalid -> -1      vs. locusfusion_v2_structure/sd_dna_protein_structure_L30
#   L30_heavy_neg1       + closest heavy atoms       vs. L30_neg1
#   L30_heavy_gate_neg1  + contact gate (19.82 A C-alpha, PAE <= 10 A)  vs. L30_heavy_neg1
#   all three also vs. locusfusion_v2/sd41_dna_protein__locusfusion_v2 (no structure)
#
# KANAMYCIN is left out: its only locus is rrs (rRNA, no structure), so every arm
# would be the control. 3 arms x 10 drugs = 30 jobs.
#
# Run from the project root:  bash results/experiments/locusfusion_v2_structure_v2/submit.sh
# Dry run:                    DRY=1 bash ...
# Canary (one drug, all arms): CANARY=1 bash ...   (LEVOFLOXACIN, n=269, ~1 min/fold)
set -euo pipefail
cd "$(dirname "$0")/../../.."
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"
DRUGS=(AMIKACIN CAPREOMYCIN ETHAMBUTOL ETHIONAMIDE ISONIAZID LEVOFLOXACIN
       MOXIFLOXACIN PYRAZINAMIDE RIFAMPICIN STREPTOMYCIN)
PREFIX="locusfusion_v2_structure_v2/sd_"
if [ "${CANARY:-0}" = "1" ]; then DRUGS=(LEVOFLOXACIN); PREFIX="locusfusion_v2_structure_v2/canary_sd_"; fi

python scripts/sbatch_all_runs.py $DRY_FLAG \
    --experiments dna_protein_structure_L30_neg1__locusfusion_v2 \
                  dna_protein_structure_L30_heavy_neg1__locusfusion_v2 \
                  dna_protein_structure_L30_heavy_gate_neg1__locusfusion_v2 \
    --drugs "${DRUGS[@]}" \
    --epochs 300 --patience 30 --min-epochs 50 --save-weights best \
    --run-prefix "$PREFIX" \
    --mem 64G --cpus 4 --gpus 1 --time 24:00:00 --constraint vram23
[ -z "$DRY_FLAG" ] && echo && echo "submitted. Monitor: squeue -u \$USER" || true

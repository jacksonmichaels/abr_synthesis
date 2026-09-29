#!/bin/bash
# locusfusion_v2, all modalities, per drug, on the drug's WHO 2023 Table-21 tier 1
# + tier 2 candidate genes instead of BIG-TB's DRUG_TO_LOCI. Separate experiment:
# its own folder (results/experiments/who_t12/). 11 jobs.
#
# Loci: 6 (LFX/MFX) to 14 (INH) genes per drug. 35 of the genes had no aligned
# FASTA and were extracted from the BIG-TB VCFs by gene_msa/ (gene body +/- 100 bp,
# the same cut as the curated 19); gene_msa/who_t12_genotypes holds all 54.
# Promoter windows follow the loaded genes (the WHO Table-22 regions of any gene
# loaded), as in every run.
#
# Same protocol, seeds and folds as the control
#   results/experiments/locusfusion_v2/sd_all_modalities__locusfusion_v2
#
# Run from the project root:  bash results/experiments/who_t12/submit.sh
# Dry run:                    DRY=1 bash results/experiments/who_t12/submit.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"

echo "### locusfusion_v2, all modalities, WHO tier 1+2 genes per drug: 11 jobs"
python scripts/sbatch_all_runs.py $DRY_FLAG \
    --experiments all_modalities_who_t12__locusfusion_v2 --drugs all \
    --epochs 300 --patience 30 --min-epochs 50 \
    --save-weights best \
    --run-prefix "who_t12/sd_" \
    --mem 128G --cpus 4 --gpus 1 --time 36:00:00 --constraint vram23
[ -z "$DRY_FLAG" ] && echo && echo "submitted. Monitor: squeue -u \$USER" || true

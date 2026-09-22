#!/bin/bash
# variant_aggregators_alllocus — the six aggregators at 19 loci.
#
# Run from the project root:  bash results/experiments/variant_aggregators_alllocus/submit.sh
# Dry run:                    DRY=1 bash .../submit.sh
# One arch only:              ARCHS="additive catalogue" bash .../submit.sh
#
# DO NOT SUBMIT UNTIL THE noisyor RERUN UNDER variant_aggregators_20260825/ HAS
# LANDED. That cell is the one member of the family whose per-drug-loci number
# is not yet trustworthy, and if the polarity fix does not restore it there is no
# point spending 11 of these 66 jobs on it.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # project root
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"
RUN=variant_aggregators_alllocus

# --- the full_run_v2 / alllocus_run_v2 method, so those are the controls -----
TRAIN=(--epochs 300 --patience 30 --min-epochs 50)
KEEP=(--save-weights best)

# The 19 curated loci, pinned explicitly (byte-identical to alllocus_run_v2's
# list, and to newmodels_full's) rather than depending on what is on disk later.
LOCI=(--loci eis embA embB embC ethA ethR fabG1 gid gyrA gyrB inhA katG pncA
      rpoB rpoC rpsL rrl rrs tlyA)

ARCHS="${ARCHS:-catalogue additive noisyor gatedpool deepsets fm}"
CELLS=(); for a in $ARCHS; do CELLS+=("all_modalities__${a}"); done

# --delta and per-locus blocks are implied by --arch, so they are not passed.
# 19 loci is ~9x the token count of the per-drug cells (median 14 variants per
# isolate against ~2), hence the memory and time bump over the 48G/12h the
# per-drug run used -- taken from newmodels_full's sd19 arm, which is the same
# tokenizer at the same locus count.
echo "### six aggregators, single-drug, all 19 loci: 6 cells x 11 drugs = 66 jobs"
python scripts/sbatch_all_runs.py $DRY_FLAG \
    --experiments "${CELLS[@]}" --drugs all \
    "${LOCI[@]}" "${TRAIN[@]}" "${KEEP[@]}" \
    --run-prefix "${RUN}/" \
    --mem 128G --cpus 4 --gpus 1 --time 36:00:00 --constraint vram23

# --- the sparse baseline at the same 19 loci --------------------------------
# CPU only, and it is the number every cell above has to beat before any of them
# is interesting. sparse_baseline.py lives with the 20260825 run; it takes the
# same --loci the runners do.
echo "### sparse baseline at 19 loci: 11 CPU jobs"
for DRUG in ISONIAZID RIFAMPICIN ETHAMBUTOL PYRAZINAMIDE STREPTOMYCIN \
            KANAMYCIN AMIKACIN CAPREOMYCIN LEVOFLOXACIN MOXIFLOXACIN ETHIONAMIDE; do
  [ "${DRY:-0}" = "1" ] && { echo "  would submit sparse19 $DRUG"; continue; }
  sbatch --job-name="abr_sparse19_${DRUG}" \
    "results/experiments/${RUN}/sbatch_sparse19.sh" "$DRUG"
done

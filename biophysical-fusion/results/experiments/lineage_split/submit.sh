#!/bin/bash
# lineage_split — the paper's grid retrained on lineage-grouped splits (2026-10-15).
#
#   DRY=1 bash results/experiments/lineage_split/submit.sh          # print, submit nothing
#   ARMS="canary" bash results/experiments/lineage_split/submit.sh  # one job first
#   ARMS="lf cnn lasso" bash results/experiments/lineage_split/submit.sh
#
# Splits: results/experiments/lineage_split/splits/{DRUG}.csv, built by
# scripts/build_lineage_splits.py from data_snapshot/lineage/lineage_calls.csv
# (TB-Profiler barcode). Sublineage groups (deepened to <=10% of a cohort where
# possible); Beijing (lineage2.2.1*) is CV fold 4 on its own; the rest is packed
# into a ~20% test split + CV folds 0-3, balanced on n and R. No group crosses
# any split. Every cell below is otherwise byte-for-byte the paper's protocol:
#   LocusFusion v3  = locusfusion_v3/{sd,sd19}_*_unk__locusfusion_v2
#   CNNs            = full_run_v2 (per-drug loci) / alllocus_run_v2 (19 loci)
#   LASSO           = lasso_wholeseq_20260901/onehot_{perdrug,all}
#   300 epochs / patience 30 / min 50 / seed 0, same five modality sets.
#
# Code runs from the frozen copy ../lineage_snapshot (BASE_COMMIT +
# working_tree.patch); data from data_snapshot via ABR_*; results land here via
# the snapshot's results/ symlink.
#
# Weights: kept for LocusFusion (Shapley follow-up) and the stacked CNN; NOT for
# cis-fusion / late-fusion (AUCs only; ~45 GB saved). Weights dir model_weights_lineage/.
set -euo pipefail
W=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis
SNAP=$W/lineage_snapshot/biophysical-fusion
DATA=$W/data_snapshot
cd "$SNAP"
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"
ARMS="${ARMS:-canary}"
SPLITS=results/experiments/lineage_split/splits
COMMON=(--project-dir "$SNAP" --split-dir "$SPLITS"
        --weights-dir "$W/model_weights_lineage"
        --env "ABR_GENOTYPE_DIR=$DATA/genomic_data/aligned"
        --env "ABR_PHENOTYPE_CSV=$DATA/phenotype_data/master_resistance_table.csv"
        --env "ABR_GENBANK=$DATA/genbank/genome.gbff")
TRAIN=(--epochs 300 --patience 30 --min-epochs 50)
LOCI=(--loci eis embA embB embC ethA ethR fabG1 gid gyrA gyrB inhA katG pncA
      rpoB rpoC rpsL rrl rrs tlyA)
MODSETS=(dna dna_protein dna_biophysical dna_regulatory all_modalities)
LF=(); for m in "${MODSETS[@]}"; do LF+=("${m}_unk__locusfusion_v2"); done
cells() { local a=$1; local out=(); for m in "${MODSETS[@]}"; do out+=("${m}__${a}"); done; echo "${out[@]}"; }

sd()   { python scripts/sbatch_all_runs.py $DRY_FLAG "${COMMON[@]}" --drugs all \
           "${TRAIN[@]}" --run-prefix "lineage_split/sd_" \
           --mem "$1" --cpus 4 --gpus 1 --time "$2" --constraint vram23 "${@:3}"; }
sd19() { python scripts/sbatch_all_runs.py $DRY_FLAG "${COMMON[@]}" --drugs all \
           "${LOCI[@]}" "${TRAIN[@]}" --run-prefix "lineage_split/sd19_" \
           --mem 128G --cpus 4 --gpus 1 --time 36:00:00 --constraint vram23 "$@"; }

for arm in $ARMS; do
  case "$arm" in
    canary)
      echo "### canary: LocusFusion v3, 19 loci, all modalities, ETHIONAMIDE (1 job)"
      python scripts/sbatch_all_runs.py $DRY_FLAG "${COMMON[@]}" --drugs ETHIONAMIDE \
          "${LOCI[@]}" "${TRAIN[@]}" --save-weights best \
          --experiments all_modalities_unk__locusfusion_v2 \
          --run-prefix "lineage_split/canary_sd19_" \
          --mem 128G --cpus 4 --gpus 1 --time 36:00:00 --constraint vram23 ;;
    lf)
      echo "### LocusFusion v3: 2 locus sets x 5 modsets x 11 drugs = 110 jobs"
      sd 64G 24:00:00 --experiments "${LF[@]}" --save-weights best
      sd19 --experiments "${LF[@]}" --save-weights best ;;
    cnn)
      echo "### stacked CNN (weights kept), cis-fusion + late-fusion (no weights): 330 jobs"
      sd 48G 16:00:00 --experiments $(cells mdcnn) --save-weights best
      sd19 --experiments $(cells mdcnn) --save-weights best
      sd 48G 16:00:00 --experiments $(cells cisfusion) $(cells late_fusion) --save-weights none
      sd19 --experiments $(cells cisfusion) $(cells late_fusion) --save-weights none ;;
    lasso)
      echo "### LASSO one-hot: 2 locus sets x 11 drugs = 22 CPU jobs"
      for loci in perdrug all; do
        for d in AMIKACIN CAPREOMYCIN ETHAMBUTOL ETHIONAMIDE ISONIAZID KANAMYCIN \
                 LEVOFLOXACIN MOXIFLOXACIN PYRAZINAMIDE RIFAMPICIN STREPTOMYCIN; do
          cmd=(sbatch --chdir="$SNAP" --job-name="lasso-lineage-${loci}-${d}"
               "$SNAP/results/experiments/lineage_split/sbatch_lasso_lineage.sh" "$d" "$loci")
          if [ -n "$DRY_FLAG" ]; then echo "${cmd[@]}"; else "${cmd[@]}"; fi
        done
      done ;;
    *) echo "unknown arm '$arm' (canary lf cnn lasso)"; exit 1 ;;
  esac
done

#!/bin/bash
# locus_sets — LocusFusion v3, stacked CNN and LASSO on two larger locus sets,
# lineage-grouped splits, single-drug and multi-drug (2026-10-30).
#
#   DRY=1 bash results/experiments/locus_sets/submit.sh            # print, submit nothing
#   ARMS="canary_who" ... ; ARMS="canary_lab" ...  # sizing jobs first
#   ARMS="sd_lf sd_cnn md lasso" bash results/experiments/locus_sets/submit.sh
#
# Locus sets (datasets.loader.LOCUS_SETS):
#   who_t12 = each drug's WHO 2023 Table-21 tier 1+2 genes (6-14 per drug; 54 in union)
#   lab64   = all 64 loci in saishradha's genomic_data/aligned, given to EVERY drug
# FASTAs: $W/locus_sets/aligned (68 symlinks; insertion-bug-repaired copies, README there).
#
# Protocol = lineage_split's, cell for cell: lineage-grouped splits (Beijing = fold 4),
# 300 epochs / patience 30 / min 50 / seed 0, the same five modality sets, LocusFusion
# v3 = *_unk__locusfusion_v2. Controls for single-drug are lineage_split/{sd,sd19}_*
# and lasso_onehot_{perdrug,all}. Multi-drug had no lineage-split run, so the 19-locus
# multi-drug control (bigtb19) is trained here, on the same repaired FASTAs.
# Regulatory regions are each drug's default (unchanged across locus sets).
#
# Code runs from the frozen copy ../locus_sets_snapshot (make_snapshot.sh); data from
# data_snapshot via ABR_*; results land here via the snapshot's results/ symlink.
set -euo pipefail
W=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis
SNAP=$W/locus_sets_snapshot/biophysical-fusion
DATA=$W/data_snapshot
GENO=$W/locus_sets/aligned
MDSPLIT=results/experiments/locus_sets/splits   # MULTIDRUG.csv: one lineage partition for the joint cohort
cd "$SNAP"
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"
ARMS="${ARMS:-canary_who}"
# Memory/time per arm; set from the canary's MaxRSS (override from the env).
SD_MEM_WHO="${SD_MEM_WHO:-128G}";  SD_TIME_WHO="${SD_TIME_WHO:-36:00:00}"
SD_MEM_LAB="${SD_MEM_LAB:-240G}";  SD_TIME_LAB="${SD_TIME_LAB:-72:00:00}"
MD_MEM="${MD_MEM:-240G}";          MD_TIME="${MD_TIME:-72:00:00}"
COMMON=(--project-dir "$SNAP"
        --split-dir results/experiments/lineage_split/splits
        --weights-dir "$W/model_weights_locus_sets"
        --env "ABR_GENOTYPE_DIR=$DATA/genomic_data/aligned"
        --env "ABR_PHENOTYPE_CSV=$DATA/phenotype_data/master_resistance_table.csv"
        --env "ABR_GENBANK=$DATA/genbank/genome.gbff"
        --genotype-dir "$GENO"
        --epochs 300 --patience 30 --min-epochs 50 --save-weights best
        --cpus 4 --gpus 1 --constraint vram23)
LOCI19=(--loci eis embA embB embC ethA ethR fabG1 gid gyrA gyrB inhA katG pncA
        rpoB rpoC rpsL rrl rrs tlyA)
MODSETS=(dna dna_protein dna_biophysical dna_regulatory all_modalities)
lf()  { local o=(); for m in "${MODSETS[@]}"; do o+=("${m}_unk__locusfusion_v2"); done; echo "${o[@]}"; }
cnn() { local o=(); for m in "${MODSETS[@]}"; do o+=("${m}__mdcnn"); done; echo "${o[@]}"; }
mem_of()  { [ "$1" = lab64 ] && echo "$SD_MEM_LAB"  || echo "$SD_MEM_WHO"; }
time_of() { [ "$1" = lab64 ] && echo "$SD_TIME_LAB" || echo "$SD_TIME_WHO"; }
run() { python scripts/sbatch_all_runs.py $DRY_FLAG "${COMMON[@]}" "$@"; }

for arm in $ARMS; do
  case "$arm" in
    canary_who)
      # who_t12 union (54 loci, ~97 kb) is ~the size of lab64 (64, ~103 kb), so the
      # multi-drug pair sizes both multi-drug arms; INH is the largest who_t12 drug.
      echo "### canary_who: who_t12 all_modalities, multi-drug + ISONIAZID single-drug, LF + stacked CNN (4 jobs)"
      run --multidrug --split-dir "$MDSPLIT" --locus-set who_t12 --experiments all_modalities_unk__locusfusion_v2 all_modalities__mdcnn \
          --run-prefix "locus_sets/canary_who_t12_" --mem 240G --time 72:00:00
      run --locus-set who_t12 --drugs ISONIAZID --experiments all_modalities_unk__locusfusion_v2 all_modalities__mdcnn \
          --run-prefix "locus_sets/canary_who_t12_sd_" --mem 128G --time 36:00:00 ;;
    canary_lab)
      echo "### canary_lab: lab64 all_modalities, RIFAMPICIN single-drug, LF + stacked CNN (2 jobs)"
      run --locus-set lab64 --drugs RIFAMPICIN --experiments all_modalities_unk__locusfusion_v2 all_modalities__mdcnn \
          --run-prefix "locus_sets/canary_lab64_sd_" --mem 240G --time 72:00:00 ;;
    sd_lf|sd_cnn)
      exps=$([ "$arm" = sd_lf ] && lf || cnn)
      for set in who_t12 lab64; do
        echo "### $arm single-drug $set: 5 modsets x 11 drugs = 55 jobs"
        run --locus-set "$set" --drugs all --experiments $exps \
            --run-prefix "locus_sets/${set}_sd_" --mem "$(mem_of $set)" --time "$(time_of $set)"
      done ;;
    md)
      echo "### multi-drug: {bigtb19, who_t12, lab64} x 5 modsets x {LF, stacked CNN} = 30 jobs"
      run --multidrug --split-dir "$MDSPLIT" "${LOCI19[@]}" --experiments $(lf) $(cnn) \
          --run-prefix "locus_sets/bigtb19_" --mem 128G --time 36:00:00
      for set in who_t12 lab64; do
        run --multidrug --split-dir "$MDSPLIT" --locus-set "$set" --experiments $(lf) $(cnn) \
            --run-prefix "locus_sets/${set}_" --mem "$MD_MEM" --time "$MD_TIME"
      done ;;
    lasso)
      echo "### LASSO one-hot: 2 locus sets x 11 drugs = 22 CPU jobs"
      for set in who_t12 lab64; do
        for d in AMIKACIN CAPREOMYCIN ETHAMBUTOL ETHIONAMIDE ISONIAZID KANAMYCIN \
                 LEVOFLOXACIN MOXIFLOXACIN PYRAZINAMIDE RIFAMPICIN STREPTOMYCIN; do
          cmd=(sbatch --chdir="$SNAP" --job-name="lasso-locusset-${set}-${d}"
               "$SNAP/results/experiments/locus_sets/sbatch_lasso_locus_sets.sh" "$d" "$set")
          if [ -n "$DRY_FLAG" ]; then echo "${cmd[@]}"; else "${cmd[@]}"; fi
        done
      done ;;
    *) echo "unknown arm '$arm' (canary_who canary_lab sd_lf sd_cnn md lasso)"; exit 1 ;;
  esac
done

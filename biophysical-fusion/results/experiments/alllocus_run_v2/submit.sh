#!/bin/bash
# alllocus_run_v2 — the single-drug grid on the joint 19-locus input, retrained
# with the full_run_v2 method and WITH SAVED WEIGHTS.
#
# Run from the project root:  bash results/experiments/alllocus_run_v2/submit.sh
# Dry run:                    DRY=1 bash .../submit.sh
#
# Why: alllocus_run (2026-08-06) answered the question but at full_run's settings
# (150 epochs / patience 15 / no warmup), checkpointed nothing, and lost 10 of
# 220 jobs to resource limits. Its results folder was then lost entirely and had
# to be rebuilt from SLURM logs. This redoes it at full_run_v2's settings so the
# comparison is against the CURRENT baseline rather than the superseded one.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # project root
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"

# --- the input under test: MD-CNN's drug-independent locus rule --------------
# Every curated locus FASTA, the same 19 datasets.loci_on_disk() gives the joint
# runs, passed explicitly so the run is pinned rather than depending on what is
# on disk later. Byte-identical to the alllocus_run list.
LOCI=(--loci eis embA embB embC ethA ethR fabG1 gid gyrA gyrB inhA katG pncA
      rpoB rpoC rpsL rrl rrs tlyA)

# --- the full_run_v2 method, unchanged --------------------------------------
# Identical to results/experiments/full_run_v2/submit.sh so that run is the
# matched control: the ONLY difference between a cell here and a cell there is
# the locus set above.
TRAIN=(--epochs 300 --patience 30 --min-epochs 50)
KEEP=(--save-weights best)

# --- the full_run_v2 GRID, named explicitly ---------------------------------
# NOT --experiments all. Since 2026-08-13 MODALITY_SETS also holds the SHAP
# leave-one-out arms (no_dna / no_regulatory / regulatory_only), so `all` is now
# 32 cells / 352 jobs. full_run_v2's grid is these 5 modality sets x 4
# architectures = 20 cells x 11 drugs = 220 jobs.
CELLS=()
for m in dna dna_protein dna_biophysical dna_regulatory all_modalities; do
    for a in late_fusion mdcnn setfusion cisfusion; do CELLS+=("${m}__${a}"); done
done

# --- resources: raised from full_run_v2, because 19 loci is a bigger input ---
# full_run_v2's single-drug 48G / 16:00:00 would reproduce the exact failures
# alllocus_run hit. Measured from that run's recovered provenance.csv:
#   --mem 128G   every surviving job peaked at 59-62 GiB against a 64G request,
#                and the 8 that died were host OUT_OF_MEMORY -- the four
#                >=12.9k-isolate drugs (PZA/EMB/INH/RIF) in the two widest
#                late_fusion cells. 388 GPU nodes have >=128G, so this costs
#                nothing in scheduling.
#   -t 36:00:00  worst job projects to ~11.4 h from the recovered best_epoch
#                distribution (x1.32 epochs vs 150/pat15), but the two setfusion
#                ISONIAZID jobs TIMED OUT at 12 h and never recorded a duration,
#                so their true 300-epoch cost is unbounded by measurement.
#   vram23       one job (KANAMYCIN all_modalities__late_fusion) died on a CUDA
#                OOM on a 10.9 GiB card. PYTORCH_CUDA_ALLOC_CONF=expandable_
#                segments:True now guards that too, but it landed in d52700a --
#                AFTER alllocus_run ran, so it is untested on this input. 146
#                nodes satisfy the constraint.
#
# Expect ~440-460 GPU-hours (348 measured at 150 epochs x 1.32 more epochs) and
# ~40 GB of weights: a 19-locus checkpoint is 139-228 MB against full_run_v2's
# 7-69 MB. The volume had 1.1 TB free when this was written.
echo "### single-drug on all 19 loci: 20 cells x 11 drugs = 220 jobs"
python scripts/sbatch_all_runs.py $DRY_FLAG \
    --experiments "${CELLS[@]}" --drugs all \
    "${LOCI[@]}" "${TRAIN[@]}" "${KEEP[@]}" \
    --run-prefix "alllocus_run_v2/" \
    --mem 128G --cpus 4 --gpus 1 --time 36:00:00 --constraint vram23

echo
echo "220 jobs submitted. Monitor: squeue -u \$USER"
echo "Results: results/experiments/alllocus_run_v2/"
echo "Weights: /project/pi_mfiterau_umass_edu/abr_model_weights/alllocus_run_v2/"
echo
echo "No joint jobs here on purpose: joint runs ALREADY use all 19 loci, so"
echo "full_run_v2's multidrug_* cells are the joint arm of this comparison."

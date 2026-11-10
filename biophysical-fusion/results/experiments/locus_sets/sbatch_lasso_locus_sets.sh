#!/bin/bash
# One LASSO (one-hot) cell on the lineage-grouped split for the extra locus
# universes. Usage, from the snapshot root:
#   sbatch --chdir=<snapshot>/biophysical-fusion --job-name=lasso-locusset-${LOCUS_SET}-${DRUG} \
#       results/experiments/locus_sets/sbatch_lasso_locus_sets.sh DRUG LOCUS_SET
#     LOCUS_SET = who_t12 | lab64
# Mirrors lineage_split/sbatch_lasso_lineage.sh; only the locus universe and the
# genotype dir ($W/locus_sets/aligned, the repaired 68-FASTA folder) differ.
#
# Resources (2026-10-30). X is a float32 CSR; peak RSS of the 19-locus lineage
# jobs was linear in nnz at ~37 B/nnz (INH 281M nnz -> 11.8 GB, RIF 285M -> 11.0,
# EMB 239M -> 10.3, PZA 195M -> 8.8, STR 92M -> 4.8; perdrug cells ~2 GB).
# nnz/isolate ~= number of variable alignment columns: 16.6k over the 19 loci,
# ~46k over lab64 (43.2k measured on 55/64 loci + ~3k for the rest), so the
# worst lab64 cell (INH/RIF, ~17.5k isolates) is ~800M nnz -> ~30-35 GB peak.
# 64G leaves ~1.8x headroom; who_t12 cells are well below. Old jobs were
# single-threaded (TotalCPU == Elapsed; liblinear), 19-locus worst 29 min, so
# lab64 should take ~1-2 h; 12 h is generous.
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -p cpu
#SBATCH -t 12:00:00
#SBATCH -o slurm_logs/slurm-lasso-locus-sets-%x-%j.out
set -euo pipefail
DRUG="$1"; LOCI="$2"
case "$LOCI" in who_t12|lab64) ;; *) echo "LOCUS_SET must be who_t12|lab64, got '$LOCI'"; exit 2 ;; esac
RUN=results/experiments/locus_sets
echo "lasso locus-sets (lineage split): $DRUG onehot loci=$LOCI (host $(hostname), $(date))"
source ~/.bashrc
conda activate abr_env
module list 2>&1 || true
which python && python --version
W=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis
DATA=$W/data_snapshot
export ABR_GENOTYPE_DIR=$W/locus_sets/aligned
export ABR_PHENOTYPE_CSV=$DATA/phenotype_data/master_resistance_table.csv
export ABR_GENBANK=$DATA/genbank/genome.gbff
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
python -u "${RUN}/lasso_locus_sets.py" \
    --drugs "$DRUG" --encoding onehot --locus-set "$LOCI" \
    --genotype-dir "$ABR_GENOTYPE_DIR" --phenotype-csv "$ABR_PHENOTYPE_CSV" \
    --split-dir results/experiments/lineage_split/splits \
    --out "${RUN}/lasso_onehot_${LOCI}"
echo "Done: $DRUG onehot $LOCI at $(date)"

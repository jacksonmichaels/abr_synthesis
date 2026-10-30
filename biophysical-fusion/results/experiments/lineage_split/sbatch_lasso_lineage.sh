#!/bin/bash
# One LASSO (one-hot) cell on the lineage-grouped split. Usage, from the snapshot root:
#   sbatch --chdir=<lineage_snapshot>/biophysical-fusion sbatch_lasso_lineage.sh DRUG LOCUS_SET
#     LOCUS_SET = perdrug | all
# Resources as lasso_wholeseq_20260901/sbatch_lasso.sh (worst cell INH/onehot/all
# 11.3 GB peak, minutes of compute).
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -p cpu
#SBATCH -t 12:00:00
#SBATCH -o slurm_logs/slurm-lasso-lineage-%x-%j.out
set -euo pipefail
DRUG="$1"; LOCI="$2"
RUN=results/experiments/lineage_split
echo "lasso lineage-split: $DRUG onehot loci=$LOCI (host $(hostname), $(date))"
source ~/.bashrc
conda activate abr_env
which python && python --version
DATA=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot
export ABR_GENOTYPE_DIR=$DATA/genomic_data/aligned
export ABR_PHENOTYPE_CSV=$DATA/phenotype_data/master_resistance_table.csv
export ABR_GENBANK=$DATA/genbank/genome.gbff
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
python -u "${RUN}/lasso_lineage.py" \
    --drugs "$DRUG" --encoding onehot --locus-set "$LOCI" \
    --genotype-dir "$ABR_GENOTYPE_DIR" --phenotype-csv "$ABR_PHENOTYPE_CSV" \
    --split-dir "${RUN}/splits" \
    --out "${RUN}/lasso_onehot_${LOCI}"
echo "Done: $DRUG onehot $LOCI at $(date)"

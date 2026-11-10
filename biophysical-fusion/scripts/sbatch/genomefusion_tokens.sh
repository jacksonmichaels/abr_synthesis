#!/bin/bash
# Build the GenomeFusion token store: one array task per shard, then merge.
#
#   cd biophysical-fusion
#   sbatch --array=0-15%16 scripts/sbatch/genomefusion_tokens.sh
#   sbatch --dependency=afterok:<jobid> scripts/sbatch/genomefusion_tokens.sh merge
#
# Reads the BIG-TB VCFs straight from the read-only pi_annagreen tree (copying
# 35,884 files off that spinning disk costs far more than one read pass) and
# writes only to $OUT.
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH --job-name=gf_tokens
#SBATCH --output=results/experiments/genomefusion/logs/%x-%A_%a.out
#SBATCH --error=results/experiments/genomefusion/logs/%x-%A_%a.err
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
set -euo pipefail

echo "host $(hostname)  start $(date)"
module list 2>&1 || true
source ~/.bashrc
conda activate abr_env
which python && python --version

WORKSPACE=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis
OUT="$WORKSPACE/genomefusion_tokens"
N_SHARDS=16

export ABR_GENOTYPE_DIR="$WORKSPACE/data_snapshot/genomic_data/aligned"
export ABR_PHENOTYPE_CSV="$WORKSPACE/data_snapshot/phenotype_data/master_resistance_table.csv"
export ABR_GENBANK="$WORKSPACE/data_snapshot/genbank/genome.gbff"
export ABR_WORKSPACE="$WORKSPACE"
export ABR_GF_REFS="$WORKSPACE/genomefusion_refs"
export ABR_GF_CACHE="$WORKSPACE/genomefusion_cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "$WORKSPACE/biophysical-fusion"

# the /project mount is flaky and a failed read must not look like success
ls /project/pi_annagreen_umass_edu >/dev/null

if [ "${1:-}" = "merge" ]; then
    python -u -m genomefusion.build_tokens --merge --n-shards "$N_SHARDS" --out "$OUT"
else
    python -u -m genomefusion.build_tokens \
        --shard "${SLURM_ARRAY_TASK_ID}" --n-shards "$N_SHARDS" --out "$OUT"
fi
echo "done $(date)"

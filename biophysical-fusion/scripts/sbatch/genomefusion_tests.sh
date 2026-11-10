#!/bin/bash
# GenomeFusion tests. The tokenizer ones read real VCFs off /project, which is
# too slow and too heavy for a login node.
#
#   cd biophysical-fusion && sbatch scripts/sbatch/genomefusion_tests.sh
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH --job-name=gf_tests
#SBATCH --output=results/experiments/genomefusion/logs/%x-%j.out
#SBATCH --error=results/experiments/genomefusion/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
set -euo pipefail

echo "host $(hostname)  start $(date)"
source ~/.bashrc
conda activate abr_env
which python && python --version

WORKSPACE=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis
export ABR_GENOTYPE_DIR="$WORKSPACE/data_snapshot/genomic_data/aligned"
export ABR_PHENOTYPE_CSV="$WORKSPACE/data_snapshot/phenotype_data/master_resistance_table.csv"
export ABR_GENBANK="$WORKSPACE/data_snapshot/genbank/genome.gbff"
export ABR_WORKSPACE="$WORKSPACE"
export ABR_GF_REFS="$WORKSPACE/genomefusion_refs"
export ABR_GF_CACHE="$WORKSPACE/genomefusion_cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "$WORKSPACE/biophysical-fusion"
ls /project/pi_annagreen_umass_edu >/dev/null

python -u -m genomefusion.tests.test_model
python -u -m genomefusion.tests.test_tokenizer
echo "done $(date)"

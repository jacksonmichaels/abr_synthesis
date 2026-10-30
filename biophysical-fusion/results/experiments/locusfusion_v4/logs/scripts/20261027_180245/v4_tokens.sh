#!/bin/bash
#SBATCH --job-name=v4_tokens
#SBATCH --output=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/locusfusion_v4/logs/%x-%j.out
#SBATCH --error=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/locusfusion_v4/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
set -euo pipefail
echo "host $(hostname)  start $(date)"
module list 2>&1 || true
source ~/.bashrc
conda activate abr_env
which python && python --version
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export ABR_GENOTYPE_DIR=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/genomic_data/aligned
export ABR_PHENOTYPE_CSV=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/phenotype_data/master_resistance_table.csv
export ABR_GENBANK=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/genbank/genome.gbff
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/v4_snapshot/20261027_180245/biophysical-fusion
python -u - /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/token_cache_v4 <<'EOF'
from bigtb_ref import REAL_GENOTYPE_DIR as G, REAL_PHENOTYPE_CSV as P, REAL_REGULATORY_DIR as R
from locusfusion_v4.tokenizer import ALL_DRUGS, load_or_build
import sys
cache = sys.argv[1]
for drop in (False, True):
    for drugs in [[d] for d in ALL_DRUGS] + [ALL_DRUGS]:
        load_or_build(cache, drugs, G, P, R, drop_synonymous=drop)
EOF
echo "done $(date)"

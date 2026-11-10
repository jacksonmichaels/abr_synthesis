#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH --job-name=gf_forward_trace
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/%x-%j.out
#SBATCH --error=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/%x-%j.err

# One real forward pass of the GenomeFusion whole-genome ISONIAZID model (inference only,
# no GPU). Defaults read the corrected checkpoint; arguments pass through to the script.
#   sbatch --dependency=afterok:<size6776 ISONIAZID job> scripts/sbatch/genomefusion_forward_trace.sh
set -euo pipefail
echo "host $(hostname)  start $(date)"
module list 2>&1 || true
source ~/.bashrc
conda activate abr_env
which python && python --version
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
W=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis
export ABR_GENOTYPE_DIR="$W/data_snapshot/genomic_data/aligned"
export ABR_PHENOTYPE_CSV="$W/data_snapshot/phenotype_data/master_resistance_table.csv"
export ABR_GENBANK="$W/data_snapshot/genbank/genome.gbff"
export ABR_WORKSPACE="$W"
export ABR_GF_REFS="$W/genomefusion_refs"
export ABR_GF_CACHE="$W/genomefusion_cache"
cd "$W/biophysical-fusion"
python -u scripts/genomefusion_forward_trace.py "$@"
# the real run (no arguments) also refreshes the figure and the notebook
if [ "$#" -eq 0 ]; then
    python scripts/genomefusion_method_figures.py
    python scripts/build_genomefusion_architectures.py
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 \
        notebooks/genomefusion/genomefusion_architectures.ipynb
fi
echo "done $(date)"

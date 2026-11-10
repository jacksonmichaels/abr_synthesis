#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH --job-name=abr_shapley_lineage_all
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram23
#SBATCH -p gpu
#SBATCH -t 02:00:00
#SBATCH --array=0-10%6
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/shapley_alltest_lineage-%A_%a.out

# Exact modality / called-uncalled Shapley + locus Shapley (32 antithetic
# permutations, exact when <= 9 loci differ) on EVERY lineage-split test isolate,
# LocusFusion v3 sd19 checkpoints retrained on lineage-grouped splits
# (model_weights_lineage/lineage_split/, --split lineage). One array task per
# drug, all five cells off one shared load. Same settings as the random-split
# all-test run (65716237: shapley_locusfusion_v3.sh --n-explain 100000 --n-perm 32
# --out results/analysis/shap_locusfusion_v3_alltest/shapley). Writes to
# results/analysis/shap_locusfusion_lineage_alltest/shapley/ only.
#
#   sbatch --array=6 -t 01:00:00 scripts/sbatch/shapley_alltest_lineage.sh \
#       --cells sd19_all_modalities_unk                     # smoke: LEVOFLOXACIN
#   sbatch scripts/sbatch/shapley_alltest_lineage.sh --skip-existing   # full grid
#
# Resources: the random-split all-test tasks peaked at 43-57 GB RSS (the shared
# genotype load) and 14-37 min; lineage test sets are 13% larger in total
# (19,756 vs 17,433; RIF 4,105 vs 3,517), so 80G / 2 h keeps the same margin.
set -euo pipefail
echo "Job ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} on $(hostname) at $(date)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate abr_env
module list 2>&1 || true
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
which python && python --version

DATA=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot
export ABR_GENOTYPE_DIR=$DATA/genomic_data/aligned
export ABR_PHENOTYPE_CSV=$DATA/phenotype_data/master_resistance_table.csv
export ABR_GENBANK=$DATA/genbank/genome.gbff
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DRUGS=(AMIKACIN CAPREOMYCIN ETHAMBUTOL ETHIONAMIDE ISONIAZID KANAMYCIN
       LEVOFLOXACIN MOXIFLOXACIN PYRAZINAMIDE RIFAMPICIN STREPTOMYCIN)
DRUG="${DRUGS[${SLURM_ARRAY_TASK_ID:-6}]}"
echo "drug: $DRUG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/shapley_locusfusion_v3.py --split lineage \
    --run lineage_split --arch locusfusion_v2 \
    --cells sd19_all_modalities_unk sd19_dna_regulatory_unk sd19_dna_unk \
            sd19_dna_protein_unk sd19_dna_biophysical_unk \
    --drugs "$DRUG" --n-explain 100000 --n-perm 32 \
    --out results/analysis/shap_locusfusion_lineage_alltest/shapley \
    --eg-out results/analysis/shap_locusfusion_lineage_alltest \
    --device cuda "$@"

echo "Done: Shapley all-test lineage $DRUG at $(date)"

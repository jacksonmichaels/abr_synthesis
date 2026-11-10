#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH --job-name=abr_predict_lf_v3
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -p cpu
#SBATCH -t 08:00:00
#SBATCH --array=0-11%6
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/predict_lf_v3-%A_%a.out

# Raw per-isolate outputs of every LocusFusion v3 checkpoint (inference only,
# CPU). Tasks 0-10: one drug each, all 23 single-drug cells (sd, sd19,
# ablation_sd19, s1000-4000, cv101-104), frozen-snapshot code; at most four data
# loads per drug (~14 GB peak each for 19 loci, one at a time). Task 11: the
# five md_multidrug cells with repo code (the multi-drug loader needs more
# memory, so submit it on its own):
#
#   sbatch --array=0-10%6 scripts/sbatch/predict_locusfusion_v3.sh
#   sbatch --array=11 --mem=160G scripts/sbatch/predict_locusfusion_v3.sh
#
# --skip-existing makes a resubmission resume. Writes
# results/analysis/predictions_locusfusion_v3/{cell}/{DRUG}.csv.gz (+ .meta.json).
set -euo pipefail
echo "Job ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} on $(hostname) at $(date)"
source ~/.bashrc
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

DRUGS=(AMIKACIN CAPREOMYCIN ETHAMBUTOL ETHIONAMIDE ISONIAZID KANAMYCIN
       LEVOFLOXACIN MOXIFLOXACIN PYRAZINAMIDE RIFAMPICIN STREPTOMYCIN)
T=${SLURM_ARRAY_TASK_ID:-0}
if [ "$T" -lt "${#DRUGS[@]}" ]; then
    python -u scripts/predict_locusfusion_v3.py --code snapshot --drugs "${DRUGS[$T]}" --skip-existing
else
    python -u scripts/predict_locusfusion_v3.py --code repo --multidrug --skip-existing
fi
echo "Done at $(date)"

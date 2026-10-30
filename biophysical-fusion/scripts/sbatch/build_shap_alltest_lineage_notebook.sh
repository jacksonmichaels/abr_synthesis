#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -p cpu
#SBATCH --qos=short
#SBATCH -t 00:30:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/build_shap_alltest_lineage_notebook-%j.out
#SBATCH --job-name=abr_shap_lineage_nb
# Aggregate the lineage all-test SHAP / Shapley per-cell files, refresh the WHO
# side-by-side (lineage-trained 19-locus LASSO), then rebuild and execute
# notebooks/paper/shap_shapley_lineage_alltest.ipynb (figures ->
# results/figures/shap_lineage_alltest/). Safe to re-run while arrays land.
set -euo pipefail
hostname; date
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate abr_env
module list 2>&1 || true
which python && python --version
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
A=results/analysis/shap_locusfusion_lineage_alltest
python scripts/shapley_locusfusion_v3.py --split lineage --aggregate --out $A/shapley
python scripts/shap_locusfusion_v3.py --split lineage --aggregate --run lineage_split --out $A || \
    echo "EG aggregate skipped (no EG results yet?)"
python scripts/shap_v3_who_compare.py --shares $A/locus_shares.csv --out $A/who \
    --lasso-onehot19 results/experiments/lineage_split/lasso_onehot_all \
    --lasso-vcf19 '' --lasso-genome ''
python scripts/build_shap_alltest_notebook.py --split lineage
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 \
    notebooks/paper/shap_shapley_lineage_alltest.ipynb
echo "Done: shap_alltest lineage notebook at $(date)"

#!/bin/bash
# Appendix Shapley / SHAP figures on the LINEAGE-split LocusFusion v3 checkpoints.
#
#   DRY=1 bash results/analysis/shap_locusfusion_lineage_alltest/submit.sh               # print only
#   ARMS="smoke"   bash results/analysis/shap_locusfusion_lineage_alltest/submit.sh      # 2 GPU jobs, LEVOFLOXACIN
#   ARMS="shapley eg" bash results/analysis/shap_locusfusion_lineage_alltest/submit.sh   # 11 + 55 GPU tasks
#   ARMS="figures" DEP=<jobid>[:<jobid>] bash .../submit.sh                              # 1 CPU job after them
#
# Inputs: model_weights_lineage/lineage_split/sd19_*_unk__locusfusion_v2 (best CV
# fold only), results/experiments/lineage_split/ (run JSONs, splits/,
# lasso_onehot_all). Outputs: this folder and results/figures/shap_lineage_alltest/;
# the random-split shap_locusfusion_v3_alltest/ and shap_v3_alltest/ are untouched.
# Failed cells still exit 0 per task: grep the logs for FAILED and look for
# shapley/per_cell/failures_*.json and lineage_split/locusfusion_v2/shap_failures_*.json.
set -euo pipefail
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
ARMS="${ARMS:-smoke}"
run() { if [ "${DRY:-0}" = "1" ]; then echo "+ $*"; else "$@"; fi; }
for arm in $ARMS; do
  case "$arm" in
    smoke)
      echo "### smoke: Shapley LEVOFLOXACIN x all_modalities + SHAP LEVOFLOXACIN x all_modalities"
      run sbatch --array=6 -t 01:00:00 scripts/sbatch/shapley_alltest_lineage.sh --cells sd19_all_modalities_unk
      run sbatch --array=30 -t 02:00:00 scripts/sbatch/shap_alltest_lineage.sh ;;
    shapley)
      echo "### Shapley: 11 drugs x 5 cells, 11 GPU tasks (max 6 at once)"
      run sbatch scripts/sbatch/shapley_alltest_lineage.sh --skip-existing ;;
    eg)
      echo "### SHAP expected gradients: 55 GPU tasks (max 8 at once)"
      run sbatch scripts/sbatch/shap_alltest_lineage.sh --skip-existing ;;
    figures)
      echo "### aggregate + notebook + figures (CPU)"
      if [ -n "${DEP:-}" ]; then
        run sbatch --dependency=afterany:"$DEP" scripts/sbatch/build_shap_alltest_lineage_notebook.sh
      else
        run sbatch scripts/sbatch/build_shap_alltest_lineage_notebook.sh
      fi ;;
    *) echo "unknown arm '$arm' (smoke shapley eg figures)"; exit 1 ;;
  esac
done

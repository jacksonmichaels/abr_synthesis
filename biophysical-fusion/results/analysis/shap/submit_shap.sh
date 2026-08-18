#!/bin/bash
# SHAP attribution sweep — all 11 drugs, DNA-only vs all-modalities SD-CNN.
#
# Run from the project root:  bash results/analysis/shap/submit_shap.sh
# Dry run:                    DRY=1 bash results/analysis/shap/submit_shap.sh
#
# One job per drug (both cells inside), because the all-modalities rebuild is the
# memory peak: load_dataset materialises every block for every isolate before the
# test split is taken (~190 KB/isolate for INH = 3.4 GB), and two drugs in one
# process would double that for no wall-clock gain.
#
# Reads the full_run_v2 checkpoints, writes to results/analysis/shap/. Nothing is
# retrained — each job rebuilds a saved model, re-scores it on its own stored
# held-out split (and REFUSES to write if the AUC does not reproduce), then runs
# expected-gradient attribution.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # project root

RUN=full_run_v2
ARCH=mdcnn
OUT=results/analysis/shap
LOGS=slurm_logs
mkdir -p "$LOGS" "$OUT"

# n_explain 200 / nsamples 128: the ISONIAZID pilot used nsamples=64, which is
# cheap but not converged — the sweep doubles it since these are the numbers that
# go in the cross-drug table. shap-batch 25 keeps peak memory at 25x128 copies of
# each block.
ARGS=(--run "$RUN" --arch "$ARCH" --cells dna all_modalities
      --n-background 100 --n-explain 200 --nsamples 128 --shap-batch 25
      --seed 0 --device cuda --out "$OUT")

DRUGS=(ISONIAZID RIFAMPICIN ETHAMBUTOL PYRAZINAMIDE STREPTOMYCIN KANAMYCIN
       AMIKACIN CAPREOMYCIN LEVOFLOXACIN MOXIFLOXACIN ETHIONAMIDE)

for DRUG in "${DRUGS[@]}"; do
  SCRIPT=$(cat <<EOF
#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 04:00:00
#SBATCH -o $LOGS/slurm-shap_${ARCH}_${DRUG}-%j.out
#SBATCH --job-name=shap_${DRUG}

echo "SHAP: $DRUG ($ARCH) on \$(hostname)"
source ~/.bashrc
conda activate abr_env
cd $(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/shap_attribution.py --drugs $DRUG ${ARGS[@]}

echo "Done: $DRUG"
EOF
)
  if [ "${DRY:-0}" = "1" ]; then
    echo "=== would submit: $DRUG"
    echo "$SCRIPT" | tail -3
  else
    echo "$SCRIPT" | sbatch
  fi
done

echo
echo "11 jobs. Monitor: squeue -u \$USER -n \$(echo shap_ISONIAZID)"
echo "Results: $OUT/$RUN/$ARCH/{DRUG}/  + blocks_all.csv / columns_all.csv"

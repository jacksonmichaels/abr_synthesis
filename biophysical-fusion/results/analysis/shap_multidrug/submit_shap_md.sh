#!/bin/bash
# Joint (multi-drug) SHAP — does MD-CNN read the right locus for the right drug?
#
# Run from the project root:  bash results/analysis/shap_multidrug/submit_shap_md.sh
# Dry run:                    DRY=1 bash results/analysis/shap_multidrug/submit_shap_md.sh
#
# One job per cell. MEMORY, not GPU, is the binding constraint: the joint cohort
# is the UNION of all 19 loci across all 17,941 isolates — ~575 KB/isolate once
# protein/biophysical/regulatory are on, so ~10 GB — and load_multidrug_dataset
# materialises the whole thing before the test split is taken. This is what
# OOM-killed the first interactive attempt. 180G is sized for that; the DNA-only
# cell needs far less but is cheap to over-request.
set -euo pipefail
cd "$(dirname "$0")/../../.."

for CELL in dna all_modalities; do
  SCRIPT=$(cat <<SBATCH
#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=180G
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 08:00:00
#SBATCH -o slurm_logs/slurm-shapmd_${CELL}-%j.out
#SBATCH --job-name=shapmd_${CELL}

source ~/.bashrc
conda activate abr_env
cd $(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/shap_multidrug.py --cells ${CELL} \\
    --run full_run_v2 --arch mdcnn \\
    --n-background 50 --n-explain 150 --nsamples 64 --shap-batch 5 \\
    --seed 0 --device cuda --out results/analysis/shap_multidrug
SBATCH
)
  if [ "${DRY:-0}" = "1" ]; then
    echo "=== would submit: $CELL"; echo "$SCRIPT" | tail -5
  else
    echo "$SCRIPT" | sbatch
  fi
done

echo "Results: results/analysis/shap_multidrug/full_run_v2/mdcnn/"

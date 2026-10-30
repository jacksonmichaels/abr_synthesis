#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -p cpu-preempt
#SBATCH -t 02:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/build_structure_notebooks-%j.out
#SBATCH --job-name=abr_structure_notebooks

# Rebuild and execute the two notebooks that show the `structure` features:
#   notebooks/structure_validation.ipynb   (AlphaFold coordinates + §7 encodings)
#   notebooks/tokenization_v2.ipynb        (v1 vs v2 tokenizer + §6 structure)
# CPU-only; each loads real single-drug data (~12 GB peak), one at a time.
# cpu-preempt: cheap to lose and resubmit.
set -euo pipefail
echo "Job: structure notebooks  (host $(hostname))"
source ~/.bashrc
conda activate abr_env
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
for nb in structure_validation tokenization_v2; do
    python scripts/build_${nb}.py
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3600 notebooks/${nb}.ipynb
    echo "executed notebooks/${nb}.ipynb"
done
echo "Done: structure notebooks"

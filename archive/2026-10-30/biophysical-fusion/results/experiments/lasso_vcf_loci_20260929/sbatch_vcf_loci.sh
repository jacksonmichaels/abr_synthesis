#!/bin/bash
# Both arms for one drug: whole-genome VCF variants restricted to (a) all 19
# aligned loci, (b) the drug's own loci. Usage: sbatch sbatch_vcf_loci.sh DRUG
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -p cpu
#SBATCH -t 04:00:00
#SBATCH -o slurm_logs/slurm-vcfloci-%x-%j.out
set -euo pipefail
DRUG="$1"
HERE=results/experiments/lasso_vcf_loci_20260929
source ~/.bashrc
conda activate abr_env
echo "vcf-restricted lasso: $DRUG (host $(hostname))"
for R in all19 perdrug; do
  python -u results/experiments/lasso_wholegenome_20260903/lasso_wholegenome.py \
      --drugs "$DRUG" --restrict-loci "$R" --procs 2 --out "${HERE}/${R}"
done
echo "Done: $DRUG"

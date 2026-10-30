#!/bin/bash
# Pooled features for the whole-genome lasso (2026-09-29).
#
# lasso_vcf_loci_20260929 showed the VCF calls and the curated alignments carry
# the same information (macro CV 0.9172 vs 0.9152 at 19 loci, same delta coding),
# so the whole-genome lasso's gap to the 19-locus ONEHOT arm (0.9241) is the
# encoding: L1 on one column per allele cannot pool rare alleles the way a
# penalised wild-type column does. These arms add pooled indicators on top of the
# per-key columns (lasso_wholegenome.py --pool):
#
#   site/        any call covering the base (deletions cover their REF span)
#   gene/        per gene: any HIGH-impact call (lof:), any HIGH|MODERATE (nonsyn:)
#   both/        site + gene
#   all19_both/  both, restricted to the 19 aligned loci
#
# Controls: whole genome, no pooling 0.9174 (lasso_wholegenome_20260903/all_drugs);
# 19 loci, VCF, no pooling 0.9172 (lasso_vcf_loci_20260929/all19);
# 19 loci, alignment onehot 0.9241 (lasso_wholeseq_20260901/onehot_all).
# Same L1 fit, 8-value C grid, split and folds as all of them.
#
# 1 annotation-cache job, then 11 drug jobs held on it (afterok).
#   bash results/experiments/lasso_pooled_20260929/submit.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
HERE=results/experiments/lasso_pooled_20260929
mkdir -p slurm_logs
a=$(sbatch --parsable --chdir="$PWD" --job-name=pooled-annot "$HERE/sbatch_annot.sh" 2>/dev/null)
echo "annotation job: $a"
for d in AMIKACIN CAPREOMYCIN ETHAMBUTOL ETHIONAMIDE ISONIAZID KANAMYCIN \
         LEVOFLOXACIN MOXIFLOXACIN PYRAZINAMIDE RIFAMPICIN STREPTOMYCIN; do
  sbatch --parsable --chdir="$PWD" --dependency="afterok:$a" --job-name="pooled-$d" \
      "$HERE/sbatch_pooled.sh" "$d" 2>/dev/null
done | tr '\n' ' '; echo

#!/bin/bash
# Is the whole-genome lasso's deficit the DATA SOURCE or the ENCODING? (2026-09-29)
#
# Same model, protocol and C grid as lasso_wholegenome_20260903 (pure L1, delta
# coding, snpEff PASS calls), with the columns cut to the H37Rv spans of the
# aligned loci (datasets.cds.genome_span). The matched 19-locus arms read the
# curated FASTA alignments instead:
#
#   all19/    vs lasso_wholeseq_20260901/delta_all      macro CV 0.9152
#   perdrug/  vs lasso_wholeseq_20260901/delta_perdrug  macro CV 0.8613
#
# ~equal  -> the two sources carry the same information; the gap to onehot_all
#            (0.9241) is the encoding (rare variants not pooled per site)
# lower   -> the VCF calls lose something the alignments keep (PASS filter,
#            no-coverage == reference, 20-nt indel truncation)
#
# 11 CPU jobs, one per drug, both arms each.
#   bash results/experiments/lasso_vcf_loci_20260929/submit.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
HERE=results/experiments/lasso_vcf_loci_20260929
mkdir -p slurm_logs
for d in AMIKACIN CAPREOMYCIN ETHAMBUTOL ETHIONAMIDE ISONIAZID KANAMYCIN \
         LEVOFLOXACIN MOXIFLOXACIN PYRAZINAMIDE RIFAMPICIN STREPTOMYCIN; do
  sbatch --parsable --chdir="$PWD" --job-name="vcfloci-$d" "$HERE/sbatch_vcf_loci.sh" "$d"
done 2>/dev/null | tr '\n' ' '; echo

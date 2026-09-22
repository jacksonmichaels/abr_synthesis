#!/bin/bash
# locusfusion_v2 — the same two-stage transformer on the one-token-per-changed-base
# tokenizer (models/locusfusion_v2.py, --arch locusfusion_v2).
#
# Run from the project root:  bash results/experiments/locusfusion_v2/submit.sh
# Dry run:                    DRY=1 bash results/experiments/locusfusion_v2/submit.sh
# One arm only:               ARMS="sd" bash ...        (any of: sd sd19 md)
#
# WHAT CHANGED against locusfusion, and nothing else about the protocol:
#   * DNA is the only source of tokens; protein and biophysical ride on the DNA
#     token at the codon it falls in (column -> codon counted in reference bases).
#   * 16 tokens per LOCUS on one 5'->3' axis, promoter placed by its H37Rv genome
#     position, and a base the gene record and its promoter window share is one
#     token. rRNA genes carry no codons.
#   * protein/biophysical read on the H37Rv codon grid (--protein-frame reference,
#     implied by the arch): residue k is reference codon k in every isolate, so an
#     indel no longer slides residues off their codon, and STOP is a residue.
#   * 38 feature channels, not 42; the per-locus summary token is [NULL].
#
# MATCHED CONTROLS, all already run at exactly this protocol:
#   sd   results/experiments/locusfusion_fixed/sd_*__locusfusion     macro CV 0.9036
#        results/experiments/full_run_v2/*__mdcnn                    macro CV 0.8995
#   sd19 results/experiments/locusfusion_fixed/sd19_*__locusfusion
#        results/experiments/alllocus_run_v2/*__mdcnn                macro CV 0.9246
#   md   results/experiments/newmodels_full/md_*__locusfusion
# Same 300 epochs / patience 30 / --min-epochs 50 / --save-weights best / seed 0,
# same five modality sets, same loci. locusfusion_fixed's tokenizer lives on the
# branch worktree-lf-token-offset-fix, not in this checkout; its numbers are the
# control, its code is not rerun here.
#
# RESOURCES. This tokenizer reads the float32 one-hot delta blocks, the same host
# footprint as newmodels_full's (the OLD locusfusion input), not locusfusion_fixed's
# int8 symbol ids — so the allocations are newmodels_full's, not the fixed run's.
# The multi-drug loader alone peaks ~14 GB for TWO drugs; all 11 needs the 128G.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # project root
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"
ARMS="${ARMS:-sd sd19 md}"

TRAIN=(--epochs 300 --patience 30 --min-epochs 50)
KEEP=(--save-weights best)
MODSETS=(dna dna_protein dna_biophysical dna_regulatory all_modalities)
CELLS=(); for m in "${MODSETS[@]}"; do CELLS+=("${m}__locusfusion_v2"); done

# byte-identical to the list locusfusion_fixed/submit_sd19.sh and
# newmodels_full/submit.sh pin, so the arms are locus-matched by construction
LOCI=(--loci eis embA embB embC ethA ethR fabG1 gid gyrA gyrB inhA katG pncA
      rpoB rpoC rpsL rrl rrs tlyA)

for arm in $ARMS; do
  case "$arm" in
    sd)
      echo "### locusfusion_v2, single-drug, per-drug loci: 5 cells x 11 drugs = 55 jobs"
      python scripts/sbatch_all_runs.py $DRY_FLAG \
          --experiments "${CELLS[@]}" --drugs all \
          "${TRAIN[@]}" "${KEEP[@]}" \
          --run-prefix "locusfusion_v2/sd_" \
          --mem 64G --cpus 4 --gpus 1 --time 24:00:00 --constraint vram23 ;;
    sd19)
      echo "### locusfusion_v2, single-drug, all 19 loci: 5 cells x 11 drugs = 55 jobs"
      python scripts/sbatch_all_runs.py $DRY_FLAG \
          --experiments "${CELLS[@]}" --drugs all \
          "${LOCI[@]}" "${TRAIN[@]}" "${KEEP[@]}" \
          --run-prefix "locusfusion_v2/sd19_" \
          --mem 128G --cpus 4 --gpus 1 --time 36:00:00 --constraint vram23 ;;
    md)
      echo "### locusfusion_v2, multi-drug, union of per-drug loci: 5 jobs"
      python scripts/sbatch_all_runs.py $DRY_FLAG --multidrug \
          --experiments "${CELLS[@]}" \
          "${TRAIN[@]}" "${KEEP[@]}" \
          --run-prefix "locusfusion_v2/md_" \
          --mem 128G --cpus 6 --gpus 1 --time 36:00:00 --constraint vram23 ;;
    *) echo "unknown arm '$arm' (use: sd sd19 md)"; exit 1 ;;
  esac
done

[ -z "$DRY_FLAG" ] && echo && echo "submitted. Monitor: squeue -u \$USER" || true

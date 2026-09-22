#!/bin/bash
# locusfusion_fixed, the 19-locus arm — the rebuilt tokenizer at the locus set the
# project's headline numbers live at.
#
# Run from the project root:  bash results/experiments/locusfusion_fixed/submit_sd19.sh
# Dry run:                    DRY=1 bash .../submit_sd19.sh
#
# WHY THIS EXISTS. `locusfusion_fixed` (2026-09-01) rebuilt the tokenizer and ran
# only the PER-DRUG-loci arm, where it was worth a uniform +0.0173 across all 55
# cells, every one of them improved, and every modality set's 95% bootstrap
# interval excluding zero (`notebooks/claims.ipynb`, A5). The 19-locus cells were
# never rerun, so `newmodels_full/sd19_*` -- which is what Figures 1, 4 and A2 of
# that notebook quote as locusfusion's headline 0.9242 -- is still measuring the
# OLD tokenizer, with:
#
#   * a nucleotide token placed at column/3 minus a LEARNED per-locus scalar
#     that never moved off its zero init, putting katG S315 43 codons away from
#     its own protein token (rpoB S450: +52.7, pncA S65: +62.7);
#   * a codon phase that was `column % 3`, i.e. a per-locus rotation;
#   * an N call and a post-stop residue indistinguishable from a match.
#
# See ../CODE_CHANGES_20260901.md for the full accounting.
#
# MATCHED CONTROLS, both already run at exactly this protocol:
#   results/experiments/newmodels_full/sd19_*__locusfusion   the same cells, old tokenizer
#   results/experiments/alllocus_run_v2/*__mdcnn             macro CV 0.9246
# Same 300 epochs / patience 30 / --min-epochs 50 / --save-weights best / seed,
# the same explicit 19 loci, the same five modality sets. `--arch` is not even
# different: the only thing that changed is the tokenizer underneath it.
#
# WHAT TO READ WHEN IT LANDS, in order:
#   1. macro CV against newmodels_full/sd19_* -- does the +0.017 transfer?
#   2. the fold-PAIRED difference against alllocus_run_v2/all_modalities__mdcnn
#      (0.9246). The per-drug loci arm tied mdcnn; if the +0.017 transfers here
#      this is the first cell in the project that beats it outside noise.
#   3. read-out attention over 19 keys. At per-drug loci there were only 2 keys,
#      so KL 0.0308 from uniform was a weak test; 19 keys is the real one, and
#      `token_signal`'s failure mode was attention pinned at exactly 1/n.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # project root
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"

TRAIN=(--epochs 300 --patience 30 --min-epochs 50)
KEEP=(--save-weights best)
MODSETS=(dna dna_protein dna_biophysical dna_regulatory all_modalities)

# Byte-identical to the list newmodels_full/submit.sh pins, so the two arms are
# locus-matched by construction rather than by what happens to be on disk.
LOCI=(--loci eis embA embB embC ethA ethR fabG1 gid gyrA gyrB inhA katG pncA
      rpoB rpoC rpsL rrl rrs tlyA)

CELLS=(); for m in "${MODSETS[@]}"; do CELLS+=("${m}__locusfusion"); done

# Resources: newmodels_full's sd19 arm asked for 128G/36h on the OLD tokenizer
# and its largest cell took 3,388 s. The rebuilt input is int8 symbol ids rather
# than float32 one-hots (5-20x leaner in host memory) and the per-drug-loci arm
# ran 44-375 s per cell against that arm's 2,460 s. 64G/12h is therefore already
# generous; asking for less than the control does not make the comparison
# unfair, it just stops the jobs queueing behind an allocation they never use.
echo "### locusfusion_fixed, single-drug, all 19 loci: 5 cells x 11 drugs = 55 jobs"
python scripts/sbatch_all_runs.py $DRY_FLAG \
    --experiments "${CELLS[@]}" --drugs all \
    "${LOCI[@]}" "${TRAIN[@]}" "${KEEP[@]}" \
    --run-prefix "locusfusion_fixed/sd19_" \
    --mem 64G --cpus 4 --gpus 1 --time 12:00:00 --constraint vram23

echo
echo "submitted. Monitor: squeue -u \$USER"

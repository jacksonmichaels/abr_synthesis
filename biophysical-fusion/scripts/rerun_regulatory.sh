#!/bin/bash
# Re-run the 8 regulatory-containing configs (single-drug + joint) under the
# current rule: regulatory regions are intersected with the loaded loci. The
# results already in those folders were produced with the full WHO region set,
# so they are replaced rather than merged.
#
# Run AFTER the rest of the sweep has drained — the joint cells write into the
# same folders, so overlapping a re-run with them would race.
set -euo pipefail
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
source ~/.bashrc; conda activate abr_env

EXPERIMENTS="dna_regulatory__late_fusion dna_regulatory__mdcnn dna_regulatory__setfusion \
dna_regulatory__cisfusion all_modalities__late_fusion all_modalities__mdcnn \
all_modalities__setfusion all_modalities__cisfusion"

for e in $EXPERIMENTS; do
  rm -rf "results/experiments/full_run/$e" "results/experiments/full_run/multidrug_$e"
done

python scripts/sbatch_all_runs.py --experiments $EXPERIMENTS --drugs all \
    --epochs 150 --run-prefix "full_run/" --mem 48G --time 08:00:00 --delay 0.4
python scripts/sbatch_all_runs.py --experiments $EXPERIMENTS --multidrug \
    --epochs 150 --run-prefix "full_run/" --mem 64G --time 16:00:00 --cpus 6 --delay 0.4

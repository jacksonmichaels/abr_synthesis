#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -p cpu-preempt
#SBATCH -t 01:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/gate_who_t12-%j.out
#SBATCH --job-name=abr_gate_who_t12
# Runs after the 35 gene_msa extractions (afterok). Validates the FASTAs, runs the
# tests and a real single-drug load, then submits results/experiments/who_t12.
set -euo pipefail
source ~/.bashrc
conda activate abr_env
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python scripts/check_who_t12.py
python tests/test_locusfusion_v2.py | tail -1
python - <<'PY'
from bigtb_ref import REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR
from datasets import load_dataset
from models import LocusFusionV2Net
G = "/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/gene_msa/who_t12_genotypes"
d = load_dataset("ISONIAZID", ["dna", "protein", "biophysical", "regulatory"], G, REAL_PHENOTYPE_CSV,
                 regulatory_dir=REAL_REGULATORY_DIR, per_modality_branch=False, delta=True,
                 protein_frame="reference", locus_set="who_t12", verbose=True)
net = LocusFusionV2Net.from_blocks(d.blocks, n_drugs=1)
unplaced = [l for l, f in net.frames.items() if not f["genome_anchored"]]
print("ISONIAZID who_t12 loci:", net.loci, "| isolates", d.n, "| not genome-anchored:", unplaced)
assert len(net.loci) == 14 and not unplaced
PY
echo "GATE PASSED -- submitting who_t12"
bash results/experiments/who_t12/submit.sh

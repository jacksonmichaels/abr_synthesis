#!/bin/bash
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 16:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/multidrug_dna_mdcnn-%j.out
#SBATCH --job-name=abr_md_dna_mdcnn

# Multi-drug DNA-only under BIG-TB's OWN topology (architecture-debt fix,
# TODO.md 2026-08-04): the 18 loci become channels on one zero-padded position
# axis and layer 1 is a 12-bp conv across all of them, instead of our per-locus
# branches with a 1x1 stem and a 137,952-wide flatten. 150 epochs to match their
# protocol (our early stopping still applies; the previous 60-epoch run peaked
# at the cap in 4/5 folds).
echo "Job: multidrug dna mdcnn 150ep  (host $(hostname))"
source ~/.bashrc
conda activate abr_env
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
mkdir -p slurm_logs

# --per-drug-loci pins the 18-locus union this run was launched with (job
# 62594967, 2026-08-04). The runner's default became "every curated locus on
# disk" (19, incl. fabG1) AFTER this job started; without the flag, re-running
# this script would silently fold hypothesis 2 into the architecture result.
python -u scripts/run_multidrug.py --real --modalities dna --drugs all --device cuda \
    --arch mdcnn --epochs 150 --batch-size 128 --per-drug-loci \
    --run-name multidrug_dna_mdcnn150

echo "Done: multidrug dna mdcnn"

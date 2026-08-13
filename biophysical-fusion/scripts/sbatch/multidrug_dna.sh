#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/multidrug_dna-%j.out
#SBATCH --job-name=abr_multidrug_dna

echo "Job: multidrug dna  (host $(hostname))"
source ~/.bashrc
conda activate abr_env
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
mkdir -p slurm_logs

python -u scripts/run_multidrug.py --real --modalities dna --drugs all --device cuda \
    --epochs 60 --run-name multidrug_dna_all

echo "Done: multidrug dna"

#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -p cpu-preempt
#SBATCH -t 00:15:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/compare_structure-%j.out
#SBATCH --job-name=abr_compare_structure
# Runs after the 11 structure jobs (submitted with --dependency=afterany:...).
source ~/.bashrc
conda activate abr_env
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python results/experiments/locusfusion_v2_structure/compare.py

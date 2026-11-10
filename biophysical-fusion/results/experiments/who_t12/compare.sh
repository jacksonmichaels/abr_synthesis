#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -p cpu-preempt
#SBATCH -t 00:15:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/compare_who_t12-%j.out
#SBATCH --job-name=abr_compare_who_t12
# Runs after the 11 who_t12 jobs (submitted with --dependency=afterany:...).
source ~/.bashrc
conda activate abr_env
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
python results/experiments/who_t12/compare.py

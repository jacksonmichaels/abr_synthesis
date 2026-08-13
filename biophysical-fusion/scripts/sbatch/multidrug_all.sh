#!/bin/bash
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 16:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/multidrug_all-%j.out
#SBATCH --job-name=abr_multidrug_all

# Multi-drug MULTI-MODAL, one branch PER MODALITY (4 branches: dna/protein/
# biophysical/regulatory, each = its loci concatenated). Light enough for a
# normal GPU + full batch (was ~103 branches under the per-locus layout).
echo "Job: multidrug all-modalities (per-modality branches)  (host $(hostname))"
source ~/.bashrc
conda activate abr_env
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion
mkdir -p slurm_logs

python -u scripts/run_multidrug.py --real --modalities all --drugs all --device cuda \
    --epochs 60 --batch-size 128 --run-name multidrug_all_modalities

echo "Done: multidrug all-modalities"

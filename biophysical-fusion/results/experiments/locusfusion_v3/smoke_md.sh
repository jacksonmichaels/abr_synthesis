#!/bin/bash
#SBATCH --account=pi_annagreen_umass_edu
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -p cpu-preempt
#SBATCH -t 04:00:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/results/experiments/locusfusion_v3/smoke_md-%j.log
#SBATCH --job-name=abr_v3_md_smoke
# Real-data smoke test of the multi-drug v3 path (2 drugs, 2 epochs, 2 folds, CPU),
# from the frozen v3md snapshot. Not a result; output goes to locusfusion_v3_smoke/.
set -euo pipefail
source ~/.bashrc
conda activate abr_env
export ABR_GENOTYPE_DIR=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/genomic_data/aligned ABR_PHENOTYPE_CSV=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/phenotype_data/master_resistance_table.csv ABR_GENBANK=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot/genbank/genome.gbff
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/v3md_snapshot/biophysical-fusion
python -u scripts/run_multidrug.py --real --device cpu --arch locusfusion_v2 --lf-unknown-symbols \
  --modalities dna protein biophysical regulatory --drugs LEVOFLOXACIN MOXIFLOXACIN \
  --loci eis embA embB embC ethA ethR fabG1 gid gyrA gyrB inhA katG pncA rpoB rpoC rpsL rrl rrs tlyA \
  --epochs 2 --n-splits 2 --save-weights none --run-name locusfusion_v3_smoke/canary_md_all_modalities_unk__locusfusion_v2
echo SMOKE_OK

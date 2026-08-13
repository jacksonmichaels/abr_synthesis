#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -p cpu-preempt
#SBATCH -t 01:30:00
#SBATCH -o /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion/slurm_logs/trace_models-%j.out
#SBATCH --job-name=abr_trace_models

# Dataflow traces for every live model on a REAL isolate. CPU-only (one batch
# of 1 — no GPU needed); the memory is for the multi-drug load, which
# materialises every modality of all 11 drugs' loci for ~17.9k isolates.
# cpu-preempt, not cpu: the cpu partition was wedged on ReqNodeNotAvail and
# this is a ~20 min job, cheap to lose and resubmit if preempted.
echo "Job: model dataflow traces  (host $(hostname))"
source ~/.bashrc
conda activate abr_env
cd /scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/biophysical-fusion

# default: every trace, single-drug = ISONIAZID, multi-drug = all 11 drugs.
# Extra args passed to sbatch land here, e.g.
#   sbatch scripts/sbatch/trace_models.sh --traces sd_setfusion md_setfusion
python -u scripts/trace_models.py --real --drug ISONIAZID "$@"

echo "Done: model dataflow traces"

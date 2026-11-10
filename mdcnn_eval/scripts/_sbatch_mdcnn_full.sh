#!/bin/bash
#SBATCH -c 20
#SBATCH --mem=400G
#SBATCH -p cpu
#SBATCH -t 16:00:00
#SBATCH -o /home/jacksonmicha_umass_edu/abr_workspace/mdcnn_eval/logs/mdcnn_full-%j.out
#SBATCH -e /home/jacksonmicha_umass_edu/abr_workspace/mdcnn_eval/logs/mdcnn_full-%j.err
#SBATCH --job-name=mdcnn_full

# BIG-TB MD-CNN (multi-drug, 13 antibiotics) full reproduction run:
#   stage 1: 5-fold cross-validation, 150 epochs/fold  -> auc.csv, cv_split_*_auc.csv
#   stage 2: held-out test-set evaluation of the cv_split_4 model -> test_set_auc.csv
#
# CPU-only by design. The authors' own run (job 49009464, gpu054) logged
# "Could not find cuda drivers ... GPU will not be used" and finished all
# 5x150 epochs in 4h55m on 20 cores with 226 GB peak RSS -- their TF 2.14 env
# ships cuDNN 9 (TF 2.14 needs libcudnn.so.8), so it never had a usable GPU.
# Matching that setup keeps this run numerically comparable to their published
# numbers; -p cpu also schedules far faster than a 400 GB GPU node.
echo "Job: MD-CNN full crossval+eval  (host $(hostname), $(date))"

export TF_ENABLE_ONEDNN_OPTS=0
CONDA_ROOT=/work/pi_annagreen_umass_edu/saishradha/miniconda3
PY=$CONDA_ROOT/envs/cnn/bin/python   # authors' env: py3.9, TF/keras 2.14, sparse 0.14, sklearn 1.3.1

cd /home/jacksonmicha_umass_edu/abr_workspace/mdcnn_eval/model_training

echo "=== stage 1/2: 5-fold cross-validation ==="
$PY -u run_MDCNN_ccp_crossval.py ../parameter_files/mdcnn_crossval.txt
cv_rc=$?
echo "crossval exit code: $cv_rc"
if [ $cv_rc -ne 0 ]; then
    echo "crossval failed -- skipping test-set evaluation"
    exit $cv_rc
fi

echo "=== stage 2/2: held-out test-set evaluation (cv_split_4 model) ==="
$PY -u run_MDCNN_ccp_eval.py ../parameter_files/mdcnn_eval.txt
echo "eval exit code: $?"

echo "Done: MD-CNN full run ($(date))"

#!/bin/bash
#SBATCH -c 20
#SBATCH --mem=400G
#SBATCH -p cpu
#SBATCH -t 12:00:00
#SBATCH -o /home/jacksonmicha_umass_edu/abr_workspace/mdcnn_eval/logs/mdcnn_fold4_eval-%j.out
#SBATCH -e /home/jacksonmicha_umass_edu/abr_workspace/mdcnn_eval/logs/mdcnn_fold4_eval-%j.err
#SBATCH --job-name=mdcnn_fold4

# Finish the MD-CNN reproduction that job 62593892 left partial.
#
# That job (-t 16:00:00) completed folds 0-3 and was cancelled by the wall clock
# 48 epochs into fold 4, so it never wrote the aggregate auc.csv and never
# reached stage 2. Observed timings from its log: ~45 min data load, then
# 3h31m-3h35m per fold -- five folds plus eval needed ~19 h, not 16.
#
#   stage 1: fold 4 only  (mdcnn_crossval_fold4.txt sets start_cv_fold: 4)
#   stage 2: merge cv_split_3_auc.csv + cv_split_4_auc.csv -> auc.csv
#   stage 3: held-out test-set evaluation of the cv_split_4 model
#
# Budget: ~45 min load + ~3h35m fold + eval (loads again) ~= 6 h. 12 h is 2x that.
#
# CPU-only by design, same as the original: the authors' own run logged
# "Could not find cuda drivers ... GPU will not be used" (their TF 2.14 env ships
# cuDNN 9 while TF 2.14 needs libcudnn.so.8), so matching that keeps the numbers
# comparable. -p cpu also schedules far faster than a 400 GB GPU node.
echo "Job: MD-CNN fold 4 + eval  (host $(hostname), $(date))"

export TF_ENABLE_ONEDNN_OPTS=0
CONDA_ROOT=/work/pi_annagreen_umass_edu/saishradha/miniconda3
PY=$CONDA_ROOT/envs/cnn/bin/python   # authors' env: py3.9, TF/keras 2.14, sparse 0.14, sklearn 1.3.1

ROOT=/home/jacksonmicha_umass_edu/abr_workspace/mdcnn_eval
RUN_DIR=$ROOT/training_output/repro_filter12_epoch150

# Refuse to start if the folds we intend to reuse are not actually there --
# without them the merge below cannot produce a complete auc.csv.
for k in 0 1 2 3; do
    if [ ! -f "$RUN_DIR/cv_split_${k}_auc.csv" ]; then
        echo "FATAL: $RUN_DIR/cv_split_${k}_auc.csv missing -- rerun all five folds"
        echo "       with _sbatch_mdcnn_full.sh (-t 24:00:00) instead of this script."
        exit 1
    fi
done

cd $ROOT/model_training

echo "=== stage 1/3: cross-validation fold 4 (folds 0-3 reused from disk) ==="
$PY -u run_MDCNN_ccp_crossval.py ../parameter_files/mdcnn_crossval_fold4.txt
cv_rc=$?
echo "crossval exit code: $cv_rc"
if [ $cv_rc -ne 0 ]; then
    echo "fold 4 failed -- skipping merge and test-set evaluation"
    exit $cv_rc
fi

# The resumed run starts from an empty results frame, so the auc.csv it just
# wrote holds fold 4 alone. Overwrite it with all five folds.
echo "=== stage 2/3: merge folds 0-3 + fold 4 -> auc.csv ==="
$PY -u $ROOT/scripts/merge_cv_auc.py $RUN_DIR
merge_rc=$?
echo "merge exit code: $merge_rc"
if [ $merge_rc -ne 0 ]; then
    echo "merge failed -- auc.csv is NOT complete; not proceeding to eval"
    exit $merge_rc
fi

echo "=== stage 3/3: held-out test-set evaluation (cv_split_4 model) ==="
$PY -u run_MDCNN_ccp_eval.py ../parameter_files/mdcnn_eval.txt
echo "eval exit code: $?"

echo "Done: MD-CNN fold 4 + eval ($(date))"

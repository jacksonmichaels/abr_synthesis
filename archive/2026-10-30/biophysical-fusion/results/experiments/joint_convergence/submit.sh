#!/bin/bash
# joint_convergence — 6-arm convergence pilot on ONE joint cell.
#
# Run from the project root:  bash results/experiments/joint_convergence/submit.sh
# Dry run:                    DRY=1 bash .../submit.sh
#
# Cell: multidrug dna__late_fusion — the §2 reference point (full_run joint CV
# 0.9132, -0.0116 vs MD-CNN) and the cell where 40% of folds hit the 150-epoch
# cap with best_epoch 120-148. Every arm changes exactly ONE thing against
# a0_control, so each margin is attributable.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # project root
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry_run"

COMMON=(--multidrug --experiments dna__late_fusion
        --save-weights all --mem 64G --cpus 6 --gpus 1)

sub () {  # sub <arm> <slurm-time> [extra flags...]
    local arm="$1" tlimit="$2"; shift 2
    echo "=== submitting $arm"
    python scripts/sbatch_all_runs.py "${COMMON[@]}" $DRY_FLAG \
        --run-prefix "joint_convergence/${arm}_" --time "$tlimit" "$@"
}

# a0: exact full_run settings. Duplicates full_run's cell on purpose — it
#     empirically confirms the new defaults are unchanged, and it is the only
#     baseline that has saved weights.
sub a0_control       20:00:00 --epochs 150

# a1: is it an epoch-budget problem? fold 0 went ep50 0.8727 -> ep100 0.8969
#     -> ep146 0.9100, still climbing at the cap. ~35h expected at 400.
sub a1_ep400         48:00:00 --epochs 400

# a2: or a learning-rate problem wearing an epoch-cap costume? exp(-9) ~ 1.2e-4
#     is ~10x below a normal Adam setting. If this reaches a0 in far fewer
#     epochs the whole grid gets CHEAPER, not more expensive.
sub a2_lr1e3         20:00:00 --epochs 150 --lr 1e-3

# a3: a2 + regularization. 46M params on 17.9k isolates with no dropout, no
#     weight decay and no norm; a faster LR may need it.
sub a3_lr1e3_reg     20:00:00 --epochs 150 --lr 1e-3 --dropout 0.3 --weight-decay 1e-4

# a4: was patience=15 firing spuriously? all_modalities__mdcnn fold 1 stopped at
#     epoch 36 scoring 0.8609 while its siblings reached 0.918.
sub a4_patience30    24:00:00 --epochs 150 --patience 30

# a5: the monitor is an unweighted mean over 11 drugs, so LEVOFLOXACIN (n=269,
#     ~15 resistant per val fold) supplies a ninth of the stop signal and mostly
#     noise. Excluded from the SIGNAL only; still trained and reported.
sub a5_monitor500    20:00:00 --epochs 150 --monitor-min-n 500

echo
echo "6 jobs submitted. Monitor: squeue -u \$USER"
echo "Weights: /project/pi_mfiterau_umass_edu/abr_model_weights/joint_convergence/"

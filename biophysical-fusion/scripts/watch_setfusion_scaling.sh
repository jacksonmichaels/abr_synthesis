#!/bin/bash
# Wait for a set of submitted setfusion_scaling jobs to leave the queue, then
# run the analysis. Exits 0 once the report is written.
#
#   bash scripts/watch_setfusion_scaling.sh <id-file> [scope] [poll-seconds]
#
# <id-file> is "JOBID run_name" per line — results/experiments/setfusion_scaling/
# joint_job_ids.txt is written by the submission step.
#
# Detached (survives this shell / an ssh disconnect):
#   setsid nohup bash scripts/watch_setfusion_scaling.sh \
#       results/experiments/setfusion_scaling/joint_job_ids.txt joint 600 \
#       > slurm_logs/watch_setfusion_scaling.log 2>&1 &
#
# Membership is tested against `squeue -u $USER` rather than `squeue --jobs=<ids>`
# on purpose: once a job is purged from the accounting window the latter errors
# out per id, and a watcher that mistakes an error for "still running" waits
# forever.
set -uo pipefail
cd "$(dirname "$0")/.."

ID_FILE="${1:?usage: watch_setfusion_scaling.sh <id-file> [scope] [poll-seconds]}"
SCOPE="${2:-joint}"
POLL="${3:-600}"
MAX_HOURS="${MAX_HOURS:-192}"          # 8 days: past the longest --time in the sweep

OUT_DIR="results/experiments/setfusion_scaling"
DONE_MARKER="$OUT_DIR/.watch_${SCOPE}_done"
STATUS="$OUT_DIR/.watch_${SCOPE}_status"

[ -f "$ID_FILE" ] || { echo "no id file: $ID_FILE"; exit 2; }
mapfile -t IDS < <(awk '{print $1}' "$ID_FILE")
TOTAL=${#IDS[@]}
echo "[watch] $TOTAL $SCOPE jobs from $ID_FILE; polling every ${POLL}s (max ${MAX_HOURS}h)"

remaining () {                          # how many of OUR ids are still queued
    local live count=0
    live=$(squeue -u "$USER" -h -o "%i" 2>/dev/null) || return 1
    for id in "${IDS[@]}"; do
        grep -qx "$id" <<<"$live" && count=$((count + 1))
    done
    echo "$count"
}

START=$(date +%s)
while :; do
    if ! LEFT=$(remaining); then
        # squeue itself failed (scheduler blip) — do NOT treat as finished
        echo "[watch] $(date '+%F %T') squeue unavailable, retrying"
        sleep "$POLL"; continue
    fi
    printf '%s  %s/%s still queued\n' "$(date '+%F %T')" "$LEFT" "$TOTAL" | tee "$STATUS"
    [ "$LEFT" -eq 0 ] && break
    ELAPSED_H=$(( ( $(date +%s) - START ) / 3600 ))
    if [ "$ELAPSED_H" -ge "$MAX_HOURS" ]; then
        echo "[watch] giving up after ${ELAPSED_H}h with $LEFT still queued"
        echo "TIMEOUT after ${ELAPSED_H}h, $LEFT jobs still queued" > "$DONE_MARKER"
        exit 3
    fi
    sleep "$POLL"
done

echo "[watch] all $TOTAL jobs left the queue; analysing"
# shellcheck disable=SC1090
source ~/.bashrc >/dev/null 2>&1
conda activate abr_env >/dev/null 2>&1
python scripts/analyze_setfusion_scaling.py --scope "$SCOPE" --write
RC=$?

# how many cells actually produced a result — a job can leave the queue by
# failing, and "no error" must not be inferred from "no longer running"
if [ "$SCOPE" = joint ]; then
    GOT=$(ls -d "$OUT_DIR"/*_multidrug_*__setfusion 2>/dev/null | wc -l)
    WROTE=$(ls "$OUT_DIR"/*_multidrug_*__setfusion/multidrug__*.json 2>/dev/null | wc -l)
else
    GOT=$(ls -d "$OUT_DIR"/*__setfusion 2>/dev/null | grep -vc multidrug)
    WROTE=$(ls "$OUT_DIR"/*__setfusion/[A-Z]*.json 2>/dev/null | wc -l)
fi
printf 'DONE %s: %s/%s job folders, %s result json, analysis rc=%s\n' \
    "$SCOPE" "$GOT" "$TOTAL" "$WROTE" "$RC" | tee "$DONE_MARKER"
exit 0

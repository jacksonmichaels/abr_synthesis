#!/bin/bash
# Freeze the code the locus_sets jobs run, like lineage_snapshot: a read-only copy of
# the python code + BASE_COMMIT + working_tree.patch; results/ and slurm_logs/ are
# symlinks back into the repo, data folders symlinks to the shared copies.
# Refuses to overwrite an existing snapshot (jobs may be reading it).
set -euo pipefail
W=/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis
REPO=$W/biophysical-fusion
DEST=$W/locus_sets_snapshot
[ -e "$DEST" ] && { echo "$DEST exists; move it aside first"; exit 1; }
mkdir -p "$DEST/biophysical-fusion"
git -C "$W" rev-parse HEAD > "$DEST/BASE_COMMIT"
git -C "$W" diff HEAD -- biophysical-fusion > "$DEST/working_tree.patch"
git -C "$W" status --short -- biophysical-fusion > "$DEST/git_status.txt"
cd "$REPO"
cp -p ./*.py "$DEST/biophysical-fusion/"
for d in datasets models training scripts; do
  rsync -a --exclude __pycache__ "$d/" "$DEST/biophysical-fusion/$d/"
done
ln -s "$REPO/results"    "$DEST/biophysical-fusion/results"
ln -s "$REPO/slurm_logs" "$DEST/biophysical-fusion/slurm_logs"
ln -s "$W/Big-TB-benchmark" "$DEST/Big-TB-benchmark"
ln -s "$W/structures"       "$DEST/structures"
chmod -R a-w "$DEST/biophysical-fusion"/{datasets,models,training,scripts} "$DEST/biophysical-fusion"/*.py
echo "snapshot at $DEST (base $(cat "$DEST/BASE_COMMIT"))"

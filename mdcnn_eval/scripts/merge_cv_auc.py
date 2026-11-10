#!/usr/bin/env python
"""Assemble the complete `auc.csv` for a MD-CNN run that was completed in stages.

`run_MDCNN_ccp_crossval.py` accumulates every fold into one in-memory `results`
frame and rewrites `cv_split_{k}_auc.csv` after each fold, so
`cv_split_3_auc.csv` already holds folds 0-3. A resumed run (`start_cv_fold: 4`)
starts from an EMPTY frame, so its `cv_split_4_auc.csv` — and the `auc.csv` it
writes on the way out — hold fold 4 alone. This concatenates the two and
renumbers the index, reproducing exactly the file a single uninterrupted run
would have written.

    python merge_cv_auc.py <run_output_dir>          # writes <dir>/auc.csv

Refuses to write unless the result is 5 distinct folds x 13 drugs = 65 rows.
"""
import csv
import os
import sys

COLUMNS = ["", "Validation Split #", "Algorithm", "Drug", "num_sensitive",
           "num_resistant", "AUC", "AUC_PR", "Threshold", "Spec", "Sens"]
N_DRUGS, N_FOLDS = 13, 5


def read(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"empty: {path}")
    got = list(rows[0].keys())
    if got != COLUMNS:
        sys.exit(f"unexpected columns in {path}:\n  got      {got}\n  expected {COLUMNS}")
    return rows


def main(run_dir):
    parts = [os.path.join(run_dir, "cv_split_3_auc.csv"),   # folds 0-3, cumulative
             os.path.join(run_dir, "cv_split_4_auc.csv")]   # fold 4, from the resume
    for p in parts:
        if not os.path.isfile(p):
            sys.exit(f"missing input: {p}")

    rows = [r for p in parts for r in read(p)]

    folds = sorted({r["Validation Split #"] for r in rows})
    if len(rows) != N_DRUGS * N_FOLDS or len(folds) != N_FOLDS:
        sys.exit(f"refusing to write: got {len(rows)} rows over {len(folds)} folds "
                 f"({', '.join(folds)}), expected {N_DRUGS * N_FOLDS} over {N_FOLDS}")

    out = os.path.join(run_dir, "auc.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for i, r in enumerate(rows):
            r[""] = str(i)
            w.writerow(r)
    print(f"wrote {out}: {len(rows)} rows, folds {', '.join(folds)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])

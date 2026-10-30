"""The BIG-TB baselines — the single source of truth for both of them.

Extracted from ``scripts/build_full_run_viewer.py`` so that more than one
notebook can compare against these numbers without a second copy of the tables
existing anywhere. (A duplicate copy in ``notebooks/results_viewer.ipynb`` is
exactly what the 2026-08-13 cleanup deleted; do not reintroduce one.)

    from bigtb_baselines import SD_BASE_CV, MD_BASE_CV, SDCNN_CV_SD

``MD_BASE_*`` is read from the BIG-TB authors' own run output on the
``pi_annagreen`` volume at import time, so this module needs cluster access.
"""
import os
import pandas as pd
from pathlib import Path

# --- BIG-TB SD-CNN (single-drug baseline) ----------------------------------
# The authors' own saved best model per drug, scored on the held-out split.
#            drug : (test_AUC, CV_mean)
SDCNN_BASE = {
    "AMIKACIN":     (0.885, 0.859),
    "CAPREOMYCIN":  (0.815, 0.847),
    "ETHAMBUTOL":   (0.925, 0.926),
    "ETHIONAMIDE":  (0.644, 0.622),
    "ISONIAZID":    (0.917, 0.912),
    "KANAMYCIN":    (0.855, 0.867),
    "LEVOFLOXACIN": (0.885, 0.850),
    "MOXIFLOXACIN": (0.825, 0.819),
    "PYRAZINAMIDE": (0.922, 0.913),
    "RIFAMPICIN":   (0.980, 0.972),
    "STREPTOMYCIN": (0.924, 0.913),
}
# Back-compatible alias for callers that still import the old name.
SDCNN_CLEAN = SDCNN_BASE
# Fold-to-fold spread of the SD-CNN's own 5-fold CV (Tasmin et al., Table 14).
# ETHIONAMIDE deliberately absent: its Table-14 row contradicts the authors'
# auc.csv, so no SD is trusted for it.
SDCNN_CV_SD = {
    "AMIKACIN": 0.0195, "CAPREOMYCIN": 0.0242, "ETHAMBUTOL": 0.0032,
    "ISONIAZID": 0.0119, "KANAMYCIN": 0.0294, "LEVOFLOXACIN": 0.0661,
    "MOXIFLOXACIN": 0.0272, "PYRAZINAMIDE": 0.0154, "RIFAMPICIN": 0.0014,
    "STREPTOMYCIN": 0.0120,
}

# --- BIG-TB MD-CNN (multi-drug baseline) ------------------------------------
# Read from the authors' OWN RUN OUTPUT, not transcribed from the paper:
#   cv mean / SD : auc.csv                     (5 folds x 13 drugs)
#   test AUC     : test_set_auc_cv_4_best.csv
# This is the run mdcnn_eval/README.md names as the comparison target -- same
# params as ours (filter_size 12, N_epochs 150, random_seed 1, test_size 0.2).
#
# Why not Tasmin et al. Table 14, which this used to hardcode: that table does
# not reproduce from any MD-CNN run on the volume. Only AMIKACIN matches it on
# both mean and SD; ETHIONAMIDE's tabled 0.9161 appears nowhere, at any fold of
# any run, while this log says 0.7955; CAP / KAN / MOXI / PZA are off by
# 0.02-0.04 in BOTH directions. Table 4's test AUCs fare much better (8/11
# reproduce to within 0.002) but are taken from the log too, so the whole row
# has a single provenance.
#
# Protocol, unchanged: their crossval is a non-stratified 80/20 seed 42 +
# KFold(5) -- the SAME as run_multidrug_cv, so CV-vs-CV is the matched
# comparison. Their `assess` trains on a predefined cohort and tests on every
# other isolate, so test-vs-test stays indicative only. Judge on CV.
# ABR_MDCNN_RUN_DIR points at a copy of the two CSVs (data_snapshot/mdcnn_baseline)
# for jobs on nodes where the /project mount is unreliable.
MDCNN_RUN_DIR = Path(os.environ.get("ABR_MDCNN_RUN_DIR",
    "/project/pi_annagreen_umass_edu/saishradha/project_data_curation"
    "/benchmarking/MTB-CNN/training_output"
    "/results_ccp_filter12_epoch150_sbatch_18_Nov"))


def load_mdcnn_baseline(run_dir=MDCNN_RUN_DIR, drugs=None):
    """drug -> (test_AUC, cv_mean, cv_std), straight from the MD-CNN run output.

    `drugs` restricts the result to the drugs we score (auc.csv also carries
    CIPROFLOXACIN and OFLOXACIN, which are not in our 11)."""
    cv_path, test_path = run_dir / "auc.csv", run_dir / "test_set_auc_cv_4_best.csv"
    for p in (cv_path, test_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"MD-CNN baseline log missing: {p}\n"
                "These are the BIG-TB authors' own outputs on the pi_annagreen "
                "volume -- this notebook needs cluster access to read them.")
    folds = pd.read_csv(cv_path).groupby("Drug")["AUC"]
    mean, sd, n = folds.mean(), folds.std(), folds.size()   # sd is ddof=1
    test = pd.read_csv(test_path).set_index("Drug")["AUC"]
    keys = sorted(set(mean.index) & set(test.index)
                  & (set(drugs) if drugs is not None else set(mean.index)))
    if drugs is not None and len(keys) < len(set(drugs)):
        raise KeyError(f"MD-CNN log is missing {sorted(set(drugs) - set(keys))}")
    print(f"MD-CNN baseline <- {run_dir.name}/auc.csv "
          f"({len(keys)} drugs, {int(n[keys].min())}-{int(n[keys].max())} folds)")
    return {d: (round(float(test[d]), 4), round(float(mean[d]), 4),
                round(float(sd[d]), 4)) for d in keys}


#                     drug : (test_AUC, cv_mean, cv_std)
BIGTB_MDCNN = load_mdcnn_baseline(drugs=SDCNN_CLEAN)

# Nothing is flagged any more. The CV column above IS the authors' auc.csv, so
# the table-vs-log contradiction that justified excluding ETHIONAMIDE is gone.
# Kept as an empty set so the "excl (warn)" plumbing downstream still resolves;
# those columns now equal the plain delta.
MDCNN_CV_SUSPECT = set()
ALL_DRUGS = sorted(SDCNN_CLEAN)

SD_BASE_CV   = pd.Series({d: v[1] for d, v in SDCNN_CLEAN.items()})
SD_BASE_TEST = pd.Series({d: v[0] for d, v in SDCNN_CLEAN.items()})
MD_BASE_CV   = pd.Series({d: v[1] for d, v in BIGTB_MDCNN.items()})
MD_BASE_TEST = pd.Series({d: v[0] for d, v in BIGTB_MDCNN.items()})

"""BIG-TB's own significance test, applied to this project's cells.

Tasmin et al. Table 14 and §2 ("Statistical testing across models"):

    paired Wilcoxon signed-rank (two-sided) across identical CV folds,
    mean +/- SD over the 5 folds, 95% CI as mu +/- 1.96 * sigma / sqrt(n),
    no multiple-testing correction.

Their per-drug p-values are computed at n = 5 (the folds of one drug). Every
value in their Table 14 falls in {0.063, 0.125, 0.188, 0.313, 0.438, 0.625,
0.813, 1.000}, which is the complete discrete support of the exact two-sided
signed-rank statistic at n = 5 -- so `method="exact"` reproduces their column.

**The n = 5 arm cannot reach p < 0.05.** The smallest attainable two-sided
p is 2/2^5 = 0.0625, returned when all five folds move the same way. Read
0.063 as "unanimous across folds", not as "not significant".

Their text also quotes aggregate p-values across drugs (e.g. p = 0.0009 for
MD-CNN vs MD-DNABERT-CNN), which n = 5 cannot produce. `--pool` reproduces
that arm by stacking all 11 x 5 = 55 fold-paired differences into one test.

Fold pairing is only valid inside the single-drug family: every such run uses
StratifiedKFold(5, shuffle=True, random_state=42) over the same cohort. It is
NOT valid against the published MD-CNN, whose crossval is a non-stratified
KFold at a different seed -- those stay paired over drugs alone.

    python scripts/bigtb_paired_test.py
    python scripts/bigtb_paired_test.py --pool
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from results_frame import load_frame

LOCI19 = "single-drug, all loci"

# (label, (model, modset, run)) -- `run` pins the cell where a model has two.
CELLS = {
    "lasso_onehot":   ("lasso_onehot", "dna", None),
    "lasso_delta":    ("lasso_delta", "dna", None),
    "mdcnn":          ("mdcnn", "all_modalities", None),
    "cisfusion":      ("cisfusion", "all_modalities", None),
    "late_fusion":    ("late_fusion", "all_modalities", None),
    "locusfusion":    ("locusfusion", "all_modalities", "locusfusion_fixed"),
    "locusfusion_old":("locusfusion", "all_modalities", "newmodels_full"),
}

# Every comparison the paper makes, as (arm, baseline).
PAIRS = [
    ("lasso_onehot", "mdcnn"),
    ("locusfusion", "lasso_onehot"),
    ("locusfusion", "mdcnn"),
    ("locusfusion", "locusfusion_old"),
    ("late_fusion", "mdcnn"),
    ("cisfusion", "mdcnn"),
    ("lasso_onehot", "lasso_delta"),
]


def folds_of(frame, label):
    model, modset, run = CELLS[label]
    q = frame[(frame.task == LOCI19) & (frame.model == model)
              & (frame.modset == modset)]
    if run is not None:
        q = q[q.run == run]
    q = q.drop_duplicates("drug").set_index("drug")
    s = q.fold_aucs
    bad = [d for d, t in s.items() if not isinstance(t, tuple) or len(t) != 5]
    if bad:
        raise ValueError(f"{label}: {bad} do not have 5 folds")
    return s.map(np.asarray)


def bigtb_ci(x):
    """Their CI: mu +/- 1.96 * sigma / sqrt(n). sigma is ddof=1, as their SD."""
    x = np.asarray(x, dtype=float)
    h = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def wilcoxon_exact(d):
    """Two-sided exact signed-rank. Returns NaN when every difference is zero."""
    d = np.asarray(d, dtype=float)
    if np.allclose(d, 0):
        return np.nan
    return stats.wilcoxon(d, method="exact", zero_method="wilcox").pvalue


def per_drug(frame, arm, base):
    a, b = folds_of(frame, arm), folds_of(frame, base)
    rows = []
    for drug in sorted(a.index):
        xa, xb = a[drug], b[drug]
        lo_a, hi_a = bigtb_ci(xa)
        d = xa - xb
        rows.append({
            "drug": drug,
            "mean_AUC": xa.mean(), "SD": xa.std(ddof=1),
            "CI": f"[{lo_a:.4f}, {hi_a:.4f}]",
            "base_mean": xb.mean(),
            "delta": d.mean(),
            "folds_up": int((d > 0).sum()),
            "p_vs_base": wilcoxon_exact(d),
        })
    return pd.DataFrame(rows).set_index("drug")


def pooled(frame, arm, base):
    """Their aggregate arm: all 11 x 5 fold-paired differences in one test."""
    a, b = folds_of(frame, arm), folds_of(frame, base)
    d = np.concatenate([a[k] - b[k] for k in sorted(a.index)])
    p = stats.wilcoxon(d, method="exact" if len(d) <= 25 else "auto",
                       zero_method="wilcox").pvalue
    return d, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="store_true",
                    help="also run the pooled 55-pair test their text quotes")
    args = ap.parse_args()

    frame, _ = load_frame()
    pd.set_option("display.width", 200)

    print("\nBIG-TB method: paired Wilcoxon (two-sided, exact) across identical "
          "CV folds,\nmean +/- SD over 5 folds, CI = mu +/- 1.96*sigma/sqrt(n), "
          "no multiple-testing correction.")
    print("At n = 5 the smallest attainable two-sided p is 0.0625, so no "
          "per-drug cell\ncan clear p < 0.05. 0.063 means all five folds "
          "moved the same way.\n")

    summary = []
    for arm, base in PAIRS:
        t = per_drug(frame, arm, base)
        n_unanimous = int((t.p_vs_base <= 0.0626).sum())
        print("=" * 96)
        print(f"{arm}  vs  {base}      (19 loci, single-drug)")
        print("=" * 96)
        print(t.round(4).to_string())
        print(f"  macro delta {t.delta.mean():+.4f} | drugs up "
              f"{int((t.delta > 0).sum())}/11 | folds unanimous "
              f"{n_unanimous}/11 | min p {t.p_vs_base.min():.3f}")
        row = {"arm": arm, "baseline": base, "macro_delta": t.delta.mean(),
               "drugs_up": int((t.delta > 0).sum()), "unanimous": n_unanimous,
               "min_p": t.p_vs_base.min()}
        if args.pool:
            d, p = pooled(frame, arm, base)
            row["pooled_n"] = len(d)
            row["pooled_p"] = p
            print(f"  pooled over {len(d)} fold-pairs: mean {d.mean():+.4f}, "
                  f"up {int((d > 0).sum())}/{len(d)}, Wilcoxon p = {p:.2e}")
        print()
        summary.append(row)

    print("=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(pd.DataFrame(summary).set_index(["arm", "baseline"]).round(5).to_string())


if __name__ == "__main__":
    main()

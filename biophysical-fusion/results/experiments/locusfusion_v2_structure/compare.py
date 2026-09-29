"""locusfusion_v2 without `structure` vs with it at each closeness decay length.

    python results/experiments/locusfusion_v2_structure/compare.py

Single-drug, per-drug loci, same protocol, seeds and folds in every arm (see
submit.sh / submit_lengths.sh). Reads each arm's summary.csv and per-fold AUCs
from each drug's JSON; writes compare.csv (per drug x arm) next to this file.
An arm with no summary.csv yet is skipped.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
CONTROL = "none"
ARMS = {"none": EXP / "locusfusion_v2" / "sd_all_modalities__locusfusion_v2",
        "10 Å": HERE / "sd_all_modalities_structure__locusfusion_v2",
        "20 Å": HERE / "sd_all_modalities_structure_L20__locusfusion_v2",
        "30 Å": HERE / "sd_all_modalities_structure_L30__locusfusion_v2"}


def fold_aucs(run_dir, drug):
    f = next(run_dir.glob(f"{drug}__*.json"), None)
    if f is None:
        return None
    return np.array([fd.get("auc") for fd in json.loads(f.read_text()).get("cv_folds", [])], float)


tabs = {a: pd.read_csv(d / "summary.csv").set_index("drug") for a, d in ARMS.items()
        if (d / "summary.csv").exists()}
if CONTROL not in tabs:
    raise SystemExit("no control summary.csv")
skipped = [a for a in ARMS if a not in tabs]
rows = []
for arm, t in tabs.items():
    if arm == CONTROL:
        continue
    for drug in sorted(set(t.index) & set(tabs[CONTROL].index)):
        c, x = tabs[CONTROL].loc[drug], t.loc[drug]
        fc, fx = fold_aucs(ARMS[CONTROL], drug), fold_aucs(ARMS[arm], drug)
        better = (f"{int((fx > fc).sum())}/{len(fx)}"
                  if fc is not None and fx is not None and len(fc) == len(fx) else "")
        rows.append({"arm": arm, "drug": drug, "n_valid": int(x["n_valid"]),
                     "cv_auc_none": c["cv_auc_mean"], "cv_auc": x["cv_auc_mean"],
                     "cv_delta": x["cv_auc_mean"] - c["cv_auc_mean"], "folds_better": better,
                     "test_auc_none": c["test_auc"], "test_auc": x["test_auc"],
                     "test_delta": x["test_auc"] - c["test_auc"]})
df = pd.DataFrame(rows)
df.to_csv(HERE / "compare.csv", index=False)
pd.set_option("display.width", 180)

wide = df.pivot(index="drug", columns="arm", values="cv_delta")
wide.insert(0, "cv_auc_none", tabs[CONTROL]["cv_auc_mean"].reindex(wide.index))
print("CV AUC of the control, and the change with structure at each decay length:\n")
print(wide.round(4).to_string())

try:
    from scipy.stats import wilcoxon
except Exception:  # noqa: BLE001
    wilcoxon = None
print("\narm    drugs  macro CV (none -> with)   delta    better  macro test delta  Wilcoxon p")
for arm, g in df.groupby("arm", sort=False):
    p = wilcoxon(g.cv_delta).pvalue if wilcoxon and len(g) > 5 else float("nan")
    print(f"{arm:6} {len(g):5d}  {g.cv_auc_none.mean():.4f} -> {g.cv_auc.mean():.4f}   "
          f"{g.cv_delta.mean():+.4f}   {(g.cv_delta > 0).sum():2d}/{len(g):<2d}   "
          f"{g.test_delta.mean():+.4f}           {p:.3g}")
    nolevo = g[g.drug != "LEVOFLOXACIN"]
    print(f"{'':6} (without LEVOFLOXACIN, n=269: CV delta {nolevo.cv_delta.mean():+.4f})")
if skipped:
    print(f"\narms with no summary.csv yet: {skipped}")

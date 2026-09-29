"""WHO tier 1+2 genes per drug vs BIG-TB's DRUG_TO_LOCI, locusfusion_v2, all
modalities, single-drug (same protocol, seeds and folds).

    python results/experiments/who_t12/compare.py      # writes compare.csv here
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CTRL = HERE.parent / "locusfusion_v2" / "sd_all_modalities__locusfusion_v2"
WHO = HERE / "sd_all_modalities_who_t12__locusfusion_v2"


def folds(d, drug):
    f = next(d.glob(f"{drug}__*.json"), None)
    return None if f is None else np.array([x["auc"] for x in json.loads(f.read_text())["cv_folds"]])


c = pd.read_csv(CTRL / "summary.csv").set_index("drug")
w = pd.read_csv(WHO / "summary.csv").set_index("drug")
rows = []
for drug in sorted(set(c.index) & set(w.index)):
    fc, fw = folds(CTRL, drug), folds(WHO, drug)
    rows.append({"drug": drug, "genes_bigtb": c.loc[drug, "genes"], "genes_who_t12": w.loc[drug, "genes"],
                 "cv_auc_bigtb": c.loc[drug, "cv_auc_mean"], "cv_auc_who_t12": w.loc[drug, "cv_auc_mean"],
                 "cv_delta": w.loc[drug, "cv_auc_mean"] - c.loc[drug, "cv_auc_mean"],
                 "folds_better": f"{int((fw > fc).sum())}/{len(fw)}" if fc is not None and fw is not None else "",
                 "test_auc_bigtb": c.loc[drug, "test_auc"], "test_auc_who_t12": w.loc[drug, "test_auc"],
                 "test_delta": w.loc[drug, "test_auc"] - c.loc[drug, "test_auc"]})
df = pd.DataFrame(rows).set_index("drug")
df.to_csv(HERE / "compare.csv")
pd.set_option("display.width", 200)
print(df.drop(columns=["genes_bigtb", "genes_who_t12"]).round(4).to_string())
print(f"\n{len(df)} drugs | macro CV {df.cv_auc_bigtb.mean():.4f} -> {df.cv_auc_who_t12.mean():.4f} "
      f"({df.cv_delta.mean():+.4f}), better {(df.cv_delta > 0).sum()}/{len(df)} | macro test "
      f"{df.test_delta.mean():+.4f}")
try:
    from scipy.stats import wilcoxon
    print(f"Wilcoxon on per-drug CV delta: p = {wilcoxon(df.cv_delta).pvalue:.3g}")
except Exception as e:  # noqa: BLE001
    print(f"(no Wilcoxon: {e})")

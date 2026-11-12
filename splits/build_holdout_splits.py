"""Leave-one-major-lineage-out splits, Sai's design (protein-tasks/lineage_split_utils.py
in Data-Curation-for-MTB__sm_lineage_aware_sdcnn), written in the split-file
format every runner here reads.

For each drug and each held-out major lineage L in 1-4:
  test  = isolates whose top-level `Lineage` is exactly L
  train = every other isolate with a lineage call (other major lineages, minor
          and mixed calls such as "1,4" or "BOV_AFRI")
  excluded = isolates with no lineage call
A (drug, L) cell is written only if train and test each hold >= --min-class
resistant and susceptible isolates (her DEFAULT_MIN_CLASS_COUNT, 50).

She trains on all of train with no validation; our models early-stop on a
validation fold, so train is cut into 5 CV folds by StratifiedKFold(5,
shuffle=True, random_state=42) on the label. TEST is the held-out lineage.

Writes, under --out (default paths.SPLIT_DIR):
  heldout_{L}/{DRUG}.csv       New_ID, Lineage, role (train/test/excluded), fold (-1 off train), R
  heldout_{L}/MULTIDRUG.csv    New_ID, Lineage, role, fold, one 1/0/-1 = R/S/missing column per drug;
                               cohort = isolates with an R/S call for >= 1 drug, folds
                               stratified on "resistant to >= 1 drug"
  summary.csv                  drug x L: n / R / S for train and test, feasible

    python splits/build_holdout_splits.py
    python -m stacked_cnn.train ... --split-dir results/splits/heldout_2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import LINEAGE_CSV, REAL_PHENOTYPE_CSV, SPLIT_DIR  # noqa: E402

DRUGS = ["AMIKACIN", "CAPREOMYCIN", "ETHAMBUTOL", "ETHIONAMIDE", "ISONIAZID", "KANAMYCIN",
         "LEVOFLOXACIN", "MOXIFLOXACIN", "PYRAZINAMIDE", "RIFAMPICIN", "STREPTOMYCIN"]
MAJOR_LINEAGES = ("1", "2", "3", "4")


def assign(ids, lineage, heldout, strat, seed, n_splits=5):
    """role and fold per isolate, in `ids` order."""
    lin = lineage.reindex(ids)
    role = np.where(lin.isna(), "excluded", np.where(lin == heldout, "test", "train"))
    fold = np.full(len(ids), -1)
    tr = np.nonzero(role == "train")[0]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf.split(tr, strat[tr])):
        fold[tr[va]] = k
    return lin.values, role, fold


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lineage-csv", default=LINEAGE_CSV)
    ap.add_argument("--phenotype-csv", default=REAL_PHENOTYPE_CSV)
    ap.add_argument("--min-class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=SPLIT_DIR)
    args = ap.parse_args()
    out = Path(args.out)

    lineage = (pd.read_csv(args.lineage_csv, dtype={"ROLLINGDB_ID": str, "Lineage": "string"})
               .set_index("ROLLINGDB_ID")["Lineage"].dropna().astype(str))
    ph = pd.read_csv(args.phenotype_csv, low_memory=False, dtype={"New_ID": str}).set_index("New_ID")
    ph = ph[~ph.index.duplicated()].sort_index()

    summary = []
    for heldout in MAJOR_LINEAGES:
        d_out = out / f"heldout_{heldout}"
        d_out.mkdir(parents=True, exist_ok=True)
        for drug in DRUGS:
            ids = ph.index[ph[drug].isin(["R", "S"])]
            y = (ph.loc[ids, drug] == "R").astype(int).values
            lin, role, fold = assign(ids, lineage, heldout, y, args.seed)
            counts = {f"{r}_{c}": int(((role == r) & (y == v)).sum())
                      for r in ("train", "test") for c, v in (("R", 1), ("S", 0))}
            feasible = min(counts.values()) >= args.min_class
            summary.append(dict(drug=drug, heldout_lineage=heldout, feasible=feasible, **counts,
                                excluded=int((role == "excluded").sum())))
            if feasible:
                pd.DataFrame({"New_ID": ids, "Lineage": lin, "role": role, "fold": fold, "R": y}
                             ).to_csv(d_out / f"{drug}.csv", index=False)

        lab = ph[DRUGS].apply(lambda c: c.map({"R": 1, "S": 0})).fillna(-1).astype(int)
        lab = lab[(lab != -1).any(axis=1)]
        any_r = (lab == 1).any(axis=1).astype(int).values
        lin, role, fold = assign(lab.index, lineage, heldout, any_r, args.seed)
        md = pd.DataFrame({"New_ID": lab.index, "Lineage": lin, "role": role, "fold": fold})
        pd.concat([md, lab.reset_index(drop=True)], axis=1).to_csv(d_out / "MULTIDRUG.csv",
                                                                     index=False)

    sm = pd.DataFrame(summary)
    sm.to_csv(out / "summary.csv", index=False)
    print(sm.pivot(index="drug", columns="heldout_lineage", values="feasible").to_string())


if __name__ == "__main__":
    main()

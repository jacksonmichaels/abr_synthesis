"""Structure sweep v2: each arm vs. its references, macro CV AUC delta and
exact Wilcoxon over the paired folds (drugs where both sides have a JSON).
Writes compare.csv next to this file.

    python results/experiments/locusfusion_v2_structure_v2/compare.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RUNS = {
    "none (sd41_dna_protein)": EXP / "locusfusion_v2/sd41_dna_protein__locusfusion_v2",
    "L30 C-alpha, invalid 0": EXP / "locusfusion_v2_structure/sd_dna_protein_structure_L30__locusfusion_v2",
    "L30 C-alpha, invalid -1": HERE / "sd_dna_protein_structure_L30_neg1__locusfusion_v2",
    "L30 heavy, invalid -1": HERE / "sd_dna_protein_structure_L30_heavy_neg1__locusfusion_v2",
    "L30 heavy + gate, invalid -1": HERE / "sd_dna_protein_structure_L30_heavy_gate_neg1__locusfusion_v2",
}
PAIRS = [  # (arm, reference)
    ("L30 C-alpha, invalid 0", "none (sd41_dna_protein)"),
    ("L30 C-alpha, invalid -1", "none (sd41_dna_protein)"),
    ("L30 heavy, invalid -1", "none (sd41_dna_protein)"),
    ("L30 heavy + gate, invalid -1", "none (sd41_dna_protein)"),
    ("L30 C-alpha, invalid -1", "L30 C-alpha, invalid 0"),
    ("L30 heavy, invalid -1", "L30 C-alpha, invalid -1"),
    ("L30 heavy + gate, invalid -1", "L30 heavy, invalid -1"),
]
SKIP = {"KANAMYCIN"}            # rrs only: no structure, every arm is the control


def folds(run):
    out = {}
    for p in glob.glob(str(run / "*.json")):
        d = json.loads(Path(p).read_text())
        if "cv_folds" not in d or d["drug"] in SKIP:
            continue
        out[d["drug"]] = {"cv": [f["auc"] for f in d["cv_folds"]],
                          "sha": [f["val_ids_sha1"] for f in d["cv_folds"]],
                          "test": d["test"]["auc"]}
    return out


def main():
    F = {k: folds(v) for k, v in RUNS.items()}
    rows = []
    for arm, ref in PAIRS:
        drugs = sorted(set(F[arm]) & set(F[ref]))
        if not drugs:
            continue
        for d in drugs:
            assert F[arm][d]["sha"] == F[ref][d]["sha"], f"{arm} vs {ref}: {d} folds differ"
        a = np.concatenate([F[arm][d]["cv"] for d in drugs])
        b = np.concatenate([F[ref][d]["cv"] for d in drugs])
        per = {d: np.mean(F[arm][d]["cv"]) - np.mean(F[ref][d]["cv"]) for d in drugs}
        ta = [F[arm][d]["test"] for d in drugs]; tb = [F[ref][d]["test"] for d in drugs]
        rows.append({"arm": arm, "reference": ref, "n_drugs": len(drugs), "n_folds": len(a),
                     "macro_cv_delta": round(float(np.mean(list(per.values()))), 4),
                     "drugs_better": sum(v > 0 for v in per.values()),
                     "folds_better": int((a > b).sum()),
                     "wilcoxon_p": round(float(wilcoxon(a, b, method="exact").pvalue), 4)
                     if (a != b).any() else 1.0,
                     "macro_test_delta": round(float(np.mean(np.array(ta) - np.array(tb))), 4),
                     **{f"d_{d}": round(float(v), 4) for d, v in per.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "compare.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.iloc[:, :10].to_string(index=False))


if __name__ == "__main__":
    main()

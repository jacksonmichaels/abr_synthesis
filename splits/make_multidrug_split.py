"""
One lineage-grouped split for the multi-drug model.

    cd biophysical-fusion
    python -m locusfusion_v4.make_multidrug_split        # seconds, login-node safe

The per-drug lineage splits (scripts/build_lineage_splits.py) are built on each
drug's own phenotyped cohort, so they disagree about where an isolate sits and a
joint model cannot use them. This is the same recipe on the union cohort --
every isolate with an R/S call for at least one of the 11 drugs: sublineage
groups (deepened while a group is above 10% of the cohort), Beijing
(lineage2.2.1*) alone as CV fold 4, the other groups packed whole into a ~20%
TEST split and folds 0-3, balanced on isolates and on EACH drug's resistant
isolates (balancing on "resistant to anything" instead left the test split
with 6 levofloxacin and 22 ethionamide R against ~140 per CV fold). No group
crosses any split.

Writes results/experiments/locusfusion_v4/splits/MULTIDRUG.csv (New_ID, group,
role, fold, R = resistant to any drug) and MULTIDRUG_summary.csv (per split:
n, and R / S per drug).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
from build_lineage_splits import BEIJING, DRUGS, SNAP, groups_for  # noqa: E402
from datasets.lineage import load_lineage  # noqa: E402

OUT = PROJECT / "results" / "experiments" / "locusfusion_v4" / "splits"


def assign_bins_multi(groups, Rmat, test_frac, seed):
    """build_lineage_splits.assign_bins with one resistant count per drug: whole
    groups go, largest first, to the bin with the biggest combined shortfall in
    isolates (weight 1) and in resistant isolates, averaged over drugs (weight 1).
    Beijing is CV fold 4 on its own; -1 = test."""
    g = pd.Series(groups)
    beij = g.str.startswith(BEIJING).values
    out = np.full(len(g), -9)
    out[beij] = 4
    rest = ~beij
    n_all, r_all = len(g), Rmat.sum(0).astype(float)
    n_rest, r_rest = int(rest.sum()), Rmat[rest].sum(0).astype(float)
    tgt = {-1: (test_frac * n_all, test_frac * r_all)}
    for k in range(4):
        tgt[k] = ((n_rest - tgt[-1][0]) / 4, (r_rest - tgt[-1][1]) / 4)
    have = {b: [0.0, np.zeros(Rmat.shape[1])] for b in tgt}
    stats = pd.DataFrame(Rmat[rest]).assign(g=g[rest].values).groupby("g")
    size = stats.size()
    rsum = stats.sum()
    order = size.sample(frac=1, random_state=seed).sort_values(ascending=False, kind="stable").index
    where = {}
    for grp in order:
        n, r = size[grp], rsum.loc[grp].values

        def shortfall(b):
            tn, tr = tgt[b]
            return (tn - have[b][0]) / max(tn, 1) + np.mean((tr - have[b][1]) / np.maximum(tr, 1))
        b = max(tgt, key=shortfall)
        where[grp] = b
        have[b][0] += n
        have[b][1] = have[b][1] + r
    out[rest] = [where[x] for x in g[rest]]
    assert (out != -9).all()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phenotype-csv", default=str(SNAP / "phenotype_data" / "master_resistance_table.csv"))
    ap.add_argument("--max-share", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lin = load_lineage()
    ph = pd.read_csv(args.phenotype_csv, low_memory=False).set_index("New_ID")
    called = ph[DRUGS].isin(["R", "S"])
    ids = ph.index[called.any(axis=1)].intersection(lin.index)
    y = (ph.loc[ids, DRUGS] == "R").any(axis=1).astype(int).values
    g, deepened = groups_for(lin.loc[ids], args.max_share)
    Rmat = (ph.loc[ids, DRUGS] == "R").values.astype(int)
    bins = assign_bins_multi(g.values, Rmat, args.test_frac, args.seed)
    df = pd.DataFrame({"New_ID": ids, "group": g.values,
                       "role": np.where(bins == -1, "test", "train"),
                       "fold": np.where(bins >= 0, bins, -1), "R": y})
    assert not set(df.group[df.role == "test"]) & set(df.group[df.role == "train"])
    for k in range(5):
        assert not set(df.group[df.fold == k]) & set(df.group[(df.role == "train") & (df.fold != k)])
    df.to_csv(out / "MULTIDRUG.csv", index=False)

    rows = []
    parts = [("test", df.role == "test")] + [(f"cv{k}", df.fold == k) for k in range(5)]
    for name, m in parts:
        sub = ph.loc[df.New_ID[m]]
        row = {"split": name, "n": int(m.sum()), "groups": int(df.group[m].nunique())}
        for d in DRUGS:
            row[f"{d}_R"] = int((sub[d] == "R").sum())
            row[f"{d}_S"] = int((sub[d] == "S").sum())
        rows.append(row)
    sm = pd.DataFrame(rows)
    sm.to_csv(out / "MULTIDRUG_summary.csv", index=False)
    print(f"{len(df)} isolates; deepened: {deepened}")
    print(sm[["split", "n", "groups"] + [f"{d}_R" for d in DRUGS]].to_string(index=False))


if __name__ == "__main__":
    main()

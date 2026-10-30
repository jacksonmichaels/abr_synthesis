"""Lineage-grouped train/test + 5-fold CV splits, one file per drug.

    python scripts/build_lineage_splits.py

For each drug's phenotyped cohort (R/S), isolates are grouped by sublineage
(datasets.lineage, level2, e.g. lineage4.3). Any level-2 group holding more than
--max-share (10%) of the drug's cohort is split one level deeper, repeatedly
(level3, then the deepest call), so no single group can dominate a fold. Mixed / uncalled isolates form one group of their own.
Beijing (lineage2.2.1 and its sub-calls; 17-41% of a cohort, ~30-39% of its
resistant isolates, indivisible at barcode resolution) is CV fold 4 on its own -- a
built-in leave-Beijing-out. Every other group goes whole into a ~20% TEST split or CV
folds 0-3, balanced on isolate and resistant counts (assign_bins). No group appears
on both sides of any split.

Writes results/experiments/lineage_split/splits/{DRUG}.csv (New_ID, group, role,
fold) and splits/summary.csv (per drug x split: n, R, S, groups, largest group).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from datasets.lineage import load_lineage  # noqa: E402

SNAP = Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis/data_snapshot")
DRUGS = ["AMIKACIN", "CAPREOMYCIN", "ETHAMBUTOL", "ETHIONAMIDE", "ISONIAZID", "KANAMYCIN",
         "LEVOFLOXACIN", "MOXIFLOXACIN", "PYRAZINAMIDE", "RIFAMPICIN", "STREPTOMYCIN"]


def groups_for(lin, max_share):
    """level2 groups; any group above max_share of the cohort is split one level
    deeper, repeatedly (level2 -> level3 -> deepest call) until none is, or the
    group cannot be split further."""
    ok = lin.status == "ok"
    g = lin.level2.where(ok, "unassigned").copy()
    deepened = []
    for nxt in ("level3", "lineage"):
        share = g.value_counts(normalize=True)
        big = share[share > max_share].index.difference(["unassigned"])
        if big.empty:
            break
        deepened += list(big)
        m = g.isin(big)
        g[m] = lin.loc[m, nxt]
    return g, deepened


BEIJING = "lineage2.2.1"


def assign_bins(groups, y, test_frac, seed):
    """Whole groups -> bins. Beijing (lineage2.2.1 and its sub-calls) is CV fold 4 on
    its own. The other groups fill a test bin (-1, ~test_frac of all isolates) and CV
    folds 0-3 (equal shares of what is left), largest group first, each into the bin
    with the biggest combined shortfall in isolates and resistant isolates. A seeded
    shuffle breaks ties between equal-size groups."""
    g = pd.Series(groups)
    beij = g.str.startswith(BEIJING).values
    out = np.full(len(g), -9)
    out[beij] = 4
    rest = ~beij
    n_all, r_all = len(g), int(y.sum())
    n_rest, r_rest = int(rest.sum()), int(y[rest].sum())
    tgt = {-1: (test_frac * n_all, test_frac * r_all)}
    n_cv, r_cv = n_rest - tgt[-1][0], r_rest - tgt[-1][1]
    for k in range(4):
        tgt[k] = (n_cv / 4, r_cv / 4)
    have = {b: [0, 0] for b in tgt}
    stats = (pd.DataFrame({"g": g[rest].values, "y": y[rest]})
             .groupby("g").y.agg(["size", "sum"]))
    stats = stats.sample(frac=1, random_state=seed).sort_values("size", ascending=False, kind="stable")
    where = {}
    for grp, (n, r) in stats.iterrows():
        def shortfall(b):
            tn, tr = tgt[b]
            return (tn - have[b][0]) / max(tn, 1) + (tr - have[b][1]) / max(tr, 1)
        b = max(tgt, key=shortfall)
        where[grp] = b
        have[b][0] += n
        have[b][1] += r
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
    ap.add_argument("--out", default=str(PROJECT / "results" / "experiments" / "lineage_split" / "splits"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lin = load_lineage()
    ph = pd.read_csv(args.phenotype_csv, low_memory=False).set_index("New_ID")
    summary = []
    for d in DRUGS:
        ids = ph.index[ph[d].isin(["R", "S"])].intersection(lin.index)
        y = (ph.loc[ids, d] == "R").astype(int).values           # 1 = resistant
        g, deepened = groups_for(lin.loc[ids], args.max_share)
        gv = g.values
        bins = assign_bins(gv, y, args.test_frac, args.seed)     # -1 = test, 0..4 = CV fold
        te_idx = np.nonzero(bins == -1)[0]
        fold = np.where(bins >= 0, bins, -1)
        role = np.where(np.isin(np.arange(len(y)), te_idx), "test", "train")
        df = pd.DataFrame({"New_ID": ids, "group": gv, "role": role, "fold": fold, "R": y})
        df.to_csv(out / f"{d}.csv", index=False)
        # no group on both sides
        assert not set(df.group[df.role == "test"]) & set(df.group[df.role == "train"])
        for k in range(5):
            ks = set(df.group[df.fold == k])
            assert not ks & set(df.group[(df.role == "train") & (df.fold != k)])
        parts = [("test", df.role == "test")] + [(f"cv{k}", df.fold == k) for k in range(5)]
        for name, m in parts:
            s = df[m]
            summary.append(dict(drug=d, split=name, n=len(s), R=int(s.R.sum()), S=int((1 - s.R).sum()),
                                groups=s.group.nunique(),
                                largest_group=s.group.value_counts().index[0],
                                largest_share=round(s.group.value_counts(normalize=True).iloc[0], 3),
                                deepened=";".join(deepened)))
    sm = pd.DataFrame(summary)
    sm.to_csv(out / "summary.csv", index=False)
    print(sm.pivot(index="drug", columns="split", values="R").to_string())
    print()
    print(sm.pivot(index="drug", columns="split", values="n").to_string())
    print()
    print(sm.groupby("drug").deepened.first().to_string())


if __name__ == "__main__":
    main()

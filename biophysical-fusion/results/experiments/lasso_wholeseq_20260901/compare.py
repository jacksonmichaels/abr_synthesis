#!/usr/bin/env python3
"""
The comparison table for this run: the four lasso arms against their matched
controls.

Lives with this run and dies with its write-up.

## What is matched to what

The lasso arms are DNA-only single-drug models on the SD-CNN 5-fold protocol,
so the matched control is the DNA-only single-drug network cell over the SAME
locus universe — nothing else may differ:

    perdrug  ->  full_run_v2/dna__{late_fusion,mdcnn,cisfusion,setfusion}
    all      ->  alllocus_run_v2/dna__{late_fusion,mdcnn,cisfusion,setfusion}

`--net-pick best` (default) takes the best architecture per drug, which is the
generous reading for the networks: the lasso has to beat whichever of the four
happened to win that drug. `--net-pick mdcnn` pins one architecture instead, if
you would rather compare against a single fixed model.

The third control is `variant_aggregators_20260825/sparse_baseline`, the SAME
model family on a distilled input (<=512 variant tokens per isolate). That is
the comparison this run exists to make: raw alignment versus called variants,
lasso either way.

`SDCNN_CLEAN` (leak-corrected published SD-CNN) comes from `bigtb_baselines`,
so it is never transcribed twice.

Usage::

    python results/experiments/lasso_wholeseq_20260901/compare.py
    python results/experiments/lasso_wholeseq_20260901/compare.py --net-pick mdcnn
"""
import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd  # noqa: E402

# bigtb_baselines is the single source of the leak-corrected SD-CNN numbers and
# must never be transcribed a second time — so if it is not importable the
# column is DROPPED, not filled in from memory. (It is currently untracked in
# the working checkout, which is exactly the situation this guard is for.)
try:
    from bigtb_baselines import SDCNN_CLEAN  # noqa: E402
except ImportError:
    SDCNN_CLEAN = None

RUN = Path(__file__).resolve().parent
EXPERIMENTS = RUN.parent
ARMS = [("onehot", "perdrug"), ("onehot", "all"),
        ("delta", "perdrug"), ("delta", "all")]
NET_CONTROL = {"perdrug": "full_run_v2", "all": "alllocus_run_v2"}
NET_ARCHS = ("late_fusion", "mdcnn", "cisfusion", "setfusion")
SPARSE = EXPERIMENTS / "variant_aggregators_20260825" / "sparse_baseline"


def _summary(folder):
    """drug -> cv_auc_mean from a run folder's summary.csv (empty if absent)."""
    path = Path(folder) / "summary.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["drug"], df["cv_auc_mean"]))


def net_control(locus_set, pick):
    """drug -> CV AUC of the DNA-only network cell over the same locus set."""
    base = EXPERIMENTS / NET_CONTROL[locus_set]
    archs = NET_ARCHS if pick == "best" else (pick,)
    out = {}
    for arch in archs:
        for drug, auc in _summary(base / f"dna__{arch}").items():
            if pick == "best":
                out[drug] = max(out.get(drug, float("-inf")), auc)
            else:
                out[drug] = auc
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net-pick", default="best",
                    choices=("best",) + NET_ARCHS,
                    help="which DNA-only network cell is the control: the best "
                         "architecture per drug (default, generous to the "
                         "networks) or one fixed architecture")
    ap.add_argument("--csv", help="also write the table here")
    args = ap.parse_args()

    lasso = {f"{enc}_{loci}": _summary(RUN / f"{enc}_{loci}") for enc, loci in ARMS}
    sparse = _summary(SPARSE)
    nets = {ls: net_control(ls, args.net_pick) for ls in ("perdrug", "all")}

    drugs = sorted(SDCNN_CLEAN) if SDCNN_CLEAN else sorted(
        {d for arm in lasso.values() for d in arm})
    if SDCNN_CLEAN is None:
        print("note: bigtb_baselines not importable — the SD-CNN column is "
              "omitted rather than transcribed from anywhere else.\n")
    rows = []
    for d in drugs:
        row = {"drug": d, "L1 variant-token": sparse.get(d)}
        if SDCNN_CLEAN:
            row["SD-CNN (leak-corr)"] = SDCNN_CLEAN[d][1]
        for enc, loci in ARMS:
            row[f"lasso {enc}/{loci}"] = lasso[f"{enc}_{loci}"].get(d)
        row["net dna perdrug"] = nets["perdrug"].get(d)
        row["net dna all"] = nets["all"].get(d)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("drug")

    # macro over the drugs each column actually has, and — separately — over the
    # drugs EVERY column has, because a macro taken over different drug sets is
    # not a comparison. Both are printed; only the second one may be quoted.
    complete = df.dropna()
    df.loc["macro (own drugs)"] = df.mean(numeric_only=True)
    df.loc[f"macro (n={len(complete)} complete)"] = complete.mean(numeric_only=True)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(df.round(4).to_string())
    if len(complete) < len(drugs):
        missing = sorted(set(drugs) - set(complete.index))
        print(f"\nincomplete rows (a cell is missing somewhere): {missing}")
    if args.csv:
        df.round(4).to_csv(args.csv)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Read the three window arms against their parent whole-genome run.

Four questions, in the order they have to be asked:

  1. Does a model restricted to 200 bp around 50 sites reach the genome-wide
     model's accuracy? Read on TEST, which is clean; CV is selection-optimistic
     because the 50 sites were chosen by a fit that saw four fifths of the
     training split (see lasso_windows.py, "Selection provenance").
  2. Is the CV inflation visible? delta-top50's CV minus the parent's CV is a
     direct estimate of what selecting on the training data is worth.
  3. Does the surrounding context add anything over the 50 keys alone?
     delta-window - delta-top50, encoding held fixed.
  4. Does the wild-type reparametrisation add anything over the same columns?
     onehot-window - delta-window, columns held fixed.

And then where the mass lands: on the seed keys, on the neighbours the window
added, or on the wild-type indicators the encoding added.

    python results/experiments/lasso_windows_20260903/analyze.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

RUN = Path(__file__).resolve().parent
ARMS = ["delta-top50", "delta-window", "onehot-window"]


def load(run_dir):
    rows = []
    for p in sorted(Path(run_dir).glob("*.json")):
        d = json.loads(p.read_text())
        s = d["selection"]
        tf = d["top_features"]
        tot = sum(abs(f["coef"]) for f in tf) or 1.0
        seed = sum(abs(f["coef"]) for f in tf if f["selected"])
        wt = sum(abs(f["coef"]) for f in tf if f["feature"].endswith(":WT"))
        rows.append(dict(
            drug=d["drug"], arm=d["arm"], n=d["n_valid"],
            n_sel=s["n_selected"], wins=s["n_windows"], bp=s["window_bp"],
            cols=d["n_features"], sites=d["n_sites"],
            cv=d["cv_auc_mean"], test=d["test"]["auc"],
            support=d["cv_support_mean"], C=d["best_C"], secs=d["seconds"],
            parent_cv=s["parent_cv_auc"], parent_test=s["parent_test_auc"],
            parent_cols=s["parent_n_features"],
            seed_pct=100 * seed / tot, wt_pct=100 * wt / tot,
            nbr_pct=100 * (tot - seed - wt) / tot))
    return pd.DataFrame(rows)


def main():
    df = load(RUN / "all_drugs")
    if df.empty:
        sys.exit(f"no results under {RUN / 'all_drugs'}")
    pd.set_option("display.width", 220)
    order = df.groupby("drug")["n"].first().sort_values(ascending=False).index

    print("=" * 96)
    print("THE RUN")
    print("=" * 96)
    cost = (df[df.arm == "onehot-window"]
            .set_index("drug")
            .loc[order, ["n", "n_sel", "wins", "bp", "sites", "cols",
                         "parent_cols"]])
    cost["pct_poly"] = (100 * cost["sites"] / cost["bp"]).round(1)
    cost["shrink"] = (cost["parent_cols"] / cost["cols"]).round(0)
    print(cost.to_string())
    print("\n  bp        = merged 200 bp window footprint (windows overlap and merge)")
    print("  sites     = polymorphic positions in it, in this drug's cohort")
    print("  cols      = one-hot columns; shrink = parent columns / these columns")

    for col, label in (("test", "TEST AUC (clean)"),
                       ("cv", "CV AUC (selection-optimistic)")):
        print("\n" + "=" * 96)
        print(label)
        print("=" * 96)
        piv = df.pivot(index="drug", columns="arm", values=col)[ARMS]
        piv.insert(0, "parent", df.groupby("drug")[f"parent_{col}"].first())
        piv = piv.loc[order]
        piv["win-parent"] = piv["onehot-window"] - piv["parent"]
        piv["ctx"] = piv["delta-window"] - piv["delta-top50"]
        piv["enc"] = piv["onehot-window"] - piv["delta-window"]
        print(piv.round(4).to_string())
        print(f"  macro: " + "  ".join(
            f"{c} {piv[c].mean():.4f}"
            for c in ["parent"] + ARMS))
        print(f"  win-parent {piv['win-parent'].mean():+.4f} "
              f"({(piv['win-parent'] > 0).sum()}/{len(piv)} wins)   "
              f"ctx {piv['ctx'].mean():+.4f} "
              f"({(piv['ctx'] > 0).sum()}/{len(piv)})   "
              f"enc {piv['enc'].mean():+.4f} "
              f"({(piv['enc'] > 0).sum()}/{len(piv)})")

    print("\n" + "=" * 96)
    print("SELECTION OPTIMISM  (delta-top50 CV - parent CV, same 50 columns)")
    print("=" * 96)
    t50 = df[df.arm == "delta-top50"].set_index("drug").loc[order]
    opt = pd.DataFrame({"n": t50["n"], "parent_cv": t50["parent_cv"],
                        "top50_cv": t50["cv"],
                        "cv_gain": t50["cv"] - t50["parent_cv"],
                        "parent_test": t50["parent_test"],
                        "top50_test": t50["test"],
                        "test_gain": t50["test"] - t50["parent_test"]})
    print(opt.round(4).to_string())
    print(f"  macro CV gain {opt['cv_gain'].mean():+.4f}  "
          f"macro TEST gain {opt['test_gain'].mean():+.4f}")

    print("\n" + "=" * 96)
    print("COEFFICIENT MASS, % of sum|coef|")
    print("=" * 96)
    for arm in ("delta-window", "onehot-window"):
        sub = df[df.arm == arm].set_index("drug").loc[order]
        m = sub[["support", "seed_pct", "nbr_pct", "wt_pct"]]
        print(f"\n-- {arm}")
        print(m.round(1).to_string())
        print("   mean  " + "  ".join(
            f"{c} {sub[c].mean():.1f}%" for c in
            ["seed_pct", "nbr_pct", "wt_pct"]))
    print("\n  seed = the <=50 keys the parent selected;  nbr = other keys in "
          "the windows;  wt = wild-type indicators")

    df.to_csv(RUN / "analysis.csv", index=False)
    print(f"\nwrote {RUN / 'analysis.csv'}")


if __name__ == "__main__":
    main()

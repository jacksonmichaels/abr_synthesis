#!/usr/bin/env python
"""
locusfusion_fixed against its two controls, per cell and per drug.

    python results/experiments/locusfusion_fixed/compare.py              # per-drug loci
    python results/experiments/locusfusion_fixed/compare.py --arm sd19   # all 19 loci

Both arms run the same architecture at the same protocol (300 epochs, patience
30, --min-epochs 50, --save-weights best, seed 0); only the tokenizer differs
from the `newmodels_full` control, and only `--arch` differs from the conv-net
control.

  arm    this run                  old-tokenizer control        conv-net control
  ----   -----------------------   --------------------------   -------------------------
  sd     sd_*__locusfusion         newmodels_full/sd_*          full_run_v2/*__mdcnn
  sd19   sd19_*__locusfusion       newmodels_full/sd19_*        alllocus_run_v2/*__mdcnn

Deltas are reported **fold-paired**: every single-drug run in this project splits
with the same `StratifiedKFold(5, shuffle=True, random_state=42)` over the same
cohort, so fold i is the same held-out isolates for both models. Pairing removes
the fold-difficulty term, which is much the larger one, and the interval is a
95% t interval over the 5 paired differences. It is anti-conservative -- CV folds
share training data -- and it contains no seed variance, because there is one
seed. Treat it as a lower bound, the same way `notebooks/claims.ipynb` does.

Control results live in the checkout the earlier runs were launched from; pass
--controls to point somewhere else.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent.parent
MODSETS = ["dna", "dna_protein", "dna_biophysical", "dna_regulatory",
           "all_modalities"]
DRUGS = ["ISONIAZID", "RIFAMPICIN", "ETHAMBUTOL", "PYRAZINAMIDE", "STREPTOMYCIN",
         "KANAMYCIN", "AMIKACIN", "CAPREOMYCIN", "LEVOFLOXACIN", "MOXIFLOXACIN",
         "ETHIONAMIDE"]
# Two-sided 95% t quantiles, df = n-1. Only df=4 is ever used (5 folds), but the
# table keeps the function honest if a run ever reports fewer.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}

ARMS = {
    # arm: (cell prefix in THIS folder, cell prefix in newmodels_full, mdcnn run)
    "sd":   ("sd_",   "sd_",   "full_run_v2"),
    "sd19": ("sd19_", "sd19_", "alllocus_run_v2"),
}


def cell(folder):
    """drug -> {"cv": mean, "folds": [per-fold AUC]} for one results folder."""
    out = {}
    if not folder.is_dir():
        return out
    for f in sorted(folder.glob("*.json")):
        drug = f.name.split("__")[0]
        if drug not in DRUGS:
            continue
        try:
            j = json.load(open(f))
            folds = [x["auc"] for x in (j.get("cv_folds") or []) if "auc" in x]
            out[drug] = {"cv": j["cv_auc_mean"], "folds": folds}
        except (KeyError, json.JSONDecodeError):
            pass
    return out


def macro(d, key="cv"):
    vals = [v[key] for v in d.values() if isinstance(v.get(key), (int, float))]
    return statistics.mean(vals) if vals else None


def t_interval(xs):
    """Mean of `xs` and its 95% t interval, or (mean, None, None) if n < 2."""
    if not xs:
        return None, None, None
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, None, None
    h = T95.get(len(xs) - 1, 1.96) * statistics.stdev(xs) / math.sqrt(len(xs))
    return m, m - h, m + h


def fold_diffs(a, b, drug):
    """The 5 fold-paired differences for ONE drug, or [] if either lacks folds."""
    fa = a.get(drug, {}).get("folds") or []
    fb = b.get(drug, {}).get("folds") or []
    return [x - y for x, y in zip(fa, fb)] if len(fa) == len(fb) and fa else []


def paired(a, b, drugs):
    """Fold-paired mean delta over `drugs`, and its 95% t interval.

    Falls back to the difference of the two CV means (point estimate only) for
    any drug whose run did not keep per-fold AUCs.
    """
    per_drug, exact = [], True
    for d in drugs:
        fa, fb = a[d]["folds"], b[d]["folds"]
        if len(fa) == len(fb) and fa:
            per_drug.append(statistics.mean(x - y for x, y in zip(fa, fb)))
        else:
            per_drug.append(a[d]["cv"] - b[d]["cv"])
            exact = False
    if not per_drug:
        return None, None, None, True
    if len(per_drug) == 1:
        # One drug: the folds are the replicates, not the drug.
        m, lo, hi = t_interval(fold_diffs(a, b, drugs[0]))
        return (m, lo, hi, exact) if m is not None else (per_drug[0], None, None, exact)
    m, lo, hi = t_interval(per_drug)
    return m, lo, hi, exact


def fmt(v, width=7):
    return f"{v:{width}.4f}" if isinstance(v, (int, float)) else " " * (width - 1) + "-"


def ci(lo, hi):
    if lo is None or hi is None:
        return " " * 20
    return f"[{lo:+.4f}, {hi:+.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), default="sd",
                    help="which locus universe (default: sd, the drug's own loci)")
    ap.add_argument("--controls", default=str(PROJECT / "results" / "experiments"),
                    help="where newmodels_full/ and the conv-net control live")
    args = ap.parse_args()
    ctl = Path(args.controls)
    mine_pre, old_pre, mdcnn_run = ARMS[args.arm]

    print(f"arm: {args.arm}   conv-net control: {mdcnn_run}   controls at: {ctl}\n")
    print(f"{'cell':18} {'v2':>8} {'old lf':>8} {'delta':>8} {'95% CI':>20} "
          f"{'mdcnn':>8} {'v2-mdcnn':>9} {'95% CI':>20}  n")
    print("-" * 112)
    per_cell, sums, degraded = {}, {"v2": [], "old": [], "mdcnn": []}, False
    for ms in MODSETS:
        new = cell(HERE / f"{mine_pre}{ms}__locusfusion")
        old = cell(ctl / "newmodels_full" / f"{old_pre}{ms}__locusfusion")
        mdc = cell(ctl / mdcnn_run / f"{ms}__mdcnn")
        per_cell[ms] = (new, old, mdc)
        # Compare only on the drugs all three actually have, so a half-finished
        # run cannot flatter itself.
        shared = sorted(set(new) & set(old) & set(mdc))
        a = macro({d: new[d] for d in shared})
        b = macro({d: old[d] for d in shared})
        c = macro({d: mdc[d] for d in shared})
        dm, dlo, dhi, e1 = paired(new, old, shared)
        gm, glo, ghi, e2 = paired(new, mdc, shared)
        degraded = degraded or not (e1 and e2)
        print(f"{ms:18} {fmt(a, 8)} {fmt(b, 8)} {fmt(dm, 8)} {ci(dlo, dhi):>20} "
              f"{fmt(c, 8)} {fmt(gm, 9)} {ci(glo, ghi):>20}  {len(shared)}/11")
        for k, v in (("v2", a), ("old", b), ("mdcnn", c)):
            if v is not None:
                sums[k].append(v)

    print("-" * 112)
    if sums["v2"]:
        mv, mo, mm = (statistics.mean(sums[k]) for k in ("v2", "old", "mdcnn"))
        print(f"{'macro over cells':18} {fmt(mv, 8)} {fmt(mo, 8)} {fmt(mv - mo, 8)} "
              f"{'':>20} {fmt(mm, 8)} {fmt(mv - mm, 9)}")
    if degraded:
        print("\nNOTE: at least one cell had no per-fold AUCs, so its delta is the\n"
              "      difference of two CV means and carries no interval.")

    print(f"\nper drug, all_modalities   (delta = v2 - old tokenizer, fold-paired)\n"
          f"{'drug':14} {'v2':>8} {'old lf':>8} {'delta':>8} {'95% CI':>20} {'mdcnn':>8}")
    print("-" * 72)
    new, old, mdc = per_cell["all_modalities"]
    for d in DRUGS:
        a = new.get(d, {}).get("cv")
        b = old.get(d, {}).get("cv")
        c = mdc.get(d, {}).get("cv")
        lo = hi = dm = None
        if d in new and d in old:
            dm, lo, hi, _ = paired(new, old, [d])
        print(f"{d:14} {fmt(a, 8)} {fmt(b, 8)} {fmt(dm, 8)} {ci(lo, hi):>20} {fmt(c, 8)}")


if __name__ == "__main__":
    main()

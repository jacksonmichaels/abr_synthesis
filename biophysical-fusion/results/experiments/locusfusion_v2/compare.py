#!/usr/bin/env python
"""
locusfusion_v2 against its two controls, per cell and per drug.

    python results/experiments/locusfusion_v2/compare.py

Controls, both already run at this exact protocol (300 epochs, patience 30,
--min-epochs 50, --save-weights best, seed 0, same per-drug loci):

  newmodels_full/sd_*__locusfusion   the SAME architecture on the old tokenizer,
                                     which is the comparison this run exists for
  full_run_v2/*__mdcnn               the project's strongest single-drug model

Control results live in the checkout the earlier runs were launched from; pass
--controls to point somewhere else.
"""
import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent.parent
MODSETS = ["dna", "dna_protein", "dna_biophysical", "dna_regulatory",
           "all_modalities"]
DRUGS = ["ISONIAZID", "RIFAMPICIN", "ETHAMBUTOL", "PYRAZINAMIDE", "STREPTOMYCIN",
         "KANAMYCIN", "AMIKACIN", "CAPREOMYCIN", "LEVOFLOXACIN", "MOXIFLOXACIN",
         "ETHIONAMIDE"]


def cell(folder):
    """drug -> CV AUC for one results folder, or {} if it does not exist."""
    out = {}
    if not folder.is_dir():
        return out
    for f in sorted(folder.glob("*.json")):
        drug = f.name.split("__")[0]
        if drug not in DRUGS:
            continue
        try:
            out[drug] = json.load(open(f))["cv_auc_mean"]
        except (KeyError, json.JSONDecodeError):
            pass
    return out


def macro(d):
    vals = [v for v in d.values() if isinstance(v, (int, float))]
    return statistics.mean(vals) if vals else None


def fmt(v, width=7):
    return f"{v:{width}.4f}" if isinstance(v, (int, float)) else " " * (width - 1) + "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", default=str(PROJECT / "results" / "experiments"),
                    help="where newmodels_full/ and full_run_v2/ live")
    args = ap.parse_args()
    ctl = Path(args.controls)

    print(f"{'cell':18} {'v2':>8} {'old lf':>8} {'delta':>8} {'mdcnn':>8} "
          f"{'v2-mdcnn':>9}  n")
    print("-" * 68)
    totals = {"v2": [], "old": [], "mdcnn": []}
    per_cell = {}
    for ms in MODSETS:
        new = cell(HERE / f"sd_{ms}__locusfusion")
        old = cell(ctl / "newmodels_full" / f"sd_{ms}__locusfusion")
        mdc = cell(ctl / "full_run_v2" / f"{ms}__mdcnn")
        per_cell[ms] = (new, old, mdc)
        # compare on the drugs all three actually have, so a half-finished run
        # cannot flatter itself
        shared = sorted(set(new) & set(old) & set(mdc))
        a = macro({d: new[d] for d in shared})
        b = macro({d: old[d] for d in shared})
        c = macro({d: mdc[d] for d in shared})
        delta = a - b if a is not None and b is not None else None
        gap = a - c if a is not None and c is not None else None
        print(f"{ms:18} {fmt(a, 8)} {fmt(b, 8)} {fmt(delta, 8)} {fmt(c, 8)} "
              f"{fmt(gap, 9)}  {len(shared)}/11")
        for k, v in (("v2", a), ("old", b), ("mdcnn", c)):
            if v is not None:
                totals[k].append(v)

    print("-" * 68)
    if totals["v2"]:
        print(f"{'macro over cells':18} {fmt(macro(dict(enumerate(totals['v2']))), 8)} "
              f"{fmt(macro(dict(enumerate(totals['old']))), 8)} "
              f"{fmt(macro(dict(enumerate(totals['v2']))) - macro(dict(enumerate(totals['old']))), 8)} "
              f"{fmt(macro(dict(enumerate(totals['mdcnn']))), 8)}")

    print(f"\nper drug, all_modalities\n{'drug':14} {'v2':>8} {'old lf':>8} "
          f"{'delta':>8} {'mdcnn':>8}")
    print("-" * 52)
    new, old, mdc = per_cell["all_modalities"]
    for d in DRUGS:
        a, b, c = new.get(d), old.get(d), mdc.get(d)
        delta = a - b if a is not None and b is not None else None
        print(f"{d:14} {fmt(a, 8)} {fmt(b, 8)} {fmt(delta, 8)} {fmt(c, 8)}")


if __name__ == "__main__":
    main()

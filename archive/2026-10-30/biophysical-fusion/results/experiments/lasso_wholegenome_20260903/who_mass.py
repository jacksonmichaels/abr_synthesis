#!/usr/bin/env python3
"""
Where the whole-genome lasso puts its weight: inside or outside the WHO regions.

One measurement per drug. For every nonzero coefficient of the selected model,
the variant's snpEff annotation is matched against WHO 2023 Catalogue Table 21
(candidate resistance genes, tier 1 + tier 2) by gene name AND by Rv tag, and
the |coefficient| mass is split three ways:

    this drug   -- a Table-21 gene for the drug being predicted, or an
                   intergenic region flanking one (which is where Table-22
                   upstream regions live, e.g. Rv1482c-fabG1 = the fabG1-inhA
                   operon promoter)
    other drug  -- a Table-21 gene for some other drug
    outside     -- everything else in the 4.41 Mb

Usage::

    python results/experiments/lasso_wholegenome_20260903/who_mass.py \
        --run results/experiments/lasso_wholegenome_20260903/all_drugs
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR))

from datasets.who_catalogue import TABLE_21, genes_for_drug  # noqa: E402

# LFX and MFX are both 'FQ' in Table 21.
ABBR = {"AMIKACIN": "AMK", "CAPREOMYCIN": "CAP", "ETHAMBUTOL": "EMB",
        "ETHIONAMIDE": "ETO", "ISONIAZID": "INH", "KANAMYCIN": "KAN",
        "LEVOFLOXACIN": "FQ", "MOXIFLOXACIN": "FQ", "PYRAZINAMIDE": "PZA",
        "RIFAMPICIN": "RIF", "STREPTOMYCIN": "STM"}


def who_sets():
    """drug -> {names and Rv tags}, plus the union over all drugs."""
    per, union = {}, set()
    for ab in set(ABBR.values()):
        s = set()
        for g, rv in genes_for_drug(ab):
            s.add(g)
            if rv:
                s.add(rv)
        per[ab] = s
        union |= s
    for ab in TABLE_21:
        for g, rv in genes_for_drug(ab):
            union.add(g)
            if rv:
                union.add(rv)
    return per, union


def tokens(feature):
    """The gene identifiers a feature's annotation names.

    snpEff writes an intergenic region as 'geneA-geneB' / 'RvA-RvB', so a
    feature in the promoter of a WHO gene names that gene. Splitting on '-'
    is what makes those count as inside.
    """
    out = set()
    for field in (feature.get("gene", ""), feature.get("gene_id", "")):
        if not field:
            continue
        out.add(field)
        out.update(p for p in re.split(r"-", field) if p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--csv")
    args = ap.parse_args()

    per, union = who_sets()
    rows = []
    for jf in sorted(Path(args.run).glob("*.json")):
        r = json.loads(jf.read_text())
        drug = r["drug"]
        mine = per[ABBR[drug]]
        mass = defaultdict(float)
        nfeat = defaultdict(int)
        for f in r["top_features"]:
            t = tokens(f)
            k = ("this drug" if t & mine
                 else "other drug" if t & union
                 else "outside")
            mass[k] += abs(f["coef"])
            nfeat[k] += 1
        tot = sum(mass.values()) or 1.0
        rows.append({
            "drug": drug, "n": r["n_valid"], "cv_auc": r["cv_auc_mean"],
            "cols": r["n_features"], "support": len(r["top_features"]),
            "this drug %": 100 * mass["this drug"] / tot,
            "other drug %": 100 * mass["other drug"] / tot,
            "outside %": 100 * mass["outside"] / tot,
            "n_this": nfeat["this drug"], "n_other": nfeat["other drug"],
            "n_outside": nfeat["outside"],
        })

    if not rows:
        sys.exit(f"no result JSONs in {args.run}")
    w = "{:<14}{:>7}{:>9}{:>9}{:>9}{:>13}{:>13}{:>11}"
    print(w.format("drug", "n", "cv_auc", "cols", "support",
                   "this drug %", "other drug %", "outside %"))
    for r in rows:
        print(w.format(r["drug"], r["n"], f"{r['cv_auc']:.4f}", r["cols"],
                       r["support"], f"{r['this drug %']:.1f}",
                       f"{r['other drug %']:.1f}", f"{r['outside %']:.1f}"))
    n = len(rows)
    print(w.format(f"MEAN (n={n})", "", "", "", "",
                   f"{sum(r['this drug %'] for r in rows) / n:.1f}",
                   f"{sum(r['other drug %'] for r in rows) / n:.1f}",
                   f"{sum(r['outside %'] for r in rows) / n:.1f}"))
    if args.csv:
        import pandas as pd
        pd.DataFrame(rows).round(4).to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()

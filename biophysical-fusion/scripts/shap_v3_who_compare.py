#!/usr/bin/env python3
"""
Which of the 19 loci do the LocusFusion v3 models attribute to, against the
genes the WHO 2023 catalogue (Table 21, tier 1 + 2) lists for each drug?

Every (drug, locus) pair is put in one category, in this precedence:

    own_t1     WHO Table-21 tier-1 gene for the drug being predicted
    own_t2     WHO Table-21 tier-2 gene for the drug being predicted
    other_who  WHO tier 1/2 gene for some OTHER drug only (any Table-21 drug,
               as in lasso_wholegenome_20260903/who_mass.py)
    non_who    in no drug's tier 1/2 list

own_t1 + own_t2 is who_mass.py's "this drug", other_who its "other drug" and
non_who its "outside", so the LocusFusion, 19-locus LASSO and whole-genome LASSO
numbers line up column for column. Every one of the 19 loci is some drug's WHO
gene, so at locus granularity non_who is empty by construction; it is non-zero
only for the whole-genome LASSO.

Inputs
------
--shares   per-(drug, cell, locus) attribution shares. Column names are
           detected (drug/Drug, cell/config/run/model, locus/gene/block,
           share/locus_share/... or a magnitude such as mean_abs_attr, which is
           normalised). Extra grouping columns (background, fold, seed, arch) are
           kept as group keys. Block-level rows ("regulatory:inhA") are folded
           to their locus. Shares are renormalised to sum to 1 per group.
           A missing file is not an error: the catalogue and LASSO tables are
           still written.

Outputs (in --out)
------------------
locus_categories.csv      drug x 19 loci flags + category + mapping note
who_genes_per_drug.csv    per drug: WHO tier 1/2 genes, which of them are loaded
lasso_locus_shares.csv    19-locus LASSO |coef| mass per (arm, drug, locus)
lasso_category_shares.csv per (arm, drug): own_t1/own_t2/other_who/non_who
lasso_top5_loci.csv       5 heaviest loci per (arm, drug), categorised
shap_category_shares.csv  per (drug, cell, ...): category shares (+ token null)
shap_top5_loci.csv        5 most-attributed loci per (drug, cell, ...)
side_by_side.csv          SHAP cells and LASSO arms stacked, one row per drug

Usage (login-node safe: pandas over small CSV/JSON)::

    python scripts/shap_v3_who_compare.py
    python scripts/shap_v3_who_compare.py --shares path/to/loci.csv --out somewhere/
    # lineage split: the lineage-trained 19-locus one-hot LASSO only
    python scripts/shap_v3_who_compare.py \
        --shares results/analysis/shap_locusfusion_lineage_alltest/locus_shares.csv \
        --out results/analysis/shap_locusfusion_lineage_alltest/who \
        --lasso-onehot19 results/experiments/lineage_split/lasso_onehot_all \
        --lasso-vcf19 '' --lasso-genome ''
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from bigtb_ref import tb  # noqa: E402
from datasets.who_catalogue import TABLE_21, genes_for_drug  # noqa: E402

DRUG_TO_LOCI = tb.DRUG_TO_LOCI
# Same overlay as datasets.loader.EXTRA_LOCI (copied, not imported, so this
# script does not pull in the whole modality stack).
EXTRA_LOCI = {"ISONIAZID": ["fabG1"], "ETHIONAMIDE": ["fabG1"]}
# Same as datasets.regulatory.DRUG_NAME_TO_WHO. Both fluoroquinolones are 'FQ'.
DRUG_NAME_TO_WHO = {
    "ISONIAZID": "INH", "RIFAMPICIN": "RIF", "ETHAMBUTOL": "EMB",
    "PYRAZINAMIDE": "PZA", "STREPTOMYCIN": "STM", "KANAMYCIN": "KAN",
    "AMIKACIN": "AMK", "CAPREOMYCIN": "CAP", "LEVOFLOXACIN": "FQ",
    "MOXIFLOXACIN": "FQ", "ETHIONAMIDE": "ETO",
}
WHO_TO_DRUG = {"INH": "ISONIAZID", "RIF": "RIFAMPICIN", "EMB": "ETHAMBUTOL",
               "PZA": "PYRAZINAMIDE", "STM": "STREPTOMYCIN", "SM": "STREPTOMYCIN",
               "KAN": "KANAMYCIN", "AMK": "AMIKACIN", "AMI": "AMIKACIN",
               "CAP": "CAPREOMYCIN", "LFX": "LEVOFLOXACIN", "LEV": "LEVOFLOXACIN",
               "MFX": "MOXIFLOXACIN", "MOX": "MOXIFLOXACIN", "ETO": "ETHIONAMIDE",
               "ETH": "ETHIONAMIDE"}
DRUGS = list(DRUG_TO_LOCI)
LOCI19 = sorted({g for v in DRUG_TO_LOCI.values() for g in v} | {"fabG1"})
assert len(LOCI19) == 19, LOCI19
CATS = ["own_t1", "own_t2", "other_who", "non_who"]

DEFAULT_SHARES = PROJECT / "results/analysis/shap_locusfusion_v3/locus_shares.csv"
DEFAULT_OUT = PROJECT / "results/analysis/shap_locusfusion_v3/who"
LASSO_VCF19 = PROJECT / "results/experiments/lasso_vcf_loci_20260929/all19"
LASSO_ONEHOT19 = PROJECT / "results/experiments/lasso_wholeseq_20260901/onehot_all"
LASSO_GENOME = PROJECT / "results/experiments/lasso_wholegenome_20260903/all_drugs"

# What each locus's aligned FASTA actually spans (H37Rv, from the reference row
# of the alignment = lasso_vcf_loci all19 'locus_spans', checked against the
# GenBank CDS/rRNA features). Every locus is the named gene +/- ~100 bp, so a
# locus can carry a neighbour's fragment or a promoter. Assigned to the named
# gene throughout.
LOCUS_NOTE = {
    "fabG1": "fabG1 CDS + 100 bp upstream = fabG1-inhA operon promoter (c-15t etc.). "
             "Not in DRUG_TO_LOCI; EXTRA_LOCI adds it for INH/ETO.",
    "inhA": "inhA CDS (+95 bp of hemZ, non-WHO). Its regulatory block is WHO Table-22 'inhA', "
            "which IS the fabG1-inhA operon promoter (tss = fabG1 start), so promoter "
            "attribution is split between fabG1 (dna) and inhA (regulatory); both INH/ETO tier 1.",
    "katG": "katG CDS +/-100 bp; upstream flank holds 63 bp of furA (INH tier 1, not loaded "
            "separately), downstream 94 bp of Rv1907c (non-WHO). regulatory:katG (642 bp) "
            "covers furA.",
    "eis": "eis CDS + 100 bp upstream = eis promoter (c-14t, g-10a); regulatory:eis is the "
           "same promoter. AMK and KAN tier 1.",
    "embC": "embC CDS + 101 bp of aftA 3' end (EMB tier 1). regulatory:embC is WHO's 1982 bp "
            "upstream window, largely inside aftA. Same drug, no category change.",
    "embA": "embA CDS; regulatory:embA = embC-embA intergenic promoter. embC-embA-embB is one "
            "operon but is loaded as three loci, each EMB tier 1 in its own right.",
    "embB": "embB CDS (+33 bp of Rv3796, non-WHO).",
    "ethA": "ethA CDS + 65 bp of rraA (non-WHO). The ethA-ethR intergenic promoter (75 bp) is in "
            "neither dna locus; only regulatory:ethA/ethR carry it.",
    "ethR": "ethR CDS + 100 bp downstream. ETO tier 2.",
    "gid": "gid CDS + 100 bp upstream + 104 bp of parA (non-WHO).",
    "gyrA": "gyrA CDS + 100 bp downstream (5 bp Rv0007). FQ tier 1.",
    "gyrB": "gyrB CDS + 100 bp upstream. FQ tier 1.",
    "pncA": "pncA CDS + promoter (100 bp upstream) + 101 bp Rv2042c, 60 bp Rv2044c (non-WHO).",
    "rpoB": "rpoB CDS + 100 bp upstream. RIF tier 1.",
    "rpoC": "rpoC CDS + 100 bp downstream. RIF tier 2.",
    "rpsL": "rpsL CDS + 100 bp upstream + 101 bp of rpsG (non-WHO, same operon).",
    "rrl": "rrl + 100 bp downstream. CAP tier 2; also LZD tier 1 (not in our panel).",
    "rrs": "rrs + 100 bp upstream. AMK/KAN/CAP/STM tier 1.",
    "tlyA": "tlyA CDS + 93 bp Rv1693 + 101 bp ppnK (both non-WHO).",
}


# ----------------------------------------------------------------- catalogue
def who_tiers():
    """abbrev -> {gene: tier}; tier-1 wins if a gene is listed twice."""
    out = {}
    for ab in TABLE_21:
        d = {}
        for g, _ in genes_for_drug(ab, 2):
            d[g] = 2
        for g, _ in genes_for_drug(ab, 1):
            d[g] = 1
        out[ab] = d
    return out


def locus_categories():
    tiers = who_tiers()
    rows = []
    for drug in DRUGS:
        ab = DRUG_NAME_TO_WHO[drug]
        mine = tiers[ab]
        for loc in LOCI19:
            t = mine.get(loc)
            other_any = sorted(a for a, d in tiers.items() if a != ab and loc in d)
            other_panel = sorted(dr for dr in DRUGS
                                 if DRUG_NAME_TO_WHO[dr] != ab
                                 and loc in tiers[DRUG_NAME_TO_WHO[dr]])
            cat = ("own_t1" if t == 1 else "own_t2" if t == 2
                   else "other_who" if other_any else "non_who")
            rows.append({
                "drug": drug, "who_abbrev": ab, "locus": loc,
                "in_drug_map": loc in DRUG_TO_LOCI[drug],
                "in_extra_loci": loc in EXTRA_LOCI.get(drug, []),
                "who_t1_own": t == 1, "who_t2_own": t == 2,
                "who_other_drug_only": t is None and bool(other_any),
                "non_who_any_drug": t is None and not other_any,
                "category": cat,
                "other_who_drugs_table21": "+".join(
                    f"{a}(t{tiers[a][loc]})" for a in other_any),
                "other_who_drugs_panel": "+".join(other_panel),
                "mapping_note": LOCUS_NOTE[loc],
            })
    return pd.DataFrame(rows)


def who_genes_per_drug():
    rows = []
    for drug in DRUGS:
        ab = DRUG_NAME_TO_WHO[drug]
        t1 = [g for g, _ in genes_for_drug(ab, 1)]
        t2 = [g for g, _ in genes_for_drug(ab, 2)]
        rows.append({
            "drug": drug, "who_abbrev": ab,
            "drug_to_loci": "+".join(DRUG_TO_LOCI[drug]),
            "who_t1": "+".join(t1), "who_t2": "+".join(t2),
            "who_t1_in_19": "+".join(g for g in t1 if g in LOCI19),
            "who_t2_in_19": "+".join(g for g in t2 if g in LOCI19),
            "who_t1_not_loaded": "+".join(g for g in t1 if g not in LOCI19),
            "who_t2_not_loaded": "+".join(g for g in t2 if g not in LOCI19),
            "who_in_19_not_in_drug_map": "+".join(
                g for g in t1 + t2 if g in LOCI19 and g not in DRUG_TO_LOCI[drug]),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- share loading
def _pick(cols, names):
    low = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


_LOCUS_CANON = {l.lower(): l for l in LOCI19}


def canon_locus(x):
    """'regulatory:inhA' / 'dna_katG' / 'KATG' / 'fabG1-inhA' -> canonical or None."""
    s = str(x).strip()
    if s.lower() in _LOCUS_CANON:
        return _LOCUS_CANON[s.lower()]
    for part in reversed(re.split(r"[:/|]", s)):
        if part.lower() in _LOCUS_CANON:
            return _LOCUS_CANON[part.lower()]
    m = re.match(r"^(dna|protein|biophysical|regulatory|structure)[_.](.+)$", s, re.I)
    if m and m.group(2).lower() in _LOCUS_CANON:
        return _LOCUS_CANON[m.group(2).lower()]
    return None


def canon_drug(x):
    s = str(x).strip().upper()
    return WHO_TO_DRUG.get(s, s)


def load_shares(path):
    """Long frame: group keys + locus + share (sums to 1 per group)."""
    raw = pd.read_csv(path)
    cols = list(raw.columns)
    c_drug = _pick(cols, ["drug", "drug_name", "target_drug", "antibiotic"])
    c_loc = _pick(cols, ["locus", "gene", "locus_name", "loci", "block", "feature"])
    c_share = _pick(cols, ["share", "locus_share", "attr_share", "attribution_share",
                           "share_mean", "mean_share", "frac", "fraction"])
    c_mag = _pick(cols, ["mean_abs_attr", "abs_attr", "mean_abs_shap", "sum_abs_attr",
                         "attr", "attribution", "value"])
    if c_drug is None or c_loc is None or (c_share is None and c_mag is None):
        sys.exit(f"cannot find drug/locus/share columns in {path}: {cols}")
    c_cell = _pick(cols, ["cell", "config", "cell_name", "run", "model", "arm", "setting"])
    extras = [c for c in ("background", "bg", "fold", "seed", "arch", "run", "split")
              if c in cols and c not in (c_cell,)]
    val = c_share or c_mag
    df = pd.DataFrame({"drug": raw[c_drug].map(canon_drug),
                       "cell": raw[c_cell].astype(str) if c_cell else "all",
                       "locus_raw": raw[c_loc].astype(str),
                       "value": pd.to_numeric(raw[val], errors="coerce").abs()})
    for e in extras:
        df[e] = raw[e]
    df["locus"] = df["locus_raw"].map(canon_locus)
    bad = df[df["locus"].isna()]["locus_raw"].unique()
    if len(bad):
        print(f"[warn] {len(bad)} locus labels not among the 19 loci, dropped: "
              f"{list(bad)[:10]}")
    unk = sorted(set(df["drug"]) - set(DRUGS))
    if unk:
        print(f"[warn] drugs not in the panel, dropped: {unk}")
    df = df.dropna(subset=["locus", "value"])
    df = df[df["drug"].isin(DRUGS)]
    keys = ["drug", "cell"] + extras
    # fold block rows (dna:katG, regulatory:katG ...) into their locus
    df = df.groupby(keys + ["locus"], dropna=False, as_index=False)["value"].sum()
    tot = df.groupby(keys, dropna=False)["value"].transform("sum")
    df["share"] = np.where(tot > 0, df["value"] / tot, np.nan)
    df["n_loci"] = df.groupby(keys, dropna=False)["locus"].transform("nunique")
    print(f"loaded {path}: drug={c_drug} locus={c_loc} share={val}"
          f"{' (normalised magnitude)' if c_share is None else ''} cell={c_cell} "
          f"extra keys={extras}; {len(df)} rows, "
          f"{df.groupby(keys, dropna=False).ngroups} groups")
    tok = _pick(cols, ["mean_variant_tokens", "variant_tokens", "n_tokens"])
    if tok is not None and c_loc is not None:
        t = pd.DataFrame({"drug": raw[c_drug].map(canon_drug),
                          "cell": raw[c_cell].astype(str) if c_cell else "all",
                          "locus": raw[c_loc].map(canon_locus),
                          "tokens": pd.to_numeric(raw[tok], errors="coerce")})
        for e in extras:
            t[e] = raw[e]
        t = t.dropna(subset=["locus"]).groupby(keys + ["locus"], dropna=False,
                                               as_index=False)["tokens"].first()
        df = df.merge(t, on=keys + ["locus"], how="left")
    return df, keys


# ------------------------------------------------------------- summarising
def category_shares(df, keys, cat):
    d = df.merge(cat[["drug", "locus", "category", "in_drug_map"]],
                 on=["drug", "locus"], how="left")
    piv = (d.pivot_table(index=keys, columns="category", values="share",
                         aggfunc="sum", fill_value=0.0)
           .reindex(columns=CATS, fill_value=0.0))
    out = piv.copy()
    out["own_who"] = out["own_t1"] + out["own_t2"]
    out["in_drug_map"] = d[d["in_drug_map"]].groupby(keys, dropna=False)["share"].sum() \
        .reindex(out.index).fillna(0.0)
    out["n_loci"] = d.groupby(keys, dropna=False)["locus"].nunique().reindex(out.index)
    if "tokens" in d and d["tokens"].notna().any():
        tt = d.groupby(keys, dropna=False)["tokens"].transform("sum")
        d["tok_share"] = np.where(tt > 0, d["tokens"] / tt, np.nan)
        null = d[d["category"].isin(["own_t1", "own_t2"])].groupby(
            keys, dropna=False)["tok_share"].sum().reindex(out.index).fillna(0.0)
        out["own_who_null_tokens"] = null
        out["own_who_enrichment"] = np.where(null > 0, out["own_who"] / null, np.nan)
    top = (d.sort_values("share", ascending=False).groupby(keys, dropna=False).head(1)
           .set_index(keys))
    out["top_locus"] = top["locus"].reindex(out.index)
    out["top_locus_category"] = top["category"].reindex(out.index)
    return out.reset_index()


def top_k(df, keys, cat, k=5):
    d = df.merge(cat[["drug", "locus", "category", "in_drug_map", "other_who_drugs_panel"]],
                 on=["drug", "locus"], how="left")
    d = d.sort_values(keys + ["share"], ascending=[True] * len(keys) + [False])
    d["rank"] = d.groupby(keys, dropna=False).cumcount() + 1
    cols = keys + ["rank", "locus", "share", "category", "in_drug_map",
                   "other_who_drugs_panel"]
    return d[d["rank"] <= k][cols].reset_index(drop=True)


# ------------------------------------------------------------------ LASSO
def lasso_vcf19(root=LASSO_VCF19):
    """19-locus VCF LASSO (full support): |coef| per locus by genome position."""
    rows = []
    for jf in sorted(Path(root).glob("*.json")):
        r = json.loads(jf.read_text())
        spans = r["locus_spans"]
        for f in r["top_features"]:
            pos = int(f["feature"].split(":", 1)[0]) - 1
            hit = [l for l, (s, e) in spans.items() if s <= pos < e]
            rows.append({"drug": r["drug"], "locus": hit[0] if hit else None,
                         "value": abs(f["coef"])})
    df = pd.DataFrame(rows)
    if len(df) and df["locus"].isna().any():
        print(f"[warn] vcf_all19: {df['locus'].isna().sum()} features outside every span "
              f"(deletions reaching in) dropped")
    return df.dropna(subset=["locus"]).assign(cell="lasso_19locus_vcf_full")


def lasso_onehot19(root=LASSO_ONEHOT19):
    """19-locus one-hot LASSO, 40 largest |coef| only (what Fig. A1 left uses)."""
    rows = []
    for jf in sorted(Path(root).glob("*.json")):
        r = json.loads(jf.read_text())
        for f in r["top_features"]:
            rows.append({"drug": r["drug"], "locus": canon_locus(f["feature"].split(":")[0]),
                         "value": abs(f["coef"])})
    return pd.DataFrame(rows).dropna(subset=["locus"]).assign(cell="lasso_19locus_onehot_top40")


def _tokens(feature):
    out = set()
    for field in (feature.get("gene", ""), feature.get("gene_id", "")):
        if field:
            out.add(field)
            out.update(p for p in re.split(r"[-&]", field) if p)
    return out


def lasso_wholegenome(root=LASSO_GENOME):
    """who_mass.py's split, with 'this drug' broken into tier 1 / tier 2."""
    tiers = {}
    for ab in TABLE_21:
        d = {}
        for g, rv in genes_for_drug(ab, 2):
            d[g] = d[rv] = 2
        for g, rv in genes_for_drug(ab, 1):
            d[g] = d[rv] = 1
        tiers[ab] = d
    union = set().union(*[set(d) for d in tiers.values()])
    rows = []
    for jf in sorted(Path(root).glob("*.json")):
        r = json.loads(jf.read_text())
        mine = tiers[DRUG_NAME_TO_WHO[r["drug"]]]
        mass = dict.fromkeys(CATS, 0.0)
        for f in r["top_features"]:
            t = _tokens(f)
            own = [mine[x] for x in t if x in mine]
            k = ("own_t1" if 1 in own else "own_t2" if own
                 else "other_who" if t & union else "non_who")
            mass[k] += abs(f["coef"])
        tot = sum(mass.values()) or 1.0
        row = {"drug": r["drug"], "cell": "lasso_wholegenome_full"}
        row.update({k: v / tot for k, v in mass.items()})
        row["own_who"] = row["own_t1"] + row["own_t2"]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shares", default=str(DEFAULT_SHARES))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--lasso-onehot19", default=str(LASSO_ONEHOT19),
                    help="19-locus one-hot LASSO run dir (JSONs with top_features); "
                         "lineage: results/experiments/lineage_split/lasso_onehot_all")
    ap.add_argument("--lasso-vcf19", default=str(LASSO_VCF19),
                    help="19-locus VCF LASSO run dir; '' to skip")
    ap.add_argument("--lasso-genome", default=str(LASSO_GENOME),
                    help="whole-genome LASSO run dir; '' to skip")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    cat = locus_categories()
    cat.to_csv(out / "locus_categories.csv", index=False)
    who_genes_per_drug().to_csv(out / "who_genes_per_drug.csv", index=False)
    print(f"wrote {out / 'locus_categories.csv'} ({len(cat)} rows)")

    # LASSO, 19 loci (locus level, same categories as the SHAP shares)
    print(f"LASSO dirs: onehot19={args.lasso_onehot19} vcf19={args.lasso_vcf19 or '-'} "
          f"genome={args.lasso_genome or '-'}")
    las = pd.concat(([lasso_vcf19(args.lasso_vcf19)] if args.lasso_vcf19 else [])
                    + [lasso_onehot19(args.lasso_onehot19)], ignore_index=True)
    lkeys = ["drug", "cell"]
    las = las.groupby(lkeys + ["locus"], as_index=False)["value"].sum()
    las["share"] = las["value"] / las.groupby(lkeys)["value"].transform("sum")
    las.to_csv(out / "lasso_locus_shares.csv", index=False)
    lcat = category_shares(las, lkeys, cat)
    lwg = lasso_wholegenome(args.lasso_genome) if args.lasso_genome else None
    lcat_all = pd.concat([lcat] + ([lwg] if lwg is not None else []), ignore_index=True)
    lcat_all.to_csv(out / "lasso_category_shares.csv", index=False)
    top_k(las, lkeys, cat, args.top).to_csv(out / "lasso_top5_loci.csv", index=False)
    print("\nLASSO category shares (macro over drugs):")
    print(lcat_all.groupby("cell")[CATS + ["own_who"]].mean().round(3).to_string())

    side = [lcat_all.assign(source="lasso")]
    shares = Path(args.shares)
    if not shares.exists():
        print(f"\n[info] {shares} not found -- catalogue and LASSO tables written; "
              "rerun once the SHAP shares exist.")
    else:
        df, keys = load_shares(shares)
        scat = category_shares(df, keys, cat)
        scat.to_csv(out / "shap_category_shares.csv", index=False)
        top_k(df, keys, cat, args.top).to_csv(out / "shap_top5_loci.csv", index=False)
        side.append(scat.assign(source="locusfusion_shap"))
        gk = [k for k in keys if k != "drug"]
        print("\nSHAP category shares (macro over drugs):")
        print(scat.groupby(gk, dropna=False)[CATS + ["own_who", "in_drug_map"]]
              .mean().round(3).to_string())
        print(f"\ntop locus is an own-WHO gene: "
              + ", ".join(f"{k}: {int(g['top_locus_category'].isin(['own_t1', 'own_t2']).sum())}"
                          f"/{len(g)}" for k, g in scat.groupby(gk, dropna=False)))
    sbs = pd.concat(side, ignore_index=True)
    lead = ["source", "drug", "cell"]
    sbs = sbs[lead + [c for c in sbs.columns if c not in lead]]
    sbs.sort_values(["drug", "source", "cell"]).to_csv(out / "side_by_side.csv", index=False)
    print(f"\nwrote outputs to {out}")


if __name__ == "__main__":
    main()

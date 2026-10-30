"""Theoretical ceiling on AUC from the DNA input: the best any model of the
aligned locus sequences can do on our data.

Every model we train sees a deterministic function of the isolate's aligned
DNA at its loci (protein and biophysical channels are computed from it). So
isolates with byte-identical sequences at those loci must receive the same
score, and the best possible ranking orders genotype groups by their resistant
fraction. That ranking's AUC on an evaluation set is the maximum any model can
reach there (the "oracle" AUC). Isolates that share a genotype but disagree on
phenotype are where it falls below 1.

Settings, per drug:
  task       single = the per-drug lineage splits (lineage_split/splits/{DRUG}.csv)
             multi  = the one multi-drug lineage split (locusfusion_v4/splits/MULTIDRUG.csv)
  locus_set  perdrug = BIG-TB's DRUG_TO_LOCI for the drug, 19 = all 19 curated loci
  input      dna          = the gene-locus FASTAs only (what a dna-only model sees)
             dna+promoter = plus the promoter-window FASTAs the regulatory
                            modality loads for that model (separate DNA, e.g. the
                            fabG1-inhA promoter carrying c-15t is in no gene record)
  eval set   test, cv0..cv4 (each CV fold as its own validation set), all

Per (task, locus_set, drug, eval set) it writes:
  oracle_auc        in-sample max AUC on the eval set (groups formed within it)
  frac_singleton    isolates whose genotype is unique within the eval set; for
                    those the oracle is trivially right, so a high value means
                    the data cannot certify a ceiling below ~1
  dup_*             the same, restricted to isolates in groups of >= 2 within
                    the eval set -- where the ceiling is actually measured
  bayes_err         sum over groups of min(n_R, n_S) / n (errors of the best
                    group-constant classifier)
  noise_auc         ceiling extrapolated to the whole eval set under a label-
                    noise model: class-conditional flip rates estimated from
                    the drug's duplicate groups over the whole cohort
                    (majority label = truth, tied groups skipped)

Also writes a per-isolate table (group id, group size and R fraction within its
eval set) so model predictions can be scored on the duplicate subset.

DNA comes from data_snapshot and the TSS-anchored promoter windows in
abr_workspace/regulatory_msa/aligned -- the files every lineage-split and v4/v5
model trained on -- read as raw strings; an uncalled N is a different genotype,
as it is to the models.

    python scripts/bayes_ceiling.py              # ~1 min, <1 GB
    python scripts/bayes_ceiling.py --drugs PYRAZINAMIDE --tasks single --locus-sets perdrug
"""
import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parent.parent
WS = PROJECT.parent
GENO_DIR = WS / "data_snapshot" / "genomic_data" / "aligned"
PHENO_CSV = WS / "data_snapshot" / "phenotype_data" / "master_resistance_table.csv"
SPLIT_DIR = PROJECT / "results" / "experiments" / "lineage_split" / "splits"
MD_SPLIT = PROJECT / "results" / "experiments" / "locusfusion_v4" / "splits" / "MULTIDRUG.csv"
REG_DIR = Path("/home/jacksonmicha_umass_edu/abr_workspace/regulatory_msa/aligned")
OUT = PROJECT / "results" / "analysis" / "bayes_ceiling"

LOCI_19 = ["eis", "embA", "embB", "embC", "ethA", "ethR", "fabG1", "gid", "gyrA",
           "gyrB", "inhA", "katG", "pncA", "rpoB", "rpoC", "rpsL", "rrl", "rrs", "tlyA"]
# tb_cnn_codebase.DRUG_TO_LOCI, as recorded in the `genes` field of every
# lineage_split/sd_* result JSON (copied so this script needs no TF import)
DRUG_TO_LOCI = {
    "AMIKACIN": ["rrs", "eis"],
    "CAPREOMYCIN": ["rrs", "rrl", "tlyA"],
    "ETHAMBUTOL": ["embC", "embA", "embB"],
    "ETHIONAMIDE": ["inhA", "ethA", "ethR"],
    "ISONIAZID": ["inhA", "katG"],
    "KANAMYCIN": ["rrs"],
    "LEVOFLOXACIN": ["gyrB", "gyrA"],
    "MOXIFLOXACIN": ["gyrB", "gyrA"],
    "PYRAZINAMIDE": ["pncA"],
    "RIFAMPICIN": ["rpoB", "rpoC"],
    "STREPTOMYCIN": ["rpsL", "rrs", "gid"],
}
DRUGS = list(DRUG_TO_LOCI)
# promoter windows each model actually loaded: the `regulatory:` blocks of the
# lineage_split/{sd,sd19}_all_modalities_unk__locusfusion_v2 JSONs (the v4/v5
# sd19 JSONs' tokens.regulatory agree), and tokens.regulatory of the md_* JSONs
PROMOTERS = {
    "perdrug": {**{d: list(DRUG_TO_LOCI[d]) for d in DRUG_TO_LOCI},
                "CAPREOMYCIN": ["rrs", "tlyA"], "ETHAMBUTOL": ["embC", "embA"]},
}
PROMOTERS["19"] = {**PROMOTERS["perdrug"], "KANAMYCIN": ["rrs", "eis"]}
PROMOTERS_MD = ["inhA", "katG", "rpoB", "rpoC", "embC", "embA", "pncA", "rpsL", "rrs",
                "gid", "eis", "tlyA", "gyrB", "gyrA", "ethA", "ethR"]
INPUTS = ["dna", "dna+promoter"]


def key_columns(task, locus_set, drug, inp):
    loci = LOCI_19 if locus_set == "19" else DRUG_TO_LOCI[drug]
    cols = [f"dna:{l}" for l in loci]
    if inp == "dna+promoter":
        prom = (PROMOTERS_MD if task == "multi" and locus_set == "19"
                else PROMOTERS[locus_set][drug])
        cols += [f"prom:{r}" for r in prom]
    return cols


def read_fasta_ids(path):
    """{isolate id: aligned sequence} for one locus FASTA."""
    opener = gzip.open if str(path).endswith(".gz") else open
    seqs, name, buf = {}, None, []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def allele_table(columns):
    """isolates x columns table of integer allele ids (identical string = same id);
    a column is 'dna:<locus>' or 'prom:<region>'."""
    cols = {}
    for col in columns:
        kind, locus = col.split(":")
        t0 = time.time()
        seqs = read_fasta_ids((GENO_DIR if kind == "dna" else REG_DIR) / f"{locus}.fasta")
        seqs.pop("MT_H37Rv", None)
        codes, uniq = pd.factorize(pd.Series(seqs))
        cols[col] = pd.Series(codes, index=list(seqs))
        print(f"  {col}: {len(seqs)} isolates, {len(uniq)} distinct sequences "
              f"({time.time() - t0:.1f}s)", flush=True)
    tab = pd.DataFrame(cols)
    if tab.isna().any().any():
        raise ValueError(f"isolates missing from some locus FASTA: {tab.isna().any(axis=1).sum()}")
    return tab.astype(np.int64)


def labels():
    """New_ID x drug, 1 = R, 0 = S, NaN = untested."""
    ph = pd.read_csv(PHENO_CSV, index_col="New_ID", dtype=str, usecols=["New_ID"] + DRUGS)
    ph = ph[~ph.index.duplicated()]
    return ph.apply(lambda c: c.map({"R": 1.0, "S": 0.0}))


def group_key(alleles, loci):
    """One integer genotype id per isolate over the given loci."""
    return pd.Series(pd.factorize(pd.MultiIndex.from_frame(alleles[loci]))[0]
                     if len(loci) > 1 else alleles[loci[0]].to_numpy(), index=alleles.index)


def flip_rates(g, y):
    """P(obs S | true R), P(obs R | true S) from groups of >= 2 (majority = truth)."""
    df = pd.DataFrame({"g": g, "y": y}).groupby("g")["y"].agg(["size", "sum"])
    df = df[df["size"] >= 2]
    nR, nS = df["sum"], df["size"] - df["sum"]
    rmaj, smaj = df[nR > nS], df[nS > nR]
    f_r = (rmaj["size"] - rmaj["sum"]).sum() / max(rmaj["size"].sum(), 1)
    f_s = smaj["sum"].sum() / max(smaj["size"].sum(), 1)
    return float(f_r), float(f_s), int(len(rmaj)), int(len(smaj))


def noise_auc(n_pos, n_neg, f_r, f_s):
    """AUC of a perfect predictor of the true class against noisy labels.

    Solves observed counts for true counts, then pairs: (true R, true S) = 1,
    same true class = 0.5, (true S, true R) = 0."""
    A = np.array([[1 - f_r, f_s], [f_r, 1 - f_s]])
    try:
        tr, ts = np.linalg.solve(A, [n_pos, n_neg])
    except np.linalg.LinAlgError:
        return np.nan
    tr, ts = max(tr, 0.0), max(ts, 0.0)
    a, b = tr * (1 - f_r), ts * f_s         # observed R: truly R / truly S
    c, d = tr * f_r, ts * (1 - f_s)         # observed S: truly R / truly S
    den = (a + b) * (c + d)
    return float((a * d + 0.5 * (a * c + b * d)) / den) if den > 0 else np.nan


def oracle(g, y):
    """Group-fraction score, its AUC, and group stats within one eval set."""
    df = pd.DataFrame({"g": g.to_numpy(), "y": y.to_numpy()}, index=g.index)
    grp = df.groupby("g")["y"]
    df["size"] = grp.transform("size")
    df["frac_R"] = grp.transform("mean")
    auc = roc_auc_score(df["y"], df["frac_R"]) if df["y"].nunique() == 2 else np.nan
    stats = df.groupby("g")["y"].agg(["size", "sum"])
    bayes_err = np.minimum(stats["sum"], stats["size"] - stats["sum"]).sum() / len(df)
    return df, auc, float(bayes_err)


def eval_sets(split):
    """{name: isolate ids} from a lineage split frame (role, fold)."""
    out = {"test": split.index[split["role"] == "test"]}
    for k in sorted(split.loc[split["role"] == "train", "fold"].unique()):
        out[f"cv{int(k)}"] = split.index[(split["role"] == "train") & (split["fold"] == k)]
    out["all"] = split.index
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drugs", nargs="+", default=DRUGS)
    ap.add_argument("--tasks", nargs="+", default=["single", "multi"])
    ap.add_argument("--locus-sets", nargs="+", default=["perdrug", "19"])
    ap.add_argument("--inputs", nargs="+", default=INPUTS)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    need = sorted({c for t in args.tasks for ls in args.locus_sets for d in args.drugs
                   for i in args.inputs for c in key_columns(t, ls, d, i)})
    print(f"reading {len(need)} FASTAs from {GENO_DIR} and {REG_DIR}", flush=True)
    alleles = allele_table(need)
    lab = labels()
    print(f"{len(alleles)} isolates with DNA, {len(lab)} phenotype rows", flush=True)

    md_split = pd.read_csv(MD_SPLIT, index_col="New_ID") if "multi" in args.tasks else None
    rows, iso = [], []
    for task in args.tasks:
        for drug in args.drugs:
            if task == "single":
                split = pd.read_csv(SPLIT_DIR / f"{drug}.csv", index_col="New_ID")
                y_all = split["R"].astype(float)
                ph = lab[drug].reindex(split.index)
                mism = int((ph.notna() & (ph != y_all)).sum())
                if mism:
                    raise ValueError(f"{drug}: {mism} split labels disagree with the phenotype table")
            else:
                split = md_split
                y_all = lab[drug].reindex(split.index)
            ids = split.index[y_all.notna() & split.index.isin(alleles.index)]
            if len(ids) < (y_all.notna()).sum():
                print(f"  {task} {drug}: {(y_all.notna()).sum() - len(ids)} labelled isolates lack DNA", flush=True)
            y = y_all.loc[ids]
            sets = {k: v.intersection(ids) for k, v in eval_sets(split.loc[ids]).items()}
            for ls, inp in [(l, i) for l in args.locus_sets for i in args.inputs]:
                g = group_key(alleles.loc[ids], key_columns(task, ls, drug, inp))
                f_r, f_s, n_rmaj, n_smaj = flip_rates(g, y)
                for name, sid in sets.items():
                    df, auc, berr = oracle(g.loc[sid], y.loc[sid])
                    dup = df[df["size"] >= 2]
                    dup_auc = (roc_auc_score(dup["y"], dup["frac_R"])
                               if dup["y"].nunique() == 2 else np.nan)
                    n_pos, n_neg = int(df["y"].sum()), int((1 - df["y"]).sum())
                    rows.append(dict(
                        task=task, locus_set=ls, input=inp, drug=drug, eval_set=name,
                        n=len(df), n_R=n_pos, n_S=n_neg, n_genotypes=df["g"].nunique(),
                        frac_singleton=float((df["size"] == 1).mean()),
                        oracle_auc=auc, bayes_err=berr,
                        frac_discordant=float(((df["frac_R"] > 0) & (df["frac_R"] < 1)).mean()),
                        dup_n=len(dup), dup_R=int(dup["y"].sum()), dup_auc=dup_auc,
                        flip_R_to_S=f_r, flip_S_to_R=f_s, n_Rmaj_groups=n_rmaj, n_Smaj_groups=n_smaj,
                        noise_auc=noise_auc(n_pos, n_neg, f_r, f_s)))
                    if name != "all":
                        iso.append(df.assign(task=task, locus_set=ls, input=inp, drug=drug, eval_set=name)
                                   .rename_axis("New_ID").reset_index())
                r = rows[-1]
                print(f"  {task:6s} {ls:7s} {inp:12s} {drug:13s} all: n={r['n']} genotypes={r['n_genotypes']} "
                      f"singleton={r['frac_singleton']:.3f} oracle={r['oracle_auc']:.4f} "
                      f"dup={r['dup_auc']:.4f} noise={r['noise_auc']:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(out / "ceiling.csv", index=False)
    pd.concat(iso).to_csv(out / "isolates.csv.gz", index=False)
    (out / "meta.json").write_text(json.dumps(dict(
        genotype_dir=str(GENO_DIR), regulatory_dir=str(REG_DIR), phenotype_csv=str(PHENO_CSV), split_dir=str(SPLIT_DIR),
        multidrug_split=str(MD_SPLIT), drugs=args.drugs, tasks=args.tasks,
        locus_sets=args.locus_sets, inputs=args.inputs,
        promoters={**PROMOTERS, "multi_19": PROMOTERS_MD},
        built=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    print(f"wrote {out}/ceiling.csv ({len(rows)} rows) and isolates.csv.gz", flush=True)


if __name__ == "__main__":
    main()

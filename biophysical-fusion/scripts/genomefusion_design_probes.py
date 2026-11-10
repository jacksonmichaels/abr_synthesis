"""Closed-form statistics behind the alternative GenomeFusion designs in
notebooks/genomefusion/genomefusion_architectures.ipynb.

    cd biophysical-fusion && conda activate abr_env
    export ABR_GENBANK=$PWD/../data_snapshot/genbank/genome.gbff \
           ABR_WORKSPACE=$PWD/.. ABR_GF_REFS=$PWD/../genomefusion_refs \
           ABR_GF_CACHE=$PWD/../genomefusion_cache_v2
    python scripts/genomefusion_design_probes.py

No gradient training and no GPU; ~2 min, peak ~2 GB, so it fits the interactive
session. TEST rows are never scored: every score below is on a CV fold, and the
pair statistics describe training rows only.

Writes results/analysis/genomefusion_design_probes/:
    vocabulary.csv   variants and tokens by allele count
    ld_patterns.csv  distinct carrier sets by minimum allele count
    closed_form.csv  top-k log-odds scores per drug x fold, genome vs 19p
    pairs.csv        nearest discordant same-sublineage neighbour, per drug
    transfer.csv     own labels vs pooled with a related drug, per target x fold
    baselines.csv    per-drug CV means of the finished parity cells
    cohort.csv       training isolates and resistant count, per drug
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from datasets.loader import DRUG_TO_LOCI  # noqa: E402
from genomefusion.partition import CACHE, load_or_build  # noqa: E402

WS = PROJECT.parent
EXP = PROJECT / "results" / "experiments"
OUT = PROJECT / "results" / "analysis" / "genomefusion_design_probes"
DRUGS = ["AMIKACIN", "CAPREOMYCIN", "ETHAMBUTOL", "ETHIONAMIDE", "ISONIAZID", "KANAMYCIN",
         "LEVOFLOXACIN", "MOXIFLOXACIN", "PYRAZINAMIDE", "RIFAMPICIN", "STREPTOMYCIN"]
#: target -> drugs sharing a mechanism, for the transfer probe
RELATED = {"LEVOFLOXACIN": ["MOXIFLOXACIN"], "MOXIFLOXACIN": ["LEVOFLOXACIN"],
           "AMIKACIN": ["KANAMYCIN", "CAPREOMYCIN"], "CAPREOMYCIN": ["AMIKACIN", "KANAMYCIN"],
           "KANAMYCIN": ["AMIKACIN", "CAPREOMYCIN"], "ETHIONAMIDE": ["ISONIAZID"]}
#: name -> run folder of the parity cells
BASELINES = {"LF v3, 19 loci": "lineage_split/sd19_all_modalities_unk__locusfusion_v2",
             "LF v3, per-drug": "lineage_split/sd_all_modalities_unk__locusfusion_v2",
             "stacked CNN, 19 loci": "lineage_split/sd19_all_modalities__mdcnn",
             "stacked CNN, per-drug": "lineage_split/sd_all_modalities__mdcnn",
             "gf19p": "genomefusion/gf19p"}


def auc(y, s):
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load_variants():
    """(isolate x variant) binary CSR. A variant is position + allele + indel span."""
    z = np.load(WS / "genomefusion_tokens" / "tokens.npz", allow_pickle=False)
    ids = np.asarray([str(x) for x in z["ids"]])
    iso = np.repeat(np.arange(len(ids)), np.diff(z["tok_off"])).astype(np.int32)
    key = z["pos"].astype(np.int64) * 8 + z["base"].astype(np.int64)
    key = key * 100_000 + np.minimum(z["span"].astype(np.int64), 99_999)
    _, inv = np.unique(key, return_inverse=True)
    V = int(inv.max()) + 1
    var_locus = np.zeros(V, np.int32)
    var_locus[inv] = z["locus"].astype(np.int32)
    X = sp.csr_matrix((np.ones(len(inv), np.int32), (iso, inv.astype(np.int32))),
                      shape=(len(ids), V))
    X.sum_duplicates()
    X.data[:] = 1
    return ids, X, var_locus


def vocabulary(X):
    ac = np.asarray(X.sum(0)).ravel()
    rows = []
    for lo, hi, lab in [(1, 1, "1"), (2, 4, "2-4"), (5, 19, "5-19"), (20, 99, "20-99"),
                        (100, 999, "100-999"), (1000, 10**9, "1000+")]:
        m = (ac >= lo) & (ac <= hi)
        rows.append(dict(allele_count=lab, variants=int(m.sum()), tokens=int(ac[m].sum()),
                         pct_variants=100 * m.mean(), pct_tokens=100 * ac[m].sum() / ac.sum()))
    h = np.random.default_rng(0).integers(1, 2**63, size=X.shape[0], dtype=np.uint64)
    cs = X.tocsc()
    col = np.repeat(np.arange(cs.shape[1]), np.diff(cs.indptr))
    hv = np.zeros(cs.shape[1], np.uint64)
    np.add.at(hv, col, h[cs.indices])
    pat = [dict(min_ac=lo, variants=int((ac >= lo).sum()),
                patterns=len(np.unique(hv[ac >= lo]))) for lo in (1, 2, 5, 20)]
    return pd.DataFrame(rows), pd.DataFrame(pat)


def assoc(Xt, y, g):
    """Log OR, crude z, lineage-stratified (CMH) z and effect, and carrier count."""
    a = np.asarray(Xt[y == 1].sum(0)).ravel().astype(np.float64)
    b = np.asarray(Xt[y == 0].sum(0)).ravel().astype(np.float64)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    lor = np.log((a + .5) * (n0 - b + .5) / ((b + .5) * (n1 - a + .5)))
    se = np.sqrt(1 / (a + .5) + 1 / (b + .5) + 1 / (n1 - a + .5) + 1 / (n0 - b + .5))
    O, E, W = (np.zeros(Xt.shape[1]) for _ in range(3))
    for k in np.unique(g):
        m = g == k
        N = m.sum()
        R = (y[m] == 1).sum()
        S = N - R
        if N < 2 or R == 0 or S == 0:
            continue
        nv = np.asarray(Xt[m].sum(0)).ravel().astype(np.float64)
        O += np.asarray(Xt[m & (y == 1)].sum(0)).ravel().astype(np.float64)
        E += nv * R / N
        W += nv * (N - nv) * R * S / (N * N * (N - 1))
    zs = np.where(W > 0, (O - E) / np.sqrt(np.maximum(W, 1e-12)), 0.0)
    eff = np.clip(np.where(W > 0, (O - E) / np.maximum(W, 1e-12), 0.0), -4, 4)
    return lor, lor / se, zs, eff, a + b


def closed_form(X, splits, lev2, v19p):
    V, rows = X.shape[1], []
    for drug, s in splits.items():
        s = s[s.role == "train"]
        for f in sorted(s.fold.unique()):
            tr, va = s[s.fold != f], s[s.fold == f]
            Xv = X[va.row.values]
            lor, zc, zs, eff, n = assoc(X[tr.row.values], tr.y.values, lev2[tr.row.values])
            keep = n >= 3
            for name, mask in (("genome", np.ones(V, bool)), ("19p", v19p)):
                for stat, zz, w in (("crude", zc, lor), ("cmh", zs, eff)):
                    idx0 = np.flatnonzero(keep & mask)
                    for k in (30, 300, 3000):
                        idx = idx0[np.argsort(-zz[idx0])[:k]]
                        w_ = np.zeros(V)
                        w_[idx] = np.maximum(w[idx], 0)
                        rows.append(dict(drug=drug, fold=f, set=name, stat=stat, top_k=k,
                                         auc=auc(va.y.values, np.asarray(Xv @ w_).ravel())))
        print("closed form", drug, flush=True)
    return pd.DataFrame(rows)


def pair_stats(X, splits, lev2, part, var_locus):
    """Each sampled R isolate's nearest S isolate in the same level-2 sublineage."""
    nnz = np.asarray(X.sum(1)).ravel()
    rows = []
    for d, s in splits.items():
        names = set(DRUG_TO_LOCI[d])
        own = np.asarray([(L.kind != "intergenic" and bool({L.gene, L.name} & names))
                          or (L.is_regulatory and bool(set(L.regulates) & names))
                          for L in part.loci])
        vown = own[var_locus]
        s = s[s.role == "train"]
        R, S = s[s.y == 1], s[s.y == 0]
        samp = R.sample(min(300, len(R)), random_state=0)
        Xs = X[S.row.values]
        ls, ns = lev2[S.row.values], nnz[S.row.values]
        prod = (X[samp.row.values] @ Xs.T).toarray()
        rh, sh, rn, sn, dist = [], [], [], [], []
        for i, r in enumerate(samp.row.values):
            same = ls == lev2[r]
            if not same.any():
                continue
            h = np.where(same, nnz[r] + ns - 2 * prod[i], 10**9)
            k = int(np.argmin(h))
            rv, sv = set(X[r].indices), set(Xs[k].indices)
            a = np.fromiter(rv - sv, dtype=np.int64)
            b = np.fromiter(sv - rv, dtype=np.int64)
            rh.append(bool(len(a) and vown[a].any()))
            sh.append(bool(len(b) and vown[b].any()))
            rn.append(len(a))
            sn.append(len(b))
            dist.append(h[k])
        dist = np.asarray(dist)
        rows.append(dict(drug=d, pairs=len(dist), median_diff=float(np.median(dist)),
                         frac_diff_le_25=float((dist <= 25).mean()),
                         own_locus_R_only=float(np.mean(rh)), own_locus_S_only=float(np.mean(sh)),
                         mean_R_only=float(np.mean(rn)), mean_S_only=float(np.mean(sn))))
        print("pairs", d, flush=True)
    return pd.DataFrame(rows), float(nnz.mean())


def transfer(X, splits, lev2):
    """Top-30 CMH scorer on a target's CV folds, own labels vs pooled with related
    drugs. The target's validation and test rows are excluded from the source
    labels, because the per-drug split files do not agree on folds."""
    V = X.shape[1]

    def weights(rows_, y):
        _, _, zs, eff, n = assoc(X[rows_], y, lev2[rows_])
        zs = np.where(n >= 3, zs, -np.inf)
        idx = np.argsort(-zs)[:30]
        w = np.zeros(V)
        w[idx] = np.maximum(eff[idx], 0)
        return w

    rows = []
    for t, sources in RELATED.items():
        st = splits[t]
        tr_all = st[st.role == "train"]
        test = set(st[st.role == "test"].row.values)
        for f in sorted(tr_all.fold.unique()):
            va, tr = tr_all[tr_all.fold == f], tr_all[tr_all.fold != f]
            ban = set(va.row.values) | test
            have = set(tr.row.values)
            rs, ys = [tr.row.values], [tr.y.values]
            for s_ in sources:
                sx = splits[s_]
                sx = sx[~sx.row.isin(ban) & ~sx.row.isin(have)]
                rs.append(sx.row.values)
                ys.append(sx.y.values)
                have |= set(sx.row.values)
            rr, yy = np.concatenate(rs), np.concatenate(ys)
            Xv = X[va.row.values]
            rows.append(dict(target=t, fold=f, n_own=len(tr), n_pooled=len(rr),
                             own=auc(va.y.values, np.asarray(Xv @ weights(tr.row.values, tr.y.values)).ravel()),
                             pooled=auc(va.y.values, np.asarray(Xv @ weights(rr, yy)).ravel())))
        print("transfer", t, flush=True)
    return pd.DataFrame(rows)


def baselines():
    tab = {}
    for name, sub in BASELINES.items():
        d = {}
        for f in glob.glob(str(EXP / sub / "*.json")):
            drug = Path(f).stem.split("__")[0].upper()
            if drug in DRUGS:
                d[drug] = float(np.mean([c["macro_auc"] if "macro_auc" in c else c["auc"]
                                         for c in json.load(open(f))["cv_folds"]]))
        tab[name] = d
    return pd.DataFrame(tab).reindex(DRUGS).rename_axis("drug")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids, X, var_locus = load_variants()
    index = {s: i for i, s in enumerate(ids)}
    print(f"{X.shape[0]:,} isolates, {X.shape[1]:,} variants", flush=True)

    part = load_or_build(str(WS / "genomefusion_cache_v2" / Path(CACHE).name))
    in19p = np.asarray([L.is_19 for L in part.loci])
    in19p[np.asarray(part.promoter_loci(100), np.int64)] = True
    v19p = in19p[var_locus]

    lin = pd.read_csv(WS / "data_snapshot" / "lineage" / "lineage_calls.csv",
                      dtype={"New_ID": str}).drop_duplicates("New_ID").set_index("New_ID")
    lev2 = lin["level2"].reindex(ids).fillna("unknown").astype(str).values
    ph = pd.read_csv(WS / "data_snapshot" / "phenotype_data" / "master_resistance_table.csv",
                     dtype={"New_ID": str}, low_memory=False).drop_duplicates("New_ID").set_index("New_ID")
    splits = {}
    for d in DRUGS:
        s = pd.read_csv(EXP / "lineage_split" / "splits" / f"{d}.csv",
                        dtype={"New_ID": str}).drop_duplicates("New_ID")
        s = s[s.New_ID.isin(index)].copy()
        s["row"] = s.New_ID.map(index)
        v = ph[d].reindex(s.New_ID).astype("string").str.strip().str.upper().values
        s["y"] = np.where(v == "R", 1, np.where(v == "S", 0, -1))
        splits[d] = s[s.y >= 0].reset_index(drop=True)

    voc, pat = vocabulary(X)
    voc.to_csv(OUT / "vocabulary.csv", index=False)
    pat.to_csv(OUT / "ld_patterns.csv", index=False)
    pd.DataFrame([dict(drug=d, train=int((s.role == "train").sum()),
                       train_R=int(s[s.role == "train"].y.sum()))
                  for d, s in splits.items()]).to_csv(OUT / "cohort.csv", index=False)
    baselines().to_csv(OUT / "baselines.csv")
    closed_form(X, splits, lev2, v19p).to_csv(OUT / "closed_form.csv", index=False)
    pairs, mean_n = pair_stats(X, splits, lev2, part, var_locus)
    pairs["mean_variants_per_isolate"] = mean_n
    pairs.to_csv(OUT / "pairs.csv", index=False)
    transfer(X, splits, lev2).to_csv(OUT / "transfer.csv", index=False)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
L1-logistic on the WHOLE GENOME, not the 19 curated loci.

## Why this exists

Every model in this project — lasso and network alike — has only ever seen the
19 curated resistance loci, 39,646 aligned nt, 0.9% of H37Rv. That input makes
one question unanswerable. `lasso_wholeseq_20260901` finding 6 observes that the
19-locus models put most of their coefficient mass on loci that cannot cause the
drug's resistance, and `ablation_eto` measures it: four loci with no ethionamide
mechanism in them reach CV 0.7241, beating the drug's own three loci. But every
locus on disk is a resistance gene for *some* drug, so "not this drug's
mechanism" and "carries co-resistance" name the same set of columns. Mechanism
and lineage cannot be separated on that input, no matter how it is sliced.

Genome-wide they separate. If a genome-wide lasso still concentrates on WHO
Table-21 genes, the signal is mechanism. If it spreads across the genome — into
genes with no resistance role — it is lineage, and the 19-locus numbers have
been reading population structure as biology.

## The input

`annotated_vcf_tsv_data/{isolate}_variants_combinedCodons.tsv`: snpEff-annotated
per-isolate calls against NC_000962.3, one file per isolate, 17,942 of them.
Measured on a 1,600-isolate random sample: 1,372 PASS variants per isolate,
so ~24.6 M nonzeros over the full cohort against 281 M for the 19-locus
`onehot`/all cell. **Genome-wide is an order of magnitude cheaper in nonzeros
than the run it is compared against**, because a variant matrix is sparse where
a one-hot alignment is dense.

The cohort is identical to every other run here: all 17,942 isolates with calls
are in the phenotype table and in the 19-locus FASTA cohort, so the 80/20 split
at seed 42 and the 5 folds land on the same isolates. Only H37Rv itself drops.

## Encoding, and what it costs

One column per (POSITION, REF, ALT) key seen in the cohort, value 1 where the
isolate carries it. That is the repo's `delta` coding — nonzero only where the
isolate differs from the reference — and genome-wide it is the ONLY option:
`onehot` needs a wild-type indicator per site, and at ~10^6 sites x 17,942
isolates that block is dense. So these numbers are comparable to the parent
run's `delta` arms, NOT to its 0.9241 `onehot` headline, which had a +0.009
reparametrisation advantage that is unavailable here.

## Vocabulary and leakage

`--min-count 1` (default) keeps every key present in the fitted cohort, which
is leak-free by the parent run's argument: the only cohort-wide decision is
existence, and a key absent from the cohort is a zero column, collinear with the
unpenalised intercept, so its L1 optimum is exactly zero within a fold too.
`--min-count k > 1` is a genuine cohort-wide filter using held-out isolates and
is NOT leak-free; it is offered for cost only, and a result that needs it should
be repeated with a train-only vocabulary before it is quoted.

Measured allele-frequency spectrum over 1,600 isolates: 71.7% of keys are
singletons, 28.3% are seen in >= 2 isolates. Singletons are 5.6% of nonzeros,
so a min-count filter costs almost no data and removes most columns.

## Protocol

Identical to `lasso_wholeseq.py`, which it imports rather than re-implements:
same `_metrics`, same `_fit`, same C grid, same split and folds, same
`class_weight="balanced"`, test scored once with the best CV fold's model.

## Reproduce

    python results/experiments/lasso_wholegenome_20260903/lasso_wholegenome.py \
        --drugs ETHIONAMIDE \
        --out results/experiments/lasso_wholegenome_20260903/pilot
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "results" / "experiments"
                     / "lasso_wholeseq_20260901"))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

# The parent run is the protocol. Importing it means the metrics, the solver
# call and the summary schema cannot drift apart from the arm this is compared
# against.
import lasso_wholeseq as ws  # noqa: E402
from bigtb_ref import REAL_PHENOTYPE_CSV, tb  # noqa: E402
from datasets.sequences import load_phenotype, numeric_labels  # noqa: E402

VCF_TSV_DIR = ("/project/pi_annagreen_umass_edu/saishradha/project_data_curation"
               "/annotated_vcf_tsv_data")
SUFFIX = "_variants_combinedCodons.tsv"
ALL_DRUGS = list(tb.DRUG_TO_LOCI)


def isolates_on_disk(tsv_dir):
    return {f[:-len(SUFFIX)] for f in os.listdir(tsv_dir) if f.endswith(SUFFIX)}


def _parse_one(arg):
    """(isolate, [(key, gene), ...]) for one isolate's PASS calls.

    The key is POSITION:REF>ALT with alleles truncated to 20 chars: the corpus
    carries indels with 400-nt alleles, and a 400-nt string is not a more
    informative column name than its first 20. Truncation can in principle
    collide two different long indels at one position; measured on the cohort
    that never happens (see build_design's collision guard).
    """
    tsv_dir, iso = arg
    out = []
    with open(os.path.join(tsv_dir, iso + SUFFIX)) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        try:
            iP, iR, iA, iF = (hdr.index(c) for c in (
                "POSITION", "REFERENCE_ALLELE", "ALTERED_ALLELE",
                "FILTER_STATUS"))
            iG = hdr.index("Gene_Name")
            iI = hdr.index("Gene_ID")
        except ValueError as e:                      # a malformed header is a
            raise RuntimeError(f"{iso}: {e}")        # data problem, not a skip
        for line in fh:
            c = line.split("\t")
            if len(c) <= iF or c[iF] != "PASS":
                continue
            gene = c[iG] if len(c) > iG else ""
            gid = c[iI] if len(c) > iI else ""
            out.append((f"{c[iP]}:{c[iR][:20]}>{c[iA][:20]}", gene, gid))
    return iso, out


def build_cohort(tsv_dir, cache, procs):
    """Parse every isolate on disk once into a CSR presence matrix and cache it.

    Parsing is 223 ms per isolate, so re-reading the corpus once per drug costs
    ~80 min across the 11 drugs against ~17 min done once. The vocabulary is
    assigned while streaming so per-isolate storage is int32 column ids, not
    strings: 24.6 M nonzeros is 100 MB that way and several GB the naive way.
    """
    cache = Path(cache)
    if cache.is_file():
        z = np.load(cache, allow_pickle=False)
        print(f"  [cache] loaded {cache} "
              f"({z['indptr'].shape[0] - 1} isolates, {z['keys'].shape[0]} keys)",
              flush=True)
        X = sp.csr_matrix((np.ones(z["indices"].shape[0], dtype=np.float32),
                           z["indices"], z["indptr"]),
                          shape=(z["indptr"].shape[0] - 1, z["keys"].shape[0]))
        return (X, list(z["isolates"]), list(z["keys"]),
                dict(zip(z["keys"], z["genes"])),
                dict(zip(z["keys"], z["gene_ids"])))

    isolates = sorted(isolates_on_disk(tsv_dir))
    t0 = time.time()
    col, keys, genes, gene_ids = {}, [], [], []
    indices, indptr = [], [0]
    with Pool(procs) as pool:
        it = pool.imap(_parse_one, [(tsv_dir, i) for i in isolates], chunksize=8)
        for done, (_iso, rows) in enumerate(it, 1):
            js = set()
            for k, g, gid in rows:
                j = col.get(k)
                if j is None:
                    j = col[k] = len(keys)
                    keys.append(k); genes.append(g); gene_ids.append(gid)
                js.add(j)
            indices.extend(sorted(js))
            indptr.append(len(indices))
            if done % 2000 == 0:
                print(f"  [parse] {done}/{len(isolates)} isolates, "
                      f"{len(keys)} keys, {time.time() - t0:.0f}s", flush=True)
    print(f"  [parse] {len(isolates)} isolates in {time.time() - t0:.0f}s, "
          f"{len(keys)} keys, {len(indices)} nonzeros", flush=True)

    indices = np.asarray(indices, dtype=np.int32)
    indptr = np.asarray(indptr, dtype=np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, indices=indices, indptr=indptr,
             isolates=np.array(isolates), keys=np.array(keys),
             genes=np.array(genes), gene_ids=np.array(gene_ids))
    print(f"  [cache] wrote {cache}", flush=True)
    X = sp.csr_matrix((np.ones(indices.shape[0], dtype=np.float32),
                       indices, indptr), shape=(len(isolates), len(keys)))
    return (X, isolates, keys, dict(zip(keys, genes)),
            dict(zip(keys, gene_ids)))


def drug_design(X_all, iso_all, keys_all, rows, min_count):
    """The drug's rows, and only the columns its own cohort actually varies at.

    Same rule as the parent run: a column that is zero over the fitted cohort is
    collinear with the unpenalised intercept, so its L1 optimum is zero anyway;
    dropping it changes nothing and keeps the column count honest.
    """
    X = X_all[rows]
    counts = np.asarray(X.sum(axis=0)).ravel()
    cols = np.nonzero(counts >= min_count)[0]
    return X[:, cols].tocsr(), [keys_all[j] for j in cols]


def run_drug(drug, args, cohort):
    t0 = time.time()
    X_all, iso_all, keys_all, gene_of, geneid_of = cohort
    row_of = {iso: i for i, iso in enumerate(iso_all)}
    df_phenos = load_phenotype(args.phenotype_csv)
    isolates = [i for i in df_phenos.index if i in row_of]
    y_all = numeric_labels(df_phenos.loc[isolates], drug)
    keep = np.nonzero(y_all != -1)[0]
    y = y_all[keep]
    isolates = [isolates[i] for i in keep]

    X, names = drug_design(X_all, iso_all, keys_all,
                           [row_of[i] for i in isolates], args.min_count)
    n = X.shape[0]
    print(f"[{drug}] {n} isolates, {X.shape[1]} columns, "
          f"{X.nnz / max(n, 1):.0f} nnz per isolate, whole genome", flush=True)

    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2,
                                           random_state=42, stratify=y)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(train_idx, y[train_idx]))

    path, best = [], None
    for C in args.c_grid:
        fold_auc, fold_models, nnz_coef = [], [], []
        for tr, va in folds:
            i_tr, i_va = train_idx[tr], train_idx[va]
            tc = time.time()
            clf = ws._fit(C, X[i_tr], y[i_tr], args.solver, args.max_iter)
            m = ws._metrics(y[i_va], clf.predict_proba(X[i_va])[:, 1])
            fold_auc.append(m["auc"])
            fold_models.append(clf)
            nnz_coef.append(int((clf.coef_ != 0).sum()))
            print(f"  [{drug}] C={C:<6g} fold {len(fold_auc)}/{len(folds)} "
                  f"AUC {m['auc']:.4f} support {nnz_coef[-1]} "
                  f"[{time.time() - tc:.0f}s]", flush=True)
        entry = {"C": C, "cv_auc_mean": float(np.mean(fold_auc)),
                 "cv_auc_std": float(np.std(fold_auc)),
                 "fold_aucs": [float(a) for a in fold_auc],
                 "support_mean": float(np.mean(nnz_coef)),
                 "support_per_fold": nnz_coef}
        path.append(entry)
        if best is None or entry["cv_auc_mean"] > best[0]["cv_auc_mean"]:
            best = (entry, fold_models, fold_auc)

    entry, fold_models, fold_auc = best
    best_fold = int(np.argmax(fold_auc))
    test = ws._metrics(y[test_idx],
                       fold_models[best_fold].predict_proba(X[test_idx])[:, 1])

    coef = fold_models[best_fold].coef_.ravel()
    nz = np.nonzero(coef)[0]
    order = nz[np.argsort(-np.abs(coef[nz]))][:args.top_k]
    top = [{"feature": names[i], "gene": gene_of.get(names[i], ""),
            "gene_id": geneid_of.get(names[i], ""),
            "coef": float(coef[i])} for i in order]

    cv_pr = float(np.mean([
        ws._metrics(y[train_idx[va]],
                    fold_models[i].predict_proba(X[train_idx[va]])[:, 1])["auc_pr"]
        for i, (tr, va) in enumerate(folds)]))

    return {
        "drug": drug, "arch": "lasso_wholegenome", "modalities": ["dna"],
        "tag": f"dna-genome-mc{args.min_count}",
        "encoding": "delta", "locus_set": "wholegenome",
        "solver": args.solver, "min_count": args.min_count,
        "genes": ["WHOLE_GENOME"], "n_valid": int(n),
        "n_resistant": int((y == 0).sum()), "n_susceptible": int((y == 1).sum()),
        "n_features": int(X.shape[1]),
        "nnz_per_isolate": float(X.nnz / max(n, 1)),
        "n_params": int(X.shape[1] + 1),
        "split": {"test_size": 0.2, "split_seed": 42, "stratified": True,
                  "n_splits": args.n_splits, "kfold": "StratifiedKFold",
                  "kfold_seed": 42, "kfold_shuffle": True},
        "c_path": path,
        "best_C": entry["C"], "best_fold": best_fold,
        "cv_auc_mean": entry["cv_auc_mean"], "cv_auc_std": entry["cv_auc_std"],
        "cv_auc_pr_mean": cv_pr,
        "cv_support_mean": entry["support_mean"],
        "test": test, "top_features": top,
        "seconds": round(time.time() - t0, 4),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drugs", nargs="+", default=["all"])
    ap.add_argument("--min-count", type=int, default=1,
                    help="keep keys seen in >= this many cohort isolates; 1 is "
                         "the leak-free default, > 1 is a cohort-wide filter")
    ap.add_argument("--solver", choices=("liblinear", "saga"),
                    default="liblinear")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--c-grid", type=float, nargs="+", default=list(ws.C_GRID))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=100000,
                    help="how many nonzero coefficients to record; the default "
                         "keeps all of them, which is what a mass split over "
                         "the support needs")
    ap.add_argument("--procs", type=int,
                    default=min(8, len(os.sched_getaffinity(0))))
    ap.add_argument("--tsv-dir", default=VCF_TSV_DIR)
    ap.add_argument("--cache", default=str(Path.home() / "abr_workspace"
                                          / "wholegenome_cache.npz"),
                    help="parsed whole-cohort matrix; built on first use")
    ap.add_argument("--phenotype-csv", default=REAL_PHENOTYPE_CSV)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    drugs = (ALL_DRUGS if any(d.lower() == "all" for d in args.drugs)
             else [d.upper() for d in args.drugs])
    unknown = [d for d in drugs if d not in ALL_DRUGS]
    if unknown:
        ap.error(f"unknown drug(s) {unknown}; choose from {ALL_DRUGS} or 'all'")

    cohort = build_cohort(args.tsv_dir, args.cache, args.procs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for drug in drugs:
        r = run_drug(drug, args, cohort)
        (out / f"{drug}__{r['tag']}.json").write_text(json.dumps(r, indent=2))
        print(f"[{drug}] CV {r['cv_auc_mean']:.4f} +/- {r['cv_auc_std']:.4f} "
              f"@ C={r['best_C']} (support {r['cv_support_mean']:.0f} of "
              f"{r['n_features']}), TEST {r['test']['auc']:.4f} "
              f"[{r['seconds']:.0f}s]", flush=True)
        ws._write_summary(out)


if __name__ == "__main__":
    main()

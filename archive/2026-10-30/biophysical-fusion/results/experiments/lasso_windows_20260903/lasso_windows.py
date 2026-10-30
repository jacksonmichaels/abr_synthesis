#!/usr/bin/env python3
"""
L1-logistic on 200 bp windows around the whole-genome model's top 50 coefficients.

## Why this exists

`lasso_wholegenome_20260903` fits the genome and reports two things: it matches
the 19-locus `delta` arm (+0.0022 macro CV, 6 wins of 11) and it puts 62.4% of
its coefficient mass outside the WHO Table-21 genes for the drug. Both are
statements about a model with ~10^5 columns and a support of 22-732. This run
asks what happens when the genome is thrown away and only the neighbourhood of
that support is kept: take the 50 largest coefficients, open a 200 bp window
around each, and refit on those windows alone.

Two things change relative to the parent, and the three arms separate them.

  1. **Context.** A window carries every variant near the selected one, not just
     the selected one. If the selected key is a tag for a haplotype, its
     neighbours carry the same information and the window should not beat the
     bare key by much. If the selected key is the causal site, the neighbours are
     noise the L1 has to spend budget rejecting.
  2. **Encoding.** The genome-wide run could only use `delta` (a wild-type
     indicator per site is dense at 10^6 sites). Over ~10^3 window sites it is
     affordable, and `lasso_wholeseq_20260901` measured `onehot` at +0.009 macro
     over `delta` on the 19 loci. This is the first chance to see whether that
     reparametrisation advantage survives outside the curated loci.

## The arms

    delta-top50     the <=50 selected keys only, delta coding
    delta-window    every cohort key inside the windows, delta coding
    onehot-window   the same keys plus a wild-type indicator per polymorphic
                    site  <- the arm the experiment was asked for

`delta-top50` is the floor: what the selection alone is worth. `delta-window`
adds context at fixed encoding. `onehot-window` adds the encoding at fixed
columns. Reading the requested arm without the other two cannot tell a context
effect from an encoding effect.

## What "one-hot the window" means here, exactly

There is no whole-genome H37Rv FASTA in the corpus -- `genomic_data/aligned`
holds the 19 loci and nothing else -- so a window's sequence is reconstructed
from the cohort's own calls rather than read off a reference. That costs
nothing, because a position where no isolate carries a call is constant across
the cohort, and a constant column is collinear with the unpenalised intercept:
its L1 optimum is exactly zero, in every fold. So the encoded window is its
polymorphic positions, and the columns at each such position are

    one per distinct REF>ALT key observed there, plus one wild-type indicator
    set where the isolate carries none of them

which is the repo's `onehot` coding (indicator per site x allele, wild type
included) lifted from an alignment column to a variant site. Indels are keyed by
their start position and treated as one allele there, matching the parent run.
The JSON records `window_bp` next to `n_sites` so the density is visible: a
200 bp window over this cohort is roughly 30% polymorphic.

## Selection provenance, and which number is clean

The 50 keys come from the parent run's best-fold model, which was fitted on four
fifths of the 80% training split at seed 42. This run reuses that same split and
those same folds. Therefore:

  * **TEST AUC is clean.** The held-out 20% took no part in choosing the windows,
    and it is scored once, so it is directly comparable to the parent's test AUC.
  * **CV AUC is selection-optimistic.** The selection saw ~64% of the data, which
    overlaps every validation fold's complement. The CV number is reported
    because the C is chosen on it and the parent quotes it, but a window-vs-genome
    comparison should be read off the test column.

Honest CV would re-select the 50 keys inside each fold, which means a
genome-wide refit per fold; that is a bigger run than this one and is not done
here. The gap between the two columns is itself the estimate of how much the
selection is worth.

## Reproduce

    python results/experiments/lasso_windows_20260903/lasso_windows.py \
        --drugs all \
        --wholegenome results/experiments/lasso_wholegenome_20260903/all_drugs \
        --out results/experiments/lasso_windows_20260903/all_drugs
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "results" / "experiments"
                       / "lasso_wholeseq_20260901"))
sys.path.insert(0, str(PROJECT_DIR / "results" / "experiments"
                       / "lasso_wholegenome_20260903"))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

import lasso_wholeseq as ws  # noqa: E402
import lasso_wholegenome as wg  # noqa: E402
from bigtb_ref import REAL_PHENOTYPE_CSV, tb  # noqa: E402
from datasets.sequences import load_phenotype, numeric_labels  # noqa: E402

ALL_DRUGS = list(tb.DRUG_TO_LOCI)
ARMS = ("delta-top50", "delta-window", "onehot-window")


def key_pos(key):
    return int(key.split(":", 1)[0])


def top_keys(wg_dir, drug, top_k):
    """The drug's top_k nonzero coefficients, largest |coef| first.

    A drug whose genome-wide support is smaller than top_k contributes its whole
    support; LEVOFLOXACIN has 22 nonzero coefficients, so its 'top 50' is 22 and
    the JSON says so in n_selected.
    """
    hits = sorted(Path(wg_dir).glob(f"{drug}__*.json"))
    if len(hits) != 1:
        raise SystemExit(f"expected one {drug} json in {wg_dir}, found {hits}")
    d = json.loads(hits[0].read_text())
    return [f["feature"] for f in d["top_features"][:top_k]], d


def windows(positions, half):
    """Merged [start, end] intervals, `half` either side of each position.

    Windows are merged when they touch or overlap, so a cluster of selected
    sites inside one gene is one interval rather than several copies of the same
    columns. `window_bp` in the JSON is the merged footprint, which is what was
    actually encoded, not 200 x n_selected.
    """
    out = []
    for p in sorted(set(positions)):
        s, e = p - half, p + half - 1          # 2 * half = the requested width
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(w) for w in out]


def cols_in_windows(pos_all, wins):
    """Cohort column ids whose position falls in any window, position-sorted."""
    order = np.argsort(pos_all, kind="stable")
    ps = pos_all[order]
    keep = []
    for s, e in wins:
        lo, hi = np.searchsorted(ps, [s, e + 1])
        keep.append(order[lo:hi])
    return np.sort(np.concatenate(keep)) if keep else np.array([], dtype=int)


def add_wildtype(X, names):
    """Append one wild-type indicator per polymorphic site.

    WT_p is 1 where the isolate carries none of the alleles keyed at p. Sites
    where every isolate carries a call have an all-zero WT column and sites where
    none does have an all-one column; both are collinear with the intercept, so
    they are dropped here rather than handed to the solver.
    """
    by_pos = {}
    for j, nm in enumerate(names):
        by_pos.setdefault(key_pos(nm), []).append(j)
    n = X.shape[0]
    cols, wt_names = [], []
    for p in sorted(by_pos):
        carried = np.asarray(X[:, by_pos[p]].sum(axis=1)).ravel()
        rows = np.nonzero(carried == 0)[0]
        if 0 < rows.size < n:
            cols.append(sp.csc_matrix(
                (np.ones(rows.size, dtype=np.float32),
                 (rows, np.zeros(rows.size, dtype=np.int32))), shape=(n, 1)))
            wt_names.append(f"{p}:WT")
    if not cols:
        return X, list(names)
    return sp.hstack([X] + cols, format="csr"), list(names) + wt_names


def build_arm(arm, X_all, rows, keys_all, pos_all, selected, wins, min_count):
    """(design, column names) for one arm of one drug."""
    if arm == "delta-top50":
        want = set(selected)
        idx = np.array([j for j, k in enumerate(keys_all) if k in want],
                       dtype=np.int64)
    else:
        idx = cols_in_windows(pos_all, wins)

    X = X_all[rows][:, idx].tocsr()
    names = [keys_all[j] for j in idx]
    counts = np.asarray(X.sum(axis=0)).ravel()      # same min-count rule as the
    live = np.nonzero(counts >= min_count)[0]       # parent: an all-zero column
    X = X[:, live].tocsr()                          # is collinear with the
    names = [names[j] for j in live]                # intercept anyway
    if arm == "onehot-window":
        X, names = add_wildtype(X, names)
    return X, names


def run_drug(drug, args, cohort):
    X_all, iso_all, keys_all, gene_of, geneid_of = cohort
    pos_all = np.array([key_pos(k) for k in keys_all], dtype=np.int64)
    row_of = {iso: i for i, iso in enumerate(iso_all)}

    selected, parent = top_keys(args.wholegenome, drug, args.top_k)
    wins = windows([key_pos(k) for k in selected], args.half_window)
    window_bp = sum(e - s + 1 for s, e in wins)

    df_phenos = load_phenotype(args.phenotype_csv)
    isolates = [i for i in df_phenos.index if i in row_of]
    y_all = numeric_labels(df_phenos.loc[isolates], drug)
    keep = np.nonzero(y_all != -1)[0]
    y = y_all[keep]
    rows = [row_of[isolates[i]] for i in keep]
    n = len(rows)

    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2,
                                           random_state=42, stratify=y)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(train_idx, y[train_idx]))

    results = []
    for arm in args.arms:
        t0 = time.time()
        X, names = build_arm(arm, X_all, rows, keys_all, pos_all, selected,
                             wins, args.min_count)
        sites = len({key_pos(nm) for nm in names})
        print(f"[{drug}/{arm}] {n} isolates, {X.shape[1]} columns over "
              f"{sites} sites, {X.nnz / max(n, 1):.0f} nnz per isolate",
              flush=True)

        path, best = [], None
        for C in args.c_grid:
            fold_auc, fold_models, nnz_coef = [], [], []
            for tr, va in folds:
                i_tr, i_va = train_idx[tr], train_idx[va]
                clf = ws._fit(C, X[i_tr], y[i_tr], args.solver, args.max_iter)
                m = ws._metrics(y[i_va], clf.predict_proba(X[i_va])[:, 1])
                fold_auc.append(m["auc"])
                fold_models.append(clf)
                nnz_coef.append(int((clf.coef_ != 0).sum()))
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
        cv_pr = float(np.mean([
            ws._metrics(y[train_idx[va]],
                        fold_models[i].predict_proba(X[train_idx[va]])[:, 1]
                        )["auc_pr"] for i, (tr, va) in enumerate(folds)]))

        coef = fold_models[best_fold].coef_.ravel()
        nz = np.nonzero(coef)[0]
        order = nz[np.argsort(-np.abs(coef[nz]))][:args.record_k]
        top = [{"feature": names[i], "gene": gene_of.get(names[i], ""),
                "gene_id": geneid_of.get(names[i], ""),
                "coef": float(coef[i]),
                "selected": names[i] in set(selected)} for i in order]

        results.append({
            "drug": drug, "arch": "lasso_windows", "modalities": ["dna"],
            "tag": arm,
            "arm": arm,
            "encoding": "onehot" if arm == "onehot-window" else "delta",
            "locus_set": f"top{args.top_k}-windows",
            "selection": {
                "source": str(Path(args.wholegenome)),
                "parent_cv_auc": parent["cv_auc_mean"],
                "parent_test_auc": parent["test"]["auc"],
                "parent_support": int(round(parent["cv_support_mean"])),
                "parent_n_features": parent["n_features"],
                "top_k_requested": args.top_k,
                "n_selected": len(selected),
                "n_windows": len(wins),
                "half_window": args.half_window,
                "window_bp": window_bp,
                "cv_is_selection_optimistic": True,
                "test_is_clean": True,
            },
            "windows": [list(w) for w in wins],
            "solver": args.solver, "min_count": args.min_count,
            "genes": ["TOP50_WINDOWS"], "n_valid": int(n),
            "n_resistant": int((y == 0).sum()),
            "n_susceptible": int((y == 1).sum()),
            "n_features": int(X.shape[1]), "n_sites": int(sites),
            "nnz_per_isolate": float(X.nnz / max(n, 1)),
            "n_params": int(X.shape[1] + 1),
            "split": {"test_size": 0.2, "split_seed": 42, "stratified": True,
                      "n_splits": args.n_splits, "kfold": "StratifiedKFold",
                      "kfold_seed": 42, "kfold_shuffle": True},
            "c_path": path,
            "best_C": entry["C"], "best_fold": best_fold,
            "cv_auc_mean": entry["cv_auc_mean"],
            "cv_auc_std": entry["cv_auc_std"],
            "cv_auc_pr_mean": cv_pr,
            "cv_support_mean": entry["support_mean"],
            "test": test, "top_features": top,
            "seconds": round(time.time() - t0, 4),
        })
    return results


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drugs", nargs="+", default=["all"])
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--top-k", type=int, default=50,
                    help="how many of the parent's coefficients seed the windows")
    ap.add_argument("--half-window", type=int, default=100,
                    help="nt either side of a selected position; 100 gives the "
                         "200 bp window")
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--solver", choices=("liblinear", "saga"),
                    default="liblinear")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--c-grid", type=float, nargs="+", default=list(ws.C_GRID))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--record-k", type=int, default=100000)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--tsv-dir", default=wg.VCF_TSV_DIR)
    ap.add_argument("--cache", default=str(Path.home() / "abr_workspace"
                                           / "wholegenome_cache.npz"))
    ap.add_argument("--phenotype-csv", default=REAL_PHENOTYPE_CSV)
    ap.add_argument("--wholegenome", required=True,
                    help="the parent run's result directory; its top_features "
                         "are the selection")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    drugs = (ALL_DRUGS if any(d.lower() == "all" for d in args.drugs)
             else [d.upper() for d in args.drugs])
    unknown = [d for d in drugs if d not in ALL_DRUGS]
    if unknown:
        ap.error(f"unknown drug(s) {unknown}; choose from {ALL_DRUGS} or 'all'")

    cohort = wg.build_cohort(args.tsv_dir, args.cache, args.procs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for drug in drugs:
        for r in run_drug(drug, args, cohort):
            (out / f"{drug}__{r['tag']}.json").write_text(json.dumps(r, indent=2))
            print(f"[{drug}/{r['arm']}] CV {r['cv_auc_mean']:.4f} +/- "
                  f"{r['cv_auc_std']:.4f} @ C={r['best_C']} "
                  f"(support {r['cv_support_mean']:.0f} of {r['n_features']}), "
                  f"TEST {r['test']['auc']:.4f} "
                  f"[parent CV {r['selection']['parent_cv_auc']:.4f} "
                  f"TEST {r['selection']['parent_test_auc']:.4f}] "
                  f"[{r['seconds']:.0f}s]", flush=True)
            ws._write_summary(out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
L1-logistic ("lasso") on the WHOLE aligned sequence — no variant calling in
between.

Lives with this run and dies with its write-up (see the working conventions in
the project README) -- it is not general project tooling.

## Why it exists

`results/experiments/variant_aggregators_20260825/sparse_baseline.py` already
runs an L1-logistic, but on a *variant design matrix*: `variant_design_matrix`
tokenises each isolate into at most 512 (locus, position, base) differences from
H37Rv and fits on those. That is the DISTILLED linear baseline, and it reaches
macro CV 0.878 over the 11 drugs.

This script asks the undistilled version of the same question. The CNNs are fed
the entire alignment — every column, variable or not — and are left to find the
signal themselves. So is this. If a lasso on the raw one-hot alignment reaches
the 0.9246 of `full_run_v2`'s best cell, the convolutional stack is not earning
its parameters.

## The two encodings, and why both are run

They are not the same model even though they carry the same information.

`--encoding onehot` — every alignment column contributes one indicator per base
observed there (A, C, T, G, -, per ``tb.BASE_TO_COLUMN``; anything else, N
included, is all-zero exactly as ``one_hot_nt`` leaves it). Columns that are
CONSTANT across the cohort are dropped. That drop is lossless, not a
convenience: a constant column is collinear with the intercept, sklearn leaves
the intercept unpenalised, so the L1 optimum of a constant column's coefficient
is exactly zero. It is also lossless *per fold* — a column constant within the
training split is likewise collinear with the intercept over the rows being fit
— so building the column set on the full cohort neither sees a label nor
changes a single fitted coefficient. Nothing here needs a `--fit-vocab-on-train`
arm the way the variant baseline did.

`--encoding delta` — the repo's reference coding (``datasets/sequences.py``):
the indicator for the H37Rv base is dropped at each position, so a row is
nonzero only where the isolate differs from the reference. Statistically this is
ordinary dummy coding with H37Rv as the reference level, and it spans the same
column space as `onehot` — but L1 is not invariant to reparametrisation, so the
two arms genuinely can, and are expected to, differ. `delta` penalises "differs
from H37Rv"; `onehot` penalises "has base b", including the wild-type base.

Neither arm standardises. The columns are 0/1 indicators already on a common
scale, and standardising would divide each by its own sqrt(p(1-p)) — inflating
singleton variants by two orders of magnitude relative to common ones, which is
the opposite of what a rare-variant penalty should do.

## What makes it a fair comparison

Identical protocol to `training/multimodal.run_modal_cv`, step for step:

  * missing phenotypes (`y == -1`) dropped BEFORE splitting,
  * held-out test = `train_test_split(test_size=0.2, random_state=42,
    stratify=y)`,
  * `StratifiedKFold(5, shuffle=True, random_state=42)` on the training split,
  * `class_weight="balanced"`, the linear analogue of the inverse-frequency
    alpha the networks train with,
  * the same metrics through the same helpers -- AUC, AUC-PR with RESISTANT as
    the positive class, and `tb.get_threshold_val` for sens/spec,
  * TEST scored once with the model from the best CV fold, as `run_modal_cv`
    does.

The C grid is reported IN FULL rather than tuned silently, as in the variant
baseline: `cv_auc_mean` is the grid's best, and every C's per-fold score and
support size (number of nonzero coefficients) is kept in the JSON, so both the
selection and the sparsity path are inspectable.

## The sweep

Two encodings x two locus universes x 11 drugs = 44 cells, DNA only.

`--locus-set perdrug` is BIG-TB's SD-CNN per-drug map (2-3 loci) — the
locus-matched arm. `--locus-set all` is every curated locus on disk (19), which
is what the joint MD-CNN-style runs see. That axis is the one README finding 2
says actually moves the number, so it is the one the sweep spends its jobs on.

Output matches `run_experiment.py`'s schema (`{DRUG}__{tag}.json` + a
`summary.csv` with the same columns), so the existing notebook builders read
each arm's folder like any other run.

Usage::

    python results/experiments/lasso_wholeseq_20260901/lasso_wholeseq.py \
        --drugs all --encoding onehot --locus-set perdrug \
        --out results/experiments/lasso_wholeseq_20260901/onehot_perdrug
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, tb  # noqa: E402
from datasets.loader import drug_loci, loci_on_disk  # noqa: E402
from datasets.sequences import (BASE_TO_COLUMN, load_phenotype,  # noqa: E402
                                load_sequence_df, numeric_labels,
                                reference_row)

ALL_DRUGS = list(tb.DRUG_TO_LOCI)

# Wider than the variant baseline's grid at the sparse end: this design matrix
# is 10-100x wider than that one, so the useful penalty is stronger and the path
# has to reach it. Reported in full either way.
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)

NT_CODES = list(BASE_TO_COLUMN)                 # ['A','C','T','G','-']
_BASE_LUT = np.full(256, -1, dtype=np.int8)
for _b, _col in BASE_TO_COLUMN.items():
    _BASE_LUT[ord(_b)] = _col


def _metrics(y_true, pred):
    """Byte-identical to training.multimodal._metrics, so the numbers land on
    the same scale as every other cell in results/experiments."""
    m = {"n": int(len(y_true)), "n_R": int((y_true == 0).sum()),
         "n_S": int((y_true == 1).sum())}
    if m["n_R"] == 0 or m["n_S"] == 0:
        return {**m, "auc": float("nan"), "auc_pr": float("nan"),
                "sens": float("nan"), "spec": float("nan")}
    m["auc"] = float(roc_auc_score(y_true, pred))
    m["auc_pr"] = float(average_precision_score(1 - y_true, 1 - pred))
    th = tb.get_threshold_val(y_true, pred)
    m.update(sens=float(th["sens"]), spec=float(th["spec"]),
             threshold=float(th["threshold"]))
    return m


# --- the design matrix ------------------------------------------------------

def _code_matrix(seqs, length):
    """(N, length) int8 of BASE_TO_COLUMN codes; -1 = not in the alphabet.

    Short or absent sequences are right-padded with -1, which is what
    ``one_hot_nt`` + ``stack_padded`` produce for the same isolate (an
    all-zero column), so the padding rule matches the networks' input exactly.
    """
    out = np.full((len(seqs), length), -1, dtype=np.int8)
    for i, s in enumerate(seqs):
        if not s:
            continue
        n = min(len(s), length)
        codes = _BASE_LUT[np.frombuffer(s[:n].encode("ascii", "replace"),
                                        dtype=np.uint8)]
        out[i, :n] = codes
    return out


def _locus_block(seq_dir, locus, seqs, encoding, min_count):
    """One locus -> (csr (N, k) of 0/1 indicators, column names).

    Column name is ``{locus}:{alignment_column}:{base}``, the alignment column
    0-based, so a coefficient is traceable back to a position in the FASTA.
    """
    length = max((len(s) for s in seqs), default=0)
    if length == 0:
        return sp.csr_matrix((len(seqs), 0), dtype=np.float32), []
    codes = _code_matrix(seqs, length)
    n = codes.shape[0]

    ref = reference_row(seq_dir, locus) if encoding == "delta" else None
    if encoding == "delta" and not ref:
        print(f"  [build] {locus}: no H37Rv row — that locus stays plain "
              f"one-hot", flush=True)
    ref_codes = np.full(length, -1, dtype=np.int8)
    if ref:
        m = min(len(ref), length)
        ref_codes[:m] = _BASE_LUT[np.frombuffer(ref[:m].encode("ascii", "replace"),
                                                dtype=np.uint8)]

    # keep/drop decision, one pass over the 5 bases
    colmap = np.full((length, len(NT_CODES)), -1, dtype=np.int32)
    names, k = [], 0
    for c in range(len(NT_CODES)):
        cnt = (codes == c).sum(axis=0)
        # constant columns carry no signal: all-zero is empty, all-one is
        # collinear with the unpenalised intercept and its L1 optimum is 0.
        keep = (cnt >= max(min_count, 1)) & (cnt < n)
        if encoding == "delta":
            # reference coding: drop the H37Rv base's indicator. Positions past
            # the end of the reference keep every base, matching
            # delta_one_hot_nt, which cannot compare there.
            keep &= ~((ref_codes == c) & (ref_codes >= 0))
        for j in np.nonzero(keep)[0]:
            colmap[j, c] = k
            names.append(f"{locus}:{int(j)}:{NT_CODES[c]}")
            k += 1
    if k == 0:
        return sp.csr_matrix((n, 0), dtype=np.float32), []

    # gather: every (isolate, position) lands in at most one column
    cols = np.where(codes >= 0, colmap[np.arange(length)[None, :], codes], -1)
    rows = np.repeat(np.arange(n, dtype=np.int32), length)
    cols = cols.ravel()
    hit = cols >= 0
    X = sp.csr_matrix(
        (np.ones(int(hit.sum()), dtype=np.float32), (rows[hit], cols[hit])),
        shape=(n, k))
    return X, names


def build_design(genotype_dir, loci, isolates, seq_df, encoding, min_count):
    """The whole-sequence design matrix: every locus's alignment, side by side."""
    blocks, names = [], []
    for locus in loci:
        seqs = seq_df[locus].fillna("").tolist()
        Xb, nb = _locus_block(genotype_dir, locus, seqs, encoding, min_count)
        print(f"  [build] {locus}: {Xb.shape[1]} columns, "
              f"{Xb.nnz / max(len(isolates), 1):.0f} nnz per isolate", flush=True)
        blocks.append(Xb)
        names.extend(nb)
    X = sp.hstack(blocks, format="csr", dtype=np.float32) if blocks else None
    return X, names


def _fit(C, X_tr, y_tr, solver, max_iter):
    """P(y == 1), i.e. P(susceptible) -- the same quantity the networks' sigmoid
    outputs, so AUC and the threshold search agree in sign."""
    # l1_ratio=1.0 IS penalty="l1"; the keyword form is deprecated as of
    # sklearn 1.8 and removed in 1.10, and this env is on 1.9.
    clf = LogisticRegression(solver=solver, l1_ratio=1.0, C=C,
                             class_weight="balanced", max_iter=max_iter,
                             tol=1e-4)
    clf.fit(X_tr, y_tr)
    return clf


def run_drug(drug, args):
    t0 = time.time()
    genotype_dir = args.genotype_dir
    loci = (loci_on_disk(genotype_dir) if args.locus_set == "all"
            else drug_loci(drug))

    # isolate axis, derived exactly as datasets/loader.py derives it: the FASTA
    # records joined across loci, intersected with the phenotype index.
    seq_df, found = load_sequence_df(genotype_dir, loci)
    missing = [g for g in loci if g not in found]
    if missing:
        print(f"  [load] {drug}: loci not found on disk, skipped: {missing}",
              flush=True)
    df_phenos = load_phenotype(args.phenotype_csv)
    isolates = list(seq_df.index.intersection(df_phenos.index))
    y_all = numeric_labels(df_phenos.loc[isolates], drug)

    # Drop missing phenotypes BEFORE building the matrix, not after. Same rows
    # as run_modal_cv either way, but it also makes the constant-column rule
    # exact: a column is dropped only if it is constant over the cohort that is
    # actually fit, which is the set the collinearity-with-intercept argument in
    # the module docstring is about.
    keep = np.nonzero(y_all != -1)[0]
    y = y_all[keep]
    isolates = [isolates[i] for i in keep]
    seq_df = seq_df.reindex(isolates)

    X, names = build_design(genotype_dir, found, isolates, seq_df,
                            args.encoding, args.min_count)
    del seq_df
    n = X.shape[0]
    print(f"[{drug}] {n} isolates, {X.shape[1]} columns, "
          f"{X.nnz / max(n, 1):.0f} nnz per isolate, "
          f"loci {'+'.join(found)}", flush=True)

    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2,
                                           random_state=42, stratify=y)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(train_idx, y[train_idx]))

    # --- the C path ---------------------------------------------------------
    path, best = [], None
    for C in args.c_grid:
        fold_auc, fold_models, nnz_coef = [], [], []
        for tr, va in folds:
            i_tr, i_va = train_idx[tr], train_idx[va]
            tc = time.time()
            clf = _fit(C, X[i_tr], y[i_tr], args.solver, args.max_iter)
            pred = clf.predict_proba(X[i_va])[:, 1]
            m = _metrics(y[i_va], pred)
            fold_auc.append(m["auc"])
            fold_models.append(clf)
            nnz_coef.append(int((clf.coef_ != 0).sum()))
            print(f"  [{drug}] C={C:<6g} fold {len(fold_auc)}/{len(folds)} "
                  f"auc={m['auc']:.4f} support={nnz_coef[-1]} "
                  f"({time.time() - tc:.1f}s)", flush=True)
        entry = {"C": C, "cv_auc_mean": float(np.mean(fold_auc)),
                 "cv_auc_std": float(np.std(fold_auc)),
                 "fold_aucs": [float(a) for a in fold_auc],
                 "support_mean": float(np.mean(nnz_coef)),
                 "support_per_fold": nnz_coef}
        path.append(entry)
        if best is None or entry["cv_auc_mean"] > best[0]["cv_auc_mean"]:
            best = (entry, fold_models, fold_auc)

    entry, fold_models, fold_auc = best
    # TEST from the best CV fold's model, as run_modal_cv does.
    best_fold = int(np.argmax(fold_auc))
    test_pred = fold_models[best_fold].predict_proba(X[test_idx])[:, 1]
    test = _metrics(y[test_idx], test_pred)

    # the largest-magnitude coefficients of that model, named — the readable
    # output a lasso has and a CNN does not.
    coef = fold_models[best_fold].coef_.ravel()
    nz = np.nonzero(coef)[0]
    order = nz[np.argsort(-np.abs(coef[nz]))][:args.top_k]
    top = [{"feature": names[i], "coef": float(coef[i])} for i in order]

    cv_pr = float(np.mean([
        _metrics(y[train_idx[va]],
                 fold_models[i].predict_proba(X[train_idx[va]])[:, 1])["auc_pr"]
        for i, (tr, va) in enumerate(folds)]))

    return {
        "drug": drug, "arch": "lasso_wholeseq", "modalities": ["dna"],
        "tag": f"dna-{args.encoding}-{args.locus_set}",
        "encoding": args.encoding, "locus_set": args.locus_set,
        "solver": args.solver, "min_count": args.min_count,
        "genes": found, "n_valid": int(n),
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


def _summary_row(r):
    return {
        "drug": r["drug"], "modalities": r["tag"],
        "genes": "+".join(r["genes"]), "n_valid": r["n_valid"],
        "n_R": r["n_resistant"], "n_S": r["n_susceptible"],
        "cv_auc_mean": r["cv_auc_mean"], "cv_auc_std": r["cv_auc_std"],
        "cv_auc_pr_mean": r["cv_auc_pr_mean"],
        "test_auc": r["test"]["auc"], "test_auc_pr": r["test"]["auc_pr"],
        "test_sens": r["test"]["sens"], "test_spec": r["test"]["spec"],
        "best_C": r["best_C"], "n_features": r["n_features"],
        "cv_support_mean": r["cv_support_mean"],
        "seconds": r["seconds"],
    }


def _write_summary(run_dir):
    rows = []
    for jf in sorted(run_dir.glob("*.json")):
        r = json.loads(jf.read_text())
        if "cv_auc_mean" in r:
            rows.append(_summary_row(r))
    if rows:
        pd.DataFrame(rows).round(4).to_csv(run_dir / "summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drugs", nargs="+", default=["all"])
    ap.add_argument("--encoding", choices=("onehot", "delta"), default="onehot")
    ap.add_argument("--locus-set", choices=("perdrug", "all"), default="perdrug")
    ap.add_argument("--solver", choices=("liblinear", "saga"), default="liblinear")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--min-count", type=int, default=1,
                    help="drop indicator columns supported by fewer than this "
                         "many isolates; 1 = keep every non-constant column, "
                         "which is the honest whole-sequence input")
    ap.add_argument("--c-grid", type=float, nargs="+", default=list(C_GRID))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--genotype-dir", default=REAL_GENOTYPE_DIR)
    ap.add_argument("--phenotype-csv", default=REAL_PHENOTYPE_CSV)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    drugs = (ALL_DRUGS if any(d.lower() == "all" for d in args.drugs)
             else [d.upper() for d in args.drugs])
    unknown = [d for d in drugs if d not in ALL_DRUGS]
    if unknown:
        ap.error(f"unknown drug(s) {unknown}; choose from {ALL_DRUGS} or 'all'")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for drug in drugs:
        r = run_drug(drug, args)
        (out / f"{drug}__{r['tag']}.json").write_text(json.dumps(r, indent=2))
        print(f"[{drug}] CV {r['cv_auc_mean']:.4f} +/- {r['cv_auc_std']:.4f} "
              f"@ C={r['best_C']} (support {r['cv_support_mean']:.0f} of "
              f"{r['n_features']}), TEST {r['test']['auc']:.4f} "
              f"[{r['seconds']:.0f}s]", flush=True)
        _write_summary(out)


if __name__ == "__main__":
    main()

"""
LASSO baseline: L1-logistic regression on the whole aligned sequence, one-hot.

Every alignment column contributes one 0/1 indicator per base observed there
(A, C, T, G, -; anything else, N included, is all-zero, as in the networks'
one-hot). Columns constant across the fitted cohort are dropped: a constant
column is collinear with the unpenalised intercept, so its L1 optimum is 0.
Columns are not standardised (standardising would inflate rare variants).

Protocol matches the networks (training/multimodal.run_modal_cv):
  * missing phenotypes dropped before splitting,
  * the lineage split from --split-dir (or a stratified 80/20 split, seed 42,
    then StratifiedKFold(5, seed 42) on train),
  * class_weight="balanced" (the linear analogue of the networks' alpha weights),
  * the same metrics (AUC; AUC-PR with resistant as positive; tb.get_threshold_val),
  * TEST scored once with the best CV fold's model.
The whole C grid is kept in the JSON (`c_path`); cv_auc_mean is the best C's.

Locus sets (--locus-set):
  perdrug                          the drug's Big-TB loci
  all                              the 19 curated loci (datasets.loader.LOCI_19)
  lab64 | who_t12_sai | who_t12      datasets.loader.LOCUS_SETS (who_t12 needs
                                   4 genes not in Sai's 64; use who_t12_sai)

    python -m lasso.lasso --drugs PYRAZINAMIDE --locus-set all \
        --split-dir results/splits/heldout_2 --out results/lasso/onehot_all_L2
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import bigtb as tb
from datasets.loader import LOCI_19, LOCUS_SETS, drug_loci
from datasets.sequences import (BASE_TO_COLUMN, load_phenotype, load_sequence_df,
                                numeric_labels)
from paths import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV

ALL_DRUGS = list(tb.DRUG_TO_LOCI)
LOCUS_SET_CHOICES = ("perdrug", "all") + tuple(s for s in LOCUS_SETS if s != "bigtb")

C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)

NT_CODES = list(BASE_TO_COLUMN)                 # ['A','C','T','G','-']
_BASE_LUT = np.full(256, -1, dtype=np.int8)
for _b, _col in BASE_TO_COLUMN.items():
    _BASE_LUT[ord(_b)] = _col


def _metrics(y_true, pred):
    """training.multimodal._metrics on already-filtered labels."""
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


def _locus_block(locus, seqs):
    """One locus -> (csr (N, k) of 0/1 indicators, column names).

    Column name is ``{locus}:{alignment_column}:{base}``, the alignment column
    0-based, so a coefficient is traceable back to a position in the FASTA.
    """
    length = max((len(s) for s in seqs), default=0)
    if length == 0:
        return sp.csr_matrix((len(seqs), 0), dtype=np.float32), []
    codes = _code_matrix(seqs, length)
    n = codes.shape[0]

    # keep/drop decision, one pass over the 5 bases
    colmap = np.full((length, len(NT_CODES)), -1, dtype=np.int32)
    names, k = [], 0
    for c in range(len(NT_CODES)):
        cnt = (codes == c).sum(axis=0)
        # constant columns carry no signal: all-zero is empty, all-one is
        # collinear with the unpenalised intercept and its L1 optimum is 0.
        keep = (cnt >= 1) & (cnt < n)
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


def build_design(loci, isolates, seq_df):
    """The whole-sequence design matrix: every locus's alignment, side by side."""
    blocks, names = [], []
    for locus in loci:
        seqs = seq_df[locus].fillna("").tolist()
        Xb, nb = _locus_block(locus, seqs)
        print(f"  [build] {locus}: {Xb.shape[1]} columns, "
              f"{Xb.nnz / max(len(isolates), 1):.0f} nnz per isolate", flush=True)
        blocks.append(Xb)
        names.extend(nb)
    X = sp.hstack(blocks, format="csr", dtype=np.float32) if blocks else None
    return X, names


def _fit(C, X_tr, y_tr, max_iter):
    """P(y == 1), i.e. P(susceptible) -- the same quantity the networks' sigmoid
    outputs, so AUC and the threshold search agree in sign."""
    # l1_ratio=1.0 IS penalty="l1"; the keyword form is deprecated as of
    # sklearn 1.8 and removed in 1.10, and this env is on 1.9.
    clf = LogisticRegression(solver="liblinear", l1_ratio=1.0, C=C,
                             class_weight="balanced", max_iter=max_iter,
                             tol=1e-4)
    clf.fit(X_tr, y_tr)
    return clf


def run_drug(drug, args):
    t0 = time.time()
    genotype_dir = args.genotype_dir
    loci = (list(LOCI_19) if args.locus_set == "all"
            else drug_loci(drug) if args.locus_set == "perdrug"
            else drug_loci(drug, locus_set=args.locus_set))

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

    X, names = build_design(found, isolates, seq_df)
    del seq_df
    n = X.shape[0]
    print(f"[{drug}] {n} isolates, {X.shape[1]} columns, "
          f"{X.nnz / max(n, 1):.0f} nnz per isolate, "
          f"loci {'+'.join(found)}", flush=True)

    if args.split_dir is None:
        train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2,
                                               random_state=42, stratify=y)
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        folds = list(skf.split(train_idx, y[train_idx]))
    else:   # held-out-lineage split from splits/build_holdout_splits.py
        sp = (pd.read_csv(Path(args.split_dir) / f"{drug}.csv", dtype={"New_ID": str})
              .set_index("New_ID").loc[[str(i) for i in isolates]])
        test_idx = np.nonzero((sp.role == "test").values)[0]
        train_idx = np.nonzero((sp.role == "train").values)[0]
        fold_of = sp.fold.values[train_idx]
        folds = [(np.nonzero(fold_of != k)[0], np.nonzero(fold_of == k)[0])
                 for k in sorted(set(fold_of))]
        assert len(folds) == args.n_splits

    # --- the C path ---------------------------------------------------------
    path, best = [], None
    for C in args.c_grid:
        fold_auc, fold_models, nnz_coef = [], [], []
        for tr, va in folds:
            i_tr, i_va = train_idx[tr], train_idx[va]
            tc = time.time()
            clf = _fit(C, X[i_tr], y[i_tr], args.max_iter)
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
        "tag": f"dna-onehot-{args.locus_set}",
        "encoding": "onehot", "locus_set": args.locus_set,
        "solver": "liblinear", "min_count": 1, "genotype_dir": str(genotype_dir),
        "genes": found, "n_valid": int(n),
        "n_resistant": int((y == 0).sum()), "n_susceptible": int((y == 1).sum()),
        "n_features": int(X.shape[1]),
        "nnz_per_isolate": float(X.nnz / max(n, 1)),
        "n_params": int(X.shape[1] + 1),
        "split": {"test_size": 0.2, "split_seed": 42, "stratified": True,
                  "n_splits": args.n_splits, "kfold": "StratifiedKFold",
                  "kfold_seed": 42, "kfold_shuffle": True,
                  "split_dir": args.split_dir},
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
    ap.add_argument("--locus-set", choices=LOCUS_SET_CHOICES, default="perdrug")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--c-grid", type=float, nargs="+", default=list(C_GRID))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--genotype-dir", default=REAL_GENOTYPE_DIR)
    ap.add_argument("--phenotype-csv", default=REAL_PHENOTYPE_CSV)
    ap.add_argument("--split-dir", default=None,
                    help="per-drug {DRUG}.csv split files (New_ID, role, fold); "
                         "overrides the stratified split")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    drugs = (ALL_DRUGS if any(d.lower() == "all" for d in args.drugs)
             else [d.upper() for d in args.drugs])
    unknown = [d for d in drugs if d not in ALL_DRUGS]
    if unknown:
        ap.error(f"unknown drug(s) {unknown}; choose from {ALL_DRUGS} or 'all'")
    if args.locus_set not in ("perdrug", "all"):
        for drug in drugs:
            absent = [g for g in drug_loci(drug, locus_set=args.locus_set)
                      if not sorted(Path(args.genotype_dir).glob(f"{g}*.fasta"))]
            if absent:
                ap.error(f"{drug}/{args.locus_set}: no FASTA in {args.genotype_dir} "
                         f"for {absent}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for drug in drugs:
        if args.split_dir and not (Path(args.split_dir) / f"{drug}.csv").is_file():
            print(f"[{drug}] skipped: no split file in {args.split_dir}", flush=True)
            continue
        r = run_drug(drug, args)
        (out / f"{drug}__{r['tag']}.json").write_text(json.dumps(r, indent=2))
        print(f"[{drug}] CV {r['cv_auc_mean']:.4f} +/- {r['cv_auc_std']:.4f} "
              f"@ C={r['best_C']} (support {r['cv_support_mean']:.0f} of "
              f"{r['n_features']}), TEST {r['test']['auc']:.4f} "
              f"[{r['seconds']:.0f}s]", flush=True)
        _write_summary(out)


if __name__ == "__main__":
    main()

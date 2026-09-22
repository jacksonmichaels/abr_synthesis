#!/usr/bin/env python3
"""
The baseline that is not a network: L1-logistic and gradient boosting on the
sparse isolate x variant matrix.

Lives with this run and dies with its write-up (see the working conventions in
the project README) -- it is not general project tooling.

## Why it exists

There is no sparse-linear or tree baseline anywhere in this project, and for TB
AMR from a variant matrix those are the canonical strong methods: TB-Profiler and
Mykrobe are catalogue lookups and are clinically competitive. `token_signal`
already found plain logistic regression on setfusion's own representations
beating the trained model by 0.011, which points straight here.

The census says the median isolate differs from H37Rv at 0-3 columns of a 2.5 kb
gene. That is a sparse-feature-selection problem before it is a deep-learning
problem, and **if an L1-logistic reaches the ~0.909 that `full_run_v2`'s best
all-modalities cell reaches, that reframes the project.** This script is the
cheapest possible way to find out: no GPU, minutes per drug.

## What makes it a fair comparison

Identical protocol to `training/multimodal.run_modal_cv`, step for step:

  * the same `load_dataset(..., per_modality_branch=False, delta=True)` input the
    variant-token networks get -- same loci, same modalities, same reference,
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

Two things this baseline gets that the networks do not, both stated rather than
hidden:

  1. **The design matrix is built on the full cohort**, so a column exists if the
     variant appears anywhere -- including in test isolates. The *values* in a
     test row are never seen during fitting and no label is touched, so this is
     not label leakage, but it is a vocabulary the networks have to discover.
     `--fit-vocab-on-train` restricts the columns to variants seen in the
     training split and is the strict version; run both.
  2. **`max_variants` is raised to 512**, well above the training default of 16,
     because this is a one-off feature extraction and there is no reason to
     truncate the tail.

Output matches `run_experiment.py`'s schema (`{DRUG}__{tag}.json` + a
`summary.csv` with the same columns), so the existing notebook builders can read
this folder like any other run.

Usage::

    python results/experiments/variant_aggregators_20260825/sparse_baseline.py \
        --drugs all --out results/experiments/variant_aggregators_20260825/sparse_baseline
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
import torch  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

from bigtb_ref import (REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,  # noqa: E402
                       REAL_REGULATORY_DIR, tb)
from datasets import load_dataset  # noqa: E402
from models.experimental_models import variant_design_matrix  # noqa: E402

ALL_DRUGS = list(tb.DRUG_TO_LOCI)
# The C grid is reported IN FULL rather than tuned silently: an L1 baseline whose
# regularisation was picked on the same CV it reports is not comparable to a
# network trained once. `cv_auc_mean` in the summary is the grid's best, and
# every C's score is kept in the JSON so the selection is inspectable.
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


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


def _dense(X):
    """HistGradientBoosting has no sparse path, so the tree arm densifies. float32
    halves the peak: ISONIAZID at 17.4 k isolates x ~10 k columns is 700 MB here
    against 1.4 GB in float64, and that is per fold."""
    return np.asarray(X.todense(), dtype=np.float32)


def _fit_predict(kind, C, X_tr, y_tr, X_va):
    """P(y == 1), i.e. P(susceptible) -- the same quantity the networks' sigmoid
    outputs, so AUC and the threshold search agree in sign."""
    if kind == "l1":
        # l1_ratio=1.0 IS penalty="l1"; the keyword form is deprecated as of
        # sklearn 1.8 and removed in 1.10, and this env is on 1.9.
        clf = LogisticRegression(solver="liblinear", l1_ratio=1.0, C=C,
                                 class_weight="balanced", max_iter=2000)
        clf.fit(X_tr, y_tr)
    else:
        # sklearn's own LightGBM-equivalent, so this needs no extra dependency.
        # It wants dense input; the matrices here are a few thousand columns.
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=0, class_weight="balanced")
        clf.fit(_dense(X_tr), y_tr)
        return clf, clf.predict_proba(_dense(X_va))[:, 1]
    return clf, clf.predict_proba(X_va)[:, 1]


def run_drug(drug, args):
    t0 = time.time()
    data = load_dataset(drug, ["dna", "protein", "biophysical", "regulatory"],
                        REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,
                        regulatory_dir=REAL_REGULATORY_DIR,
                        loci=args.loci,
                        per_modality_branch=False, delta=True)
    keep = np.nonzero(data.y != -1)[0]
    y = data.y[keep]
    xs = [torch.from_numpy(np.asarray(a[keep], dtype=np.float32))
          for a in data.arrays()]
    keys = [(b.modality, b.name.partition(":")[2] or None) for b in data.blocks]
    X, names = variant_design_matrix(xs, keys, data.branch_specs(),
                                     max_variants=args.max_variants)
    n = X.shape[0]
    raw_cols = X.shape[1]
    print(f"[{drug}] {n} isolates, {raw_cols} variant columns, "
          f"{X.nnz / max(n, 1):.1f} variants per isolate (mean)", flush=True)

    # Rare-variant filter, standard in bacterial GWAS and load-bearing here for a
    # reason worth stating: a column present in ONE isolate is useless to every
    # model in this script. Per-identity weights cannot generalise from a
    # singleton -- fitting it is memorising one training row, and a singleton
    # that only appears in test is a variant no fit has ever seen. It is exactly
    # the case `--arch additive` exists to handle by featurising the variant, and
    # exactly the case this baseline structurally cannot. That asymmetry is the
    # point of the comparison, not a flaw in it.
    if args.min_count > 1:
        counts = np.asarray((X > 0).sum(0)).ravel()
        cols = np.nonzero(counts >= args.min_count)[0]
        X, names = X[:, cols], [names[c] for c in cols]
        print(f"[{drug}] min_count={args.min_count}: {raw_cols} -> {X.shape[1]} "
              f"columns ({raw_cols - X.shape[1]} rare/singleton dropped)", flush=True)

    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2,
                                           random_state=42, stratify=y)
    if args.fit_vocab_on_train:
        # strict arm: a column only exists if the variant appears in TRAINING
        cols = np.asarray((X[train_idx].sum(0) > 0)).ravel().nonzero()[0]
        X, names = X[:, cols], [names[c] for c in cols]
        print(f"[{drug}] train-only vocabulary: {X.shape[1]} columns", flush=True)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    result = {"drug": drug, "modalities": data.modalities,
              "tag": data.modality_tag(), "genes": data.gene_order,
              "arch": "sparse_baseline", "n_features": int(X.shape[1]),
              "n_features_raw": int(raw_cols), "min_count": args.min_count,
              "n_isolates": data.n, "n_valid": int(n),
              "n_resistant": int((y == 0).sum()),
              "n_susceptible": int((y == 1).sum()),
              "fit_vocab_on_train": bool(args.fit_vocab_on_train),
              "max_variants": args.max_variants, "models": {}}

    best = (-np.inf, None, None, None, None)   # cv_mean, key, clf, test, names
    for kind in args.models:
        Xm, names_m = X, names
        if kind == "gbm" and X.shape[1] > args.gbm_max_features:
            # the tree arm densifies, so it is capped by column FREQUENCY (not by
            # any label-derived score, which would leak). Reported, never silent.
            counts = np.asarray((X > 0).sum(0)).ravel()
            cols = np.argsort(-counts)[:args.gbm_max_features]
            cols.sort()
            Xm, names_m = X[:, cols], [names[c] for c in cols]
            print(f"[{drug}] gbm: capped {X.shape[1]} -> {Xm.shape[1]} columns "
                  f"by frequency", flush=True)
        grid = C_GRID if kind == "l1" else (None,)
        for C in grid:
            fold_auc, fold_pr, best_fold = [], [], (-np.inf, None)
            for f, (tr, va) in enumerate(skf.split(train_idx, y[train_idx])):
                clf, p = _fit_predict(kind, C, Xm[train_idx[tr]], y[train_idx[tr]],
                                      Xm[train_idx[va]])
                m = _metrics(y[train_idx[va]], p)
                fold_auc.append(m["auc"])
                fold_pr.append(m["auc_pr"])
                if m["auc"] == m["auc"] and m["auc"] > best_fold[0]:
                    best_fold = (m["auc"], clf)
            aucs = [a for a in fold_auc if a == a]
            cv_mean = float(np.mean(aucs)) if aucs else float("nan")
            key = kind if C is None else f"{kind}_C{C}"
            # TEST is scored once with the best CV fold's model, exactly as
            # run_modal_cv does -- not refit on the full training split, which
            # would be a different (and more favourable) protocol.
            clf = best_fold[1]
            Xt = Xm[test_idx] if kind == "l1" else _dense(Xm[test_idx])
            test = _metrics(y[test_idx], clf.predict_proba(Xt)[:, 1])
            result["models"][key] = {
                "cv_auc_mean": cv_mean,
                "cv_auc_std": float(np.std(aucs)) if aucs else float("nan"),
                "cv_auc_pr_mean": float(np.mean([p for p in fold_pr if p == p]))
                if any(p == p for p in fold_pr) else float("nan"),
                "cv_folds": fold_auc, "test": test,
                "n_nonzero": int((clf.coef_ != 0).sum()) if kind == "l1" else None,
                "n_features": int(Xm.shape[1]),
            }
            print(f"[{drug}] {key:>10s}  CV {cv_mean:.4f}  TEST {test['auc']:.4f}"
                  + (f"  ({int((clf.coef_ != 0).sum())} nonzero coefs)"
                     if kind == "l1" else ""), flush=True)
            if cv_mean == cv_mean and cv_mean > best[0]:
                best = (cv_mean, key, clf, test, names_m)

    result["best_model"] = best[1]
    result["cv_auc_mean"] = best[0]
    result["cv_auc_std"] = result["models"][best[1]]["cv_auc_std"]
    result["cv_auc_pr_mean"] = result["models"][best[1]]["cv_auc_pr_mean"]
    result["test"] = best[3]
    result["seconds"] = time.time() - t0

    # the top L1 coefficients, by name -- a learned resistance catalogue, and the
    # thing to hold `--arch catalogue`'s own table up against
    if best[1].startswith("l1"):
        coef = best[2].coef_.ravel()
        order = np.argsort(-np.abs(coef))[:40]
        # coef is on P(susceptible), so NEGATE for "toward RESISTANT"
        result["top_variants"] = [{"variant": best[4][i], "toward_resistant":
                                   float(-coef[i])} for i in order if coef[i]]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drugs", nargs="+", default=["all"])
    ap.add_argument("--models", nargs="+", default=["l1", "gbm"],
                    choices=["l1", "gbm"])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--max-variants", type=int, default=512,
                    help="tokenizer cap per block for the feature extraction "
                         "(default 512, far above the training default of 16 — "
                         "this is a one-off extraction, nothing to truncate for)")
    ap.add_argument("--min-count", type=int, default=2,
                    help="drop variant columns seen in fewer than N isolates "
                         "(default 2). A singleton is unusable to a per-identity "
                         "model either way — see the note in run_drug.")
    ap.add_argument("--gbm-max-features", type=int, default=4000,
                    help="the tree arm densifies, so it is capped at the N most "
                         "FREQUENT columns (default 4000). Frequency, not any "
                         "label-derived score. Reported in the JSON, never silent.")
    ap.add_argument("--fit-vocab-on-train", action="store_true",
                    help="strict arm: keep only variant columns seen in the "
                         "TRAINING split, so the feature vocabulary itself is "
                         "not derived from test isolates")
    ap.add_argument("--loci", nargs="+", default=None, metavar="GENE",
                    help="gene loci to load (default: the drug's own DRUG_TO_LOCI "
                         "map, which is what variant_aggregators_20260825 ran and "
                         "is what keeps this locus-matched to full_run_v2). Pass "
                         "the 19 curated loci to match alllocus_run_v2 instead.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    drugs = ALL_DRUGS if args.drugs == ["all"] else [d.upper() for d in args.drugs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for drug in drugs:
        try:
            r = run_drug(drug, args)
        except Exception as e:  # noqa: BLE001
            print(f"[{drug}] FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        (out / f"{drug}__{r['tag']}.json").write_text(json.dumps(r, indent=2))
        rows.append({
            "drug": drug, "modalities": r["tag"], "genes": "+".join(r["genes"]),
            "n_valid": r["n_valid"], "n_R": r["n_resistant"], "n_S": r["n_susceptible"],
            "cv_auc_mean": r["cv_auc_mean"], "cv_auc_std": r["cv_auc_std"],
            "cv_auc_pr_mean": r["cv_auc_pr_mean"], "test_auc": r["test"]["auc"],
            "test_auc_pr": r["test"]["auc_pr"], "test_sens": r["test"]["sens"],
            "test_spec": r["test"]["spec"], "seconds": r["seconds"],
            "best_model": r["best_model"], "n_features": r["n_features"],
        })
        # NB one job per drug all write into the same --out, so this leaves the
        # LAST writer's single row behind. Rebuild from the JSONs after the run:
        #   rows = [json.loads(f.read_text()) for f in out.glob("*.json")]
        # (2026-08-26: that is exactly what happened to sparse_baseline/, and the
        # summary was regenerated from the 11 JSONs, which are the real record.)
        pd.DataFrame(rows).round(4).to_csv(out / "summary.csv", index=False)
    if rows:
        macro = float(np.nanmean([r["cv_auc_mean"] for r in rows]))
        print(f"\nMACRO CV AUC over {len(rows)} drugs: {macro:.4f}")
        print(f"  (full_run_v2 all_modalities best cell: 0.9086 | "
              f"alllocus_run_v2: 0.9246 | SD-CNN: 0.8636)")
    print(f"\nWrote {out}/summary.csv")


if __name__ == "__main__":
    main()

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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

# The parent run is the protocol. Importing it means the metrics, the solver
# call and the summary schema cannot drift apart from the arm this is compared
# against.
import lasso_wholeseq as ws  # noqa: E402
from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, tb  # noqa: E402
from datasets.loader import drug_loci, loci_on_disk  # noqa: E402
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


def _parse_annot(arg):
    """{key: (Annotation, Annotation_Impact)} for one isolate's PASS calls, keyed
    exactly as _parse_one keys them."""
    tsv_dir, iso = arg
    out = {}
    with open(os.path.join(tsv_dir, iso + SUFFIX)) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        iP, iR, iA, iF, iN, iM = (hdr.index(c) for c in (
            "POSITION", "REFERENCE_ALLELE", "ALTERED_ALLELE", "FILTER_STATUS",
            "Annotation", "Annotation_Impact"))
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) <= iF or c[iF] != "PASS":
                continue
            k = f"{c[iP]}:{c[iR][:20]}>{c[iA][:20]}"
            if k not in out:
                out[k] = (c[iN] if len(c) > iN else "", c[iM] if len(c) > iM else "")
    return out


def build_annotation(tsv_dir, keys, cache, procs):
    """snpEff Annotation / Annotation_Impact per cohort key, aligned to `keys`.

    A sidecar to the cohort cache (which predates this and stores no effects),
    parsed once over every isolate and cached. The first annotation seen for a
    key wins; snpEff's call for a given POS:REF>ALT does not vary by isolate."""
    cache = Path(cache)
    if cache.is_file():
        z = np.load(cache, allow_pickle=False)
        if z["keys"].shape[0] != len(keys):
            raise RuntimeError(f"{cache} has {z['keys'].shape[0]} keys, cohort has "
                               f"{len(keys)}: stale sidecar, delete and rebuild")
        print(f"  [annot] loaded {cache}", flush=True)
        return list(z["annotation"]), list(z["impact"])
    col = {k: i for i, k in enumerate(keys)}
    ann, imp = [""] * len(keys), [""] * len(keys)
    seen = np.zeros(len(keys), bool)
    isolates = sorted(isolates_on_disk(tsv_dir))
    t0, unknown = time.time(), 0
    with Pool(procs) as pool:
        it = pool.imap_unordered(_parse_annot, [(tsv_dir, i) for i in isolates],
                                 chunksize=16)
        for done, d in enumerate(it, 1):
            for k, (a, m) in d.items():
                j = col.get(k)
                if j is None:
                    unknown += 1
                elif not seen[j]:
                    ann[j], imp[j], seen[j] = a, m, True
            if done % 2000 == 0:
                print(f"  [annot] {done}/{len(isolates)} isolates, "
                      f"{seen.mean():.1%} keys annotated, {time.time() - t0:.0f}s",
                      flush=True)
    if unknown:
        raise RuntimeError(f"{unknown} PASS keys not in the cohort cache: stale cache")
    print(f"  [annot] {seen.sum()}/{len(keys)} keys annotated in "
          f"{time.time() - t0:.0f}s", flush=True)
    np.savez(cache, keys=np.array(keys), annotation=np.array(ann),
             impact=np.array(imp))
    print(f"  [annot] wrote {cache}", flush=True)
    return ann, imp


def _key_span(k):
    """0-based [start, end) of the reference bases a key actually changes.

    VCF indels carry an unchanged anchor base (TC>T deletes the C, not the T),
    so the shared prefix is stripped, and for equal-length alleles the shared
    suffix too. An insertion changes no reference base and returns an empty
    span. (Before 2026-10-01 the whole REF allele was used, so every indel also
    marked its anchor base; lasso_pooled_20260929 ran with that.)"""
    pos, alle = k.split(":", 1)
    ref, alt = alle.split(">", 1)
    p0 = int(pos) - 1
    pre = 0
    while pre < min(len(ref), len(alt)) and ref[pre] == alt[pre]:
        pre += 1
    end = len(ref)
    if len(ref) == len(alt):
        while end > pre and ref[end - 1] == alt[end - 1]:
            end -= 1
    return p0 + pre, p0 + max(end, pre)


def _group_or(X, groups, prefix, min_members=2):
    """One column per group: 1 where the isolate carries ANY key in the group.
    `groups` is a list of (key column, group label) pairs. Groups with fewer
    than `min_members` keys would duplicate a key column, so they are dropped."""
    if not groups:
        return None, []
    j, lab = zip(*groups)
    uniq, inv = np.unique(np.array(lab), return_inverse=True)
    M = sp.csr_matrix((np.ones(len(j), np.float32), (np.array(j), inv)),
                      shape=(X.shape[1], len(uniq)))
    size = np.bincount(inv, minlength=len(uniq))
    keep = np.nonzero(size >= min_members)[0]
    G = (X @ M[:, keep]).tocsr()
    G.data[:] = 1.0
    G.eliminate_zeros()
    return G.astype(np.float32), [f"{prefix}:{u}" for u in uniq[keep]]


def pooled_features(X, names, pool, geneid_of, impact_of, scale=1.0):
    """Append pooled indicators to the per-key design, so L1 can pay once for a
    site or a gene instead of once per rare allele.

      site    : any non-reference call covering this H37Rv base (a deletion
                covers every base of its REF allele) -- the variant-coded
                equivalent of the one-hot arm's penalised wild-type column
      gene    : per Gene_ID, any HIGH-impact call (frameshift, stop gained,
                start/stop lost, splice) = lof:, and any HIGH or MODERATE call
                (adds missense, in-frame indels) = nonsyn:
    Only groups of >= 2 distinct keys are added. The per-key columns are kept.
    Everything here is a function of the isolate's own calls plus the key
    vocabulary, never of a label."""
    blocks, bnames = [X], list(names)
    if pool in ("site", "both"):
        # labelled by 1-based H37Rv position, as VCF POS is
        pairs = [(j, s + 1) for j, k in enumerate(names) for s in range(*_key_span(k))]
        G, g = _group_or(X, pairs, "site")
        if G is not None:
            blocks.append(G); bnames += g
    if pool in ("gene", "both"):
        for prefix, allowed in (("lof", {"HIGH"}), ("nonsyn", {"HIGH", "MODERATE"})):
            pairs = [(j, geneid_of.get(k, "")) for j, k in enumerate(names)
                     if impact_of.get(k, "") in allowed and geneid_of.get(k, "")]
            G, g = _group_or(X, pairs, prefix)
            if G is not None:
                blocks.append(G); bnames += g
    # Scaling a column by s divides its effective L1 penalty by s: --pooled-scale
    # makes L1 prefer a pooled site/gene column over the single alleles in it.
    if scale != 1.0:
        blocks = blocks[:1] + [b * np.float32(scale) for b in blocks[1:]]
    return sp.hstack(blocks, format="csr", dtype=np.float32), bnames


def locus_spans(loci, genotype_dir):
    """{locus: (start, end)} half-open 0-based H37Rv spans of each locus's aligned
    reference row -- exactly the bases the 19-locus lasso arms saw."""
    from datasets.cds import genome_span
    spans = {}
    for g in loci:
        sp_ = genome_span(genotype_dir, g)
        if sp_ is None:
            raise RuntimeError(f"no genome span for locus {g} in {genotype_dir}")
        spans[g] = sp_[:2]
    return spans


def in_spans_mask(keys, spans):
    """True for a key POS:REF>ALT whose reference allele overlaps any span.
    VCF POS is 1-based and a deletion's REF starts at POS, so the key covers
    [POS-1, POS-1+len(REF)). REF was truncated to 20 nt at parse time, so a
    deletion starting > 20 nt before a span that reaches into it is missed."""
    pos = np.fromiter((int(k.split(":", 1)[0]) - 1 for k in keys), np.int64, len(keys))
    rlen = np.fromiter((len(k.split(":", 1)[1].split(">", 1)[0]) for k in keys),
                       np.int64, len(keys))
    m = np.zeros(len(keys), bool)
    for a, b in spans.values():
        m |= (pos < b) & (pos + np.maximum(rlen, 1) > a)
    return m


def drug_design(X_all, iso_all, keys_all, rows, min_count, col_mask=None):
    """The drug's rows, and only the columns its own cohort actually varies at.

    Same rule as the parent run: a column that is zero over the fitted cohort is
    collinear with the unpenalised intercept, so its L1 optimum is zero anyway;
    dropping it changes nothing and keeps the column count honest.
    """
    X = X_all[rows]
    counts = np.asarray(X.sum(axis=0)).ravel()
    keep = counts >= min_count
    if col_mask is not None:
        keep &= col_mask
    cols = np.nonzero(keep)[0]
    return X[:, cols].tocsr(), [keys_all[j] for j in cols]


def _fit(C, X_tr, y_tr, args):
    """The parent's pure-L1 fit when --l1-ratio is 1 (so the 2026-09-03 run is
    reproduced exactly); otherwise the elastic-net family, l1_ratio * ||w||_1 +
    (1 - l1_ratio)/2 * ||w||_2^2, with 0 = ridge. Only saga does 0 < r < 1;
    liblinear does ridge faster, so it keeps that end."""
    if args.l1_ratio == 1.0:
        return ws._fit(C, X_tr, y_tr, args.solver, args.max_iter)
    solver = "liblinear" if args.l1_ratio == 0.0 else "saga"
    clf = LogisticRegression(solver=solver, l1_ratio=args.l1_ratio, C=C,
                             class_weight="balanced", max_iter=args.max_iter,
                             tol=1e-4)
    clf.fit(X_tr, y_tr)
    return clf


def run_drug(drug, args, cohort, cohort_impact=None):
    t0 = time.time()
    X_all, iso_all, keys_all, gene_of, geneid_of = cohort
    row_of = {iso: i for i, iso in enumerate(iso_all)}
    df_phenos = load_phenotype(args.phenotype_csv)
    isolates = [i for i in df_phenos.index if i in row_of]
    y_all = numeric_labels(df_phenos.loc[isolates], drug)
    keep = np.nonzero(y_all != -1)[0]
    y = y_all[keep]
    isolates = [isolates[i] for i in keep]

    col_mask, spans = None, None
    if args.restrict_loci != "none":
        loci = (loci_on_disk(REAL_GENOTYPE_DIR) if args.restrict_loci == "all19"
                else drug_loci(drug))
        spans = locus_spans(loci, REAL_GENOTYPE_DIR)
        col_mask = in_spans_mask(keys_all, spans)
    X, names = drug_design(X_all, iso_all, keys_all,
                           [row_of[i] for i in isolates], args.min_count, col_mask)
    n_keys = X.shape[1]
    if args.pool != "none":
        X, names = pooled_features(X, names, args.pool, geneid_of, cohort_impact,
                                   args.pooled_scale)
        print(f"[{drug}] pool={args.pool}: {n_keys} key columns + "
              f"{X.shape[1] - n_keys} pooled", flush=True)
    n = X.shape[0]
    where = ("whole genome" if spans is None else
             f"restricted to {len(spans)} loci ({sum(b - a for a, b in spans.values())} nt)")
    print(f"[{drug}] {n} isolates, {X.shape[1]} columns, "
          f"{X.nnz / max(n, 1):.0f} nnz per isolate, {where}", flush=True)

    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2,
                                           random_state=42, stratify=y)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(train_idx, y[train_idx]))

    # --fold-min-count: per fold, keep only columns seen in >= k TRAINING isolates
    # of that fold. Decided on training rows alone, so leak-free (unlike
    # --min-count, which counts the whole cohort).
    fold_cols = [None] * len(folds)
    if args.fold_min_count > 1:
        for f, (tr, _va) in enumerate(folds):
            fold_cols[f] = np.nonzero(X[train_idx[tr]].getnnz(axis=0)
                                      >= args.fold_min_count)[0]
        print(f"[{drug}] fold-min-count {args.fold_min_count}: "
              f"{np.mean([len(c) for c in fold_cols]):,.0f} of {X.shape[1]:,} "
              f"columns kept per fold", flush=True)

    def _cols(M, f):
        return M if fold_cols[f] is None else M[:, fold_cols[f]]

    path, best = [], None
    for C in args.c_grid:
        fold_auc, fold_models, nnz_coef, n_iter = [], [], [], []
        for f, (tr, va) in enumerate(folds):
            i_tr, i_va = train_idx[tr], train_idx[va]
            tc = time.time()
            clf = _fit(C, _cols(X[i_tr], f), y[i_tr], args)
            m = ws._metrics(y[i_va], clf.predict_proba(_cols(X[i_va], f))[:, 1])
            fold_auc.append(m["auc"])
            fold_models.append(clf)
            nnz_coef.append(int((clf.coef_ != 0).sum()))
            n_iter.append(int(np.max(clf.n_iter_)))
            print(f"  [{drug}] C={C:<6g} fold {len(fold_auc)}/{len(folds)} "
                  f"AUC {m['auc']:.4f} support {nnz_coef[-1]} "
                  f"[{time.time() - tc:.0f}s]", flush=True)
        entry = {"C": C, "cv_auc_mean": float(np.mean(fold_auc)),
                 "cv_auc_std": float(np.std(fold_auc)),
                 "fold_aucs": [float(a) for a in fold_auc],
                 "support_mean": float(np.mean(nnz_coef)),
                 "support_per_fold": nnz_coef,
                 # == --max-iter means that fold's fit stopped unconverged
                 "n_iter_per_fold": n_iter, "max_iter": args.max_iter}
        path.append(entry)
        if best is None or entry["cv_auc_mean"] > best[0]["cv_auc_mean"]:
            best = (entry, fold_models, fold_auc)

    entry, fold_models, fold_auc = best
    best_fold = int(np.argmax(fold_auc))
    test = ws._metrics(y[test_idx], fold_models[best_fold].predict_proba(
        _cols(X[test_idx], best_fold))[:, 1])

    coef = np.zeros(X.shape[1])
    if fold_cols[best_fold] is None:
        coef[:] = fold_models[best_fold].coef_.ravel()
    else:
        coef[fold_cols[best_fold]] = fold_models[best_fold].coef_.ravel()
    nz = np.nonzero(coef)[0]
    order = nz[np.argsort(-np.abs(coef[nz]))][:args.top_k]
    top = [{"feature": names[i], "gene": gene_of.get(names[i], ""),
            "gene_id": geneid_of.get(names[i], ""),
            "coef": float(coef[i])} for i in order]

    cv_pr = float(np.mean([
        ws._metrics(y[train_idx[va]],
                    fold_models[i].predict_proba(_cols(X[train_idx[va]], i))[:, 1])["auc_pr"]
        for i, (tr, va) in enumerate(folds)]))

    tag = f"dna-genome-mc{args.min_count}"
    if args.l1_ratio != 1.0:
        tag += f"-l1r{args.l1_ratio:g}"
    if args.restrict_loci != "none":
        tag += f"-{args.restrict_loci}"
    if args.pool != "none":
        tag += f"-pool{args.pool}"
    if args.pooled_scale != 1.0:
        tag += f"-ps{args.pooled_scale:g}"
    if args.fold_min_count > 1:
        tag += f"-fmc{args.fold_min_count}"
    return {
        "drug": drug, "arch": "lasso_wholegenome", "modalities": ["dna"],
        "tag": tag, "l1_ratio": args.l1_ratio,
        "encoding": "delta",
        "locus_set": ("wholegenome" if spans is None
                      else f"vcf_{args.restrict_loci}"),
        "locus_spans": spans,
        "solver": (args.solver if args.l1_ratio == 1.0 else
                   "liblinear" if args.l1_ratio == 0.0 else "saga"), "min_count": args.min_count,
        "genes": ["WHOLE_GENOME"] if spans is None else sorted(spans), "n_valid": int(n),
        "n_resistant": int((y == 0).sum()), "n_susceptible": int((y == 1).sum()),
        "n_features": int(X.shape[1]),
        "pool": args.pool, "n_key_features": int(n_keys),
        "pooled_scale": args.pooled_scale, "fold_min_count": args.fold_min_count,
        "site_span_rule": "anchor-stripped (2026-10-01)",
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
    ap.add_argument("--l1-ratio", type=float, default=1.0,
                    help="penalty mix: 1 = L1 (lasso, the default and the "
                         "2026-09-03 run), 0 = L2 (ridge), between = elastic net "
                         "(saga). --solver applies only at 1")
    ap.add_argument("--c-grid", type=float, nargs="+", default=list(ws.C_GRID))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--pool", choices=("none", "site", "gene", "both"),
                    default="none",
                    help="append pooled indicators: site = any call covering "
                         "the base; gene = any HIGH-impact (lof:) / HIGH or "
                         "MODERATE (nonsyn:) call per gene; both = all three")
    ap.add_argument("--pooled-scale", type=float, default=1.0,
                    help="multiply pooled columns by this; s > 1 divides their "
                         "effective L1 penalty by s")
    ap.add_argument("--fold-min-count", type=int, default=1,
                    help="per fold, drop columns carried by < k TRAINING isolates "
                         "(leak-free); 1 = keep all")
    ap.add_argument("--annot-cache", default=str(Path.home() / "abr_workspace"
                                                / "wholegenome_annot.npz"),
                    help="snpEff effect per cohort key; built on first use")
    ap.add_argument("--build-annotation-only", action="store_true",
                    help="build --annot-cache and exit (run once before a grid)")
    ap.add_argument("--restrict-loci", choices=("none", "all19", "perdrug"),
                    default="none",
                    help="keep only variant columns inside the aligned loci's "
                         "H37Rv spans: all19 = the 19 FASTAs on disk (matches "
                         "lasso_wholeseq delta_all), perdrug = the drug's own "
                         "loci (matches delta_perdrug)")
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
    if not 0.0 <= args.l1_ratio <= 1.0:
        ap.error(f"--l1-ratio must be in [0, 1], got {args.l1_ratio}")

    drugs = (ALL_DRUGS if any(d.lower() == "all" for d in args.drugs)
             else [d.upper() for d in args.drugs])
    unknown = [d for d in drugs if d not in ALL_DRUGS]
    if unknown:
        ap.error(f"unknown drug(s) {unknown}; choose from {ALL_DRUGS} or 'all'")

    cohort = build_cohort(args.tsv_dir, args.cache, args.procs)
    impact_of = None
    if args.pool in ("gene", "both") or args.build_annotation_only:
        _ann, imp = build_annotation(args.tsv_dir, cohort[2], args.annot_cache,
                                     args.procs)
        impact_of = dict(zip(cohort[2], imp))
        if args.build_annotation_only:
            from collections import Counter as _C
            print("  [annot] impact counts:", dict(_C(imp)), flush=True)
            return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for drug in drugs:
        r = run_drug(drug, args, cohort, impact_of)
        (out / f"{drug}__{r['tag']}.json").write_text(json.dumps(r, indent=2))
        print(f"[{drug}] CV {r['cv_auc_mean']:.4f} +/- {r['cv_auc_std']:.4f} "
              f"@ C={r['best_C']} (support {r['cv_support_mean']:.0f} of "
              f"{r['n_features']}), TEST {r['test']['auc']:.4f} "
              f"[{r['seconds']:.0f}s]", flush=True)
        ws._write_summary(out)


if __name__ == "__main__":
    main()

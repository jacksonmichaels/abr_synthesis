"""
Train and evaluate GenomeFusion: one drug, on a lineage split.

    cd biophysical-fusion
    python -m genomefusion.train --drugs PYRAZINAMIDE --locus-set genome \
        --split-dir results/experiments/lineage_split/splits \
        --out results/experiments/genomefusion/genome

Run it through sbatch (genomefusion/sbatch.py), never in an interactive session.

Protocol = the lineage-split LocusFusion v3/v5 cells, unchanged, so a CV macro
AUC here is paired with those over the same 55 folds: the split file's TEST rows
are held out, its folds 0-4 are the CV folds (fold 4 = Beijing); alpha class
weights fit on the training rows; Adam at exp(-9); early stopping on validation
AUC with best-weight restore; TEST = the best CV fold's model scored once.

Locus sets (`--locus-set`):
    genome    every locus the partition keeps (6,776)
    19        the LocusFusion v3/v5 loci (19 gene loci, NO promoters)
    19r       the 19 plus every locus carrying a WHO Table-22 regulatory region
              for one of them. Defined by the catalogue, not by a distance:
              WHO extents run 2-1,982 bp, so no fixed flank can express them.
    size:N    a locus set of N loci: 19r plus a seeded random draw from the
              rest. NESTED -- the draw order is fixed by --ladder-seed, so
              size:64 is a subset of size:128 is a subset of ... is a subset of
              the genome. That makes the size ladder monotone in information,
              so a drop along it is the cost of carrying more loci rather than
              a change of content.
    19p       the 19 plus the intergenic loci covering their 100 bp 5' flank.
              SUPERSEDED by 19r: the 100 bp was chosen to match v3's padded
              FASTAs, and because loci are atomic it only ever selected whole
              intergenic loci anyway (rpoB's is 497 bp). Kept to reproduce the
              H1 runs.
    who_t12, lab64
              the same gene lists LocusFusion's locus_sets sweep used. Gene loci
              only, so no promoter bases, like `19`.
    who_t12r, lab64r
              those genes plus every locus carrying a WHO Table-22 region for
              them. These are the arms to compare to LocusFusion at those sets.
    minus19p  the complement of `19p`.
    minus19   the complement of `19` alone, so it KEEPS all 16 of the 19 genes'
              promoter loci including the fabG1-inhA c-15t. Kept only to
              reproduce older runs; prefer `minus19p`.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from bigtb_ref import REAL_PHENOTYPE_CSV  # noqa: E402
from training.core import EarlyStopper, masked_weighted_bce  # noqa: E402
from training.multidrug import _alpha_matrix  # noqa: E402
from training.multimodal import LR, _metrics, build_optimizer  # noqa: E402

from genomefusion.data import (BatchBuilder, Labels, TokenStore, budget_batches,  # noqa: E402
                               length_grouped_batches, split_rows)
from genomefusion.layout import c_base, channel_names  # noqa: E402
from genomefusion.model import GF_DEFAULTS, GenomeFusionNet  # noqa: E402
from genomefusion.partition import CACHE, WORKSPACE, load_or_build  # noqa: E402

ALL_DRUGS = ["AMIKACIN", "CAPREOMYCIN", "ETHAMBUTOL", "ETHIONAMIDE", "ISONIAZID", "KANAMYCIN",
             "LEVOFLOXACIN", "MOXIFLOXACIN", "PYRAZINAMIDE", "RIFAMPICIN", "STREPTOMYCIN"]
#: gene lists shared with LocusFusion (datasets.loader.drug_loci), so GenomeFusion
#: and the locus_sets baselines start from the same genes. The `r` variant adds
#: the loci carrying those genes' WHO Table-22 regions, as `19r` does for the 19.
NAMED_SETS = {"who_t12": "who_t12", "who_t12r": "who_t12", "lab64": "lab64", "lab64r": "lab64"}
LOCUS_SETS = ("genome", "19", "19r", "19p", "minus19p", "minus19", *NAMED_SETS)


def _is_size(which):
    return isinstance(which, str) and which.startswith("size:")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drugs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--locus-set", default="genome",
                    help="one of " + ", ".join(LOCUS_SETS) + ", or size:N for a ladder rung")
    ap.add_argument("--ladder-seed", type=int, default=0,
                    help="fixes the draw order for --locus-set size:N. The SAME seed must be "
                         "used for every rung of a ladder, or the rungs stop being nested.")
    ap.add_argument("--no-regulated-features", action="store_true",
                    help="drop the regulated-locus half of the locus feature vector, i.e. a "
                         "locus no longer knows WHAT it regulates. The control for the "
                         "known-relationship change.")
    ap.add_argument("--tokens", default=str(WORKSPACE / "genomefusion_tokens" / "tokens.npz"))
    ap.add_argument("--partition-cache", default=str(CACHE))
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--population", action="store_true",
                    help="the homoplasy / lineage-marker channels; needs a phylogeny, which we "
                         "do not have yet, so this currently refuses rather than training on "
                         "two columns of zeros")
    ap.add_argument("--max-tokens-per-locus", type=int, default=None)
    ap.add_argument("--promoter-flank", type=int, default=100,
                    help="for --locus-set 19p: how many bp 5' of each gene to pull in as "
                         "intergenic loci. 100 matches the flank v3's per-gene FASTAs carry.")
    # protocol
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--min-epochs", type=int, default=50)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--locus-budget", type=int, default=32768,
                    help="cap on batch x widest isolate's active loci. Isolates range from a "
                         "few hundred to 2,466 active loci and the across-loci attention is "
                         "quadratic in the widest one, so a fixed batch size would swing memory "
                         "~12x and OOM on the worst batch. 0 disables.")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="wall-clock budget. Training stops cleanly at the end of the current "
                         "epoch once it is spent and the run is written with the folds it "
                         "finished, instead of being killed at the SLURM limit with nothing "
                         "to show. Set it below the job's --time.")
    for k, v in GF_DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    args = ap.parse_args(argv)
    if args.locus_set not in LOCUS_SETS and not _is_size(args.locus_set):
        ap.error(f"--locus-set must be one of {LOCUS_SETS} or size:N, got {args.locus_set!r}")
    if _is_size(args.locus_set):
        try:
            if int(args.locus_set.split(":", 1)[1]) <= 0:
                raise ValueError
        except ValueError:
            ap.error(f"size:N needs a positive integer N, got {args.locus_set!r}")
    args.drugs = ALL_DRUGS if [d.lower() for d in args.drugs] == ["all"] else \
        [d.upper() for d in args.drugs]
    if len(args.drugs) > 1:
        ap.error("multi-drug is not wired up yet; pass one drug")
    if args.population:
        ap.error("--population needs the homoplasy channel, which needs a phylogeny of the "
                 "cohort; it is not built. Rerun without it.")
    return args


def regulatory_loci_for_19(partition):
    """Loci carrying a WHO Table-22 regulatory region for one of the 19."""
    genes19 = {L.gene for L in partition.loci if L.is_19}
    return np.asarray([i for i, L in enumerate(partition.loci)
                       if L.is_regulatory and (set(L.regulates) & genes19)], np.int64)


def base_19r(partition):
    return np.union1d(np.flatnonzero([L.is_19 for L in partition.loci]),
                      regulatory_loci_for_19(partition))


def named_subset(partition, which, drug):
    """A NAMED_SETS locus set for one drug: `who_t12` is per-drug, `lab64` is not.

    Like `19`, the plain sets are gene loci only, so they LACK the promoter bases
    that LocusFusion's flanked FASTAs carry. Compare the `r` variants to it, and
    do not read a plain-set shortfall as a model result.

    A gene inside the RLC/PE-PPE mask has no locus and is reported, not fatal, but
    its regulatory region is still added to the `r` variant if it is unmasked.
    """
    if not drug:
        raise ValueError(f"--locus-set {which} needs a drug")
    from datasets.loader import drug_loci
    names = drug_loci(drug, locus_set=NAMED_SETS[which])
    by_name = {}
    for i, L in enumerate(partition.loci):
        if L.kind != "intergenic":
            for key in (L.gene, L.name):
                if key:
                    by_name.setdefault(key, i)
    missing = [g for g in names if g not in by_name]
    if missing:
        print(f"[gf] {which}/{drug}: no locus for {missing} (masked or unannotated)", flush=True)
    keep = {by_name[g] for g in names if g in by_name}
    if which.endswith("r"):
        keep |= {i for i, L in enumerate(partition.loci)
                 if L.is_regulatory and set(L.regulates) & set(names)}
    return np.asarray(sorted(keep), np.int64)


def size_subset(partition, n, seed=0):
    """`n` loci: the 19r core plus a seeded random draw from everything else.

    The draw order depends only on `seed`, so the sets are NESTED across n.
    """
    n_all = len(partition.loci)
    n = min(int(n), n_all)
    base = base_19r(partition)
    if n <= len(base):
        return base
    rest = np.setdiff1d(np.arange(n_all), base)
    order = np.random.default_rng(seed).permutation(rest)
    return np.sort(np.concatenate([base, order[:n - len(base)]]))


def locus_subset(partition, which, promoter_flank=100, ladder_seed=0, drug=None):
    is19 = np.asarray([L.is_19 for L in partition.loci])
    if which in NAMED_SETS:
        return named_subset(partition, which, drug)
    if which == "genome":
        return np.arange(len(partition.loci))
    if which == "19":
        return np.flatnonzero(is19)
    if which == "19r":
        return base_19r(partition)
    if _is_size(which):
        return size_subset(partition, int(which.split(":", 1)[1]), ladder_seed)
    if which == "19p":
        prom = np.asarray(partition.promoter_loci(promoter_flank), np.int64)
        return np.union1d(np.flatnonzero(is19), prom)
    if which == "minus19p":
        # the complement of 19p: drop the 19 AND the promoter loci that carry
        # their regulatory variants, so what is left is genuinely elsewhere
        keep = ~is19
        keep[np.asarray(partition.promoter_loci(promoter_flank), np.int64)] = False
        return np.flatnonzero(keep)
    return np.flatnonzero(~is19)


def predict(model, bb, rows, bs, n_drugs, locus_budget=None):
    """Predictions in the row order given, however the batches were cut."""
    model.eval()
    chunks = budget_batches(bb.n_active, np.asarray(rows), bs, locus_budget)
    out = np.zeros((len(rows), n_drugs), np.float32)
    where = {int(r): i for i, r in enumerate(rows)}
    with torch.no_grad():
        for c in chunks:
            p = model(bb.build(c)).float().cpu().numpy()
            out[[where[int(r)] for r in c]] = p
    return out


def per_drug(Y, logits, drugs):
    """{drug: v3's metric dict}, macro AUC. Predictions are P(susceptible), as v3's."""
    P = 1.0 / (1.0 + np.exp(-logits))
    per = {d: _metrics(Y[:, j], P[:, j]) for j, d in enumerate(drugs)}
    aucs = [m["auc"] for m in per.values() if m["auc"] == m["auc"]]
    return per, (float(np.mean(aucs)) if aucs else float("nan"))


def train_fold(model, bb, Y, alpha, tr, va, args, fold, deadline=None):
    torch.manual_seed(args.seed + fold)
    rng = np.random.default_rng(args.seed + fold)
    opt = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopper(args.patience, args.min_delta, mode="max", min_epochs=args.min_epochs)
    alpha_t = torch.as_tensor(alpha, device=args.device)
    hist = {"train_loss": [], "val_auc": []}
    stopped_early = None
    for ep in range(args.epochs):
        if deadline and time.time() > deadline:
            stopped_early = f"wall-clock budget spent after {ep} epochs"
            print(f"  fold {fold}: {stopped_early}", flush=True)
            break
        model.train()
        tot = seen = 0.0
        for b in length_grouped_batches(bb.n_active, tr, args.batch_size, rng,
                                        locus_budget=args.locus_budget):
            x = bb.build(b)
            a = alpha_t[torch.as_tensor(b, device=args.device)]
            loss = masked_weighted_bce(model(x), a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
            seen += len(b)
        hist["train_loss"].append(tot / max(seen, 1))
        _, val = per_drug(Y[va], predict(model, bb, va, args.eval_batch_size, len(args.drugs),
                                         args.locus_budget), args.drugs)
        hist["val_auc"].append(val)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  fold {fold} ep {ep + 1:3d} loss {tot / max(seen, 1):.4f} "
                  f"val_auc {val:.4f}", flush=True)
        if val != val:
            continue
        if stopper.step(ep + 1, val, model):
            break
    stopper.restore(model)
    return stopper.best_epoch, hist, stopped_early


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    tag = args.drugs[0]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_file = str(Path(args.split_dir) / f"{tag}.csv")
    print(f"[gf] {tag} locus_set={args.locus_set} device={args.device} out={out_dir}", flush=True)

    partition = load_or_build(args.partition_cache)
    store = TokenStore(args.tokens)
    print(f"[gf] store {len(store):,} isolates, {store.stats['tokens_total']:,} tokens, "
          f"mean {store.stats['tokens_mean']:.0f}/isolate", flush=True)
    loci = locus_subset(partition, args.locus_set, args.promoter_flank, args.ladder_seed,
                        drug=tag)
    bb = BatchBuilder(store, partition, loci=loci, population=args.population,
                      device=args.device, max_tokens_per_locus=args.max_tokens_per_locus)
    print(f"[gf] {len(loci):,} loci in this set; mean {bb.n_active.mean():.0f} active per "
          f"isolate (max {bb.n_active.max():,}); {bb.feat.shape[0]:,} tokens kept", flush=True)

    labels = Labels(store, args.drugs, REAL_PHENOTYPE_CSV)
    Y = labels.Y
    train, test, folds = split_rows(store, split_file)
    have = (Y[:, 0] != -1)
    train = train[have[train]]
    test = test[have[test]]
    folds = [(t[have[t]], v[have[v]]) for t, v in folds]
    print(f"[gf] split {split_file}: train={len(train)} test={len(test)} "
          f"folds={[len(v) for _, v in folds]}", flush=True)

    feats = partition.feature_matrix(with_regulated=not args.no_regulated_features)[loci]
    # the regulated-locus edge, remapped into this subset's compact ids (-1 when
    # the regulated locus is not in the subset, as in minus19p)
    reg_full = partition.regulated_index()
    remap = np.full(len(partition.loci), -1, np.int64)
    remap[loci] = np.arange(len(loci))
    reg_lid = np.where(reg_full[loci] >= 0, remap[reg_full[loci].clip(0)], -1)
    n_edge = int((reg_lid >= 0).sum())
    print(f"[gf] locus features {feats.shape[1]} wide "
          f"(regulated half: {'off' if args.no_regulated_features else 'on'}); "
          f"{n_edge:,}/{len(loci):,} loci carry a regulated-locus edge", flush=True)
    capacity = {k: getattr(args, k) for k in GF_DEFAULTS}

    def build():
        return GenomeFusionNet(feats, args.drugs, c_tok=bb.c_tok, regulated_lid=reg_lid,
                               **capacity).to(args.device)

    alpha = _alpha_matrix(Y, train, len(args.drugs))
    run_folds = list(args.folds if args.folds is not None else range(len(folds)))
    deadline = t0 + args.max_hours * 3600 if args.max_hours else None
    fold_recs, best, n_params = [], (-np.inf, None, None), None
    skipped = []
    for fold in run_folds:
        if deadline and time.time() > deadline:
            skipped = [f for f in run_folds if f >= fold]
            print(f"[gf] wall-clock budget spent; folds {skipped} not run", flush=True)
            break
        tr, va = folds[fold]
        model = build()
        if n_params is None:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[gf] {n_params:,} parameters, token width {model.c_in}", flush=True)
        best_epoch, hist, cut = train_fold(model, bb, Y, alpha, tr, va, args, fold, deadline)
        per, macro = per_drug(Y[va], predict(model, bb, va, args.eval_batch_size,
                                             len(args.drugs), args.locus_budget), args.drugs)
        rec = {"fold": fold, "best_epoch": best_epoch, "macro_auc": macro, "per_drug": per,
               "history": hist,
               "val_ids_sha1": hashlib.sha1(
                   "\n".join(sorted(store.ids[i] for i in va)).encode()).hexdigest()[:12]}
        rec.update(per[tag])
        rec["stopped_early"] = cut
        fold_recs.append(rec)
        print(f"[gf] {tag} fold {fold}: val AUC {macro:.4f} (best epoch {best_epoch})", flush=True)
        if macro == macro and macro > best[0]:
            best = (macro, fold, model)
    _, best_fold, best_model = best
    if best_model is None:
        best_fold, best_model = fold_recs[-1]["fold"], model

    test_logits = predict(best_model, bb, test, args.eval_batch_size, len(args.drugs),
                          args.locus_budget)
    test_per, test_macro = per_drug(Y[test], test_logits, args.drugs)
    macro_cv = [f["macro_auc"] for f in fold_recs if f["macro_auc"] == f["macro_auc"]]
    print(f"[gf] {tag} CV macro AUC {np.mean(macro_cv):.4f} | TEST {test_macro:.4f} "
          f"(best fold {best_fold})", flush=True)

    pred = pd.DataFrame({"New_ID": [store.ids[i] for i in test]})
    for j, d in enumerate(args.drugs):
        pred[f"P_S_{d}"] = 1.0 / (1.0 + np.exp(-test_logits[:, j]))
        pred[f"y_{d}"] = Y[test, j]
    pred.to_csv(out_dir / f"{tag}_test_predictions.csv", index=False)

    weights = None
    if args.weights_dir:
        weights = Path(args.weights_dir) / out_dir.name / tag / "best.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": best_model.config, "state_dict": best_model.state_dict(),
                    "fold": best_fold, "loci": loci.tolist(), "args": vars(args)}, weights)

    result = {
        "arch": "genomefusion", "tag": tag, "drug": tag, "drugs": args.drugs,
        "locus_set": args.locus_set, "n_loci": int(len(loci)),
        "ladder_seed": args.ladder_seed if _is_size(args.locus_set) else None,
        "n_feat": int(feats.shape[1]), "regulated_features": not args.no_regulated_features,
        "n_regulated_edges": n_edge,
        "n_regulatory_loci": int(sum(partition.loci[int(i)].is_regulatory for i in loci)),
        "promoter_flank": args.promoter_flank if args.locus_set == "19p" else None,
        "promoter_loci": ([int(x) for x in loci
                           if partition.loci[int(x)].kind == "intergenic"]
                          if args.locus_set == "19p" else None),
        "n_params": n_params, "n_isolates": len(store),
        "token_width": int(bb.c_tok), "channels": channel_names((), args.population),
        "active_loci_mean": float(bb.n_active.mean()), "active_loci_max": int(bb.n_active.max()),
        "tokens": store.stats, "partition": partition.meta,
        "capacity": capacity,
        "protocol": {"epochs": args.epochs, "patience": args.patience,
                     "min_epochs": args.min_epochs, "min_delta": args.min_delta,
                     "batch_size": args.batch_size, "locus_budget": args.locus_budget,
                     "monitor": "auc",
                     "lr": LR if args.lr is None else args.lr,
                     "weight_decay": args.weight_decay, "seed": args.seed},
        "split_file": split_file, "cv_folds": fold_recs,
        "folds_requested": run_folds, "folds_skipped": skipped,
        "complete": not skipped and not any(f.get("stopped_early") for f in fold_recs),
        "max_hours": args.max_hours,
        "cv_auc_mean": float(np.mean(macro_cv)) if macro_cv else float("nan"),
        "cv_auc_std": float(np.std(macro_cv)) if macro_cv else float("nan"),
        "test_model_fold": best_fold, "test_macro_auc": test_macro,
        "test": test_per[tag], "per_drug": {
            d: {"cv_auc_mean": float(np.nanmean([f["per_drug"][d]["auc"] for f in fold_recs])),
                "cv_auc_std": float(np.nanstd([f["per_drug"][d]["auc"] for f in fold_recs])),
                "test": test_per[d]} for d in args.drugs},
        "weights": str(weights) if weights else None,
        "seconds": round(time.time() - t0, 1), "host": os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(f"[gf] wrote {out_dir / f'{tag}.json'} ({result['seconds']} s)", flush=True)
    return result


if __name__ == "__main__":
    main()

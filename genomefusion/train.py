"""
Train and evaluate GenomeFusion: one drug, on a lineage split.

    python -m genomefusion.train --drugs PYRAZINAMIDE --locus-set 19r --max-hours 20 \
        --out results/genomefusion/gf19r

Run it in a SLURM job (1 GPU), never on a login node. Defaults read the
partition cache, token store, lineage splits and weights dir from paths.py.

Protocol = LocusFusion's, so on the same split files the folds pair: the split
file's TEST rows (the held-out lineage) are held out, its folds 0-4 are the CV
folds (splits/build_holdout_splits.py); alpha class weights fit on the
training rows; Adam at exp(-9); early stopping on validation AUC with best-weight
restore; TEST = the best CV fold's model scored once.

Arms (one job per drug; `--out results/genomefusion/<arm>`, which also names the
weights subdirectory). --max-hours stops 4 h short of a 24 h / 48 h SLURM limit:
    canary        --drugs PYRAZINAMIDE --locus-set genome --folds 0 --locus-budget 16384
    gf19p         --locus-set 19p      --max-hours 20
    gf19r         --locus-set 19r      --max-hours 20
    gf19r_nofeat  --locus-set 19r      --max-hours 20 --no-regulated-features
    gf_who_t12r   --locus-set who_t12r --max-hours 20
    gf_lab64r     --locus-set lab64r   --max-hours 20
    gf_minus19p   --locus-set minus19p --max-hours 44
    size6776      --locus-set size:6776 --ladder-seed 0 --max-hours 44   (whole genome)

Locus sets (`--locus-set`):
    genome    every locus the partition keeps (6,776)
    19        the 19 LocusFusion gene loci, NO promoters
    19r       the 19 plus every locus carrying a WHO Table-22 regulatory region
              for one of them. Defined by the catalogue, not by a distance:
              WHO extents run 2-1,982 bp, so no fixed flank can express them.
    19p       the 19 plus the intergenic loci covering their 100 bp 5' flank
              (the flank LocusFusion's FASTAs carry). Loci are atomic, so this
              only ever selects whole intergenic loci (rpoB's is 497 bp).
    minus19p  the complement of `19p`.
    who_t12r, lab64r
              the gene lists LocusFusion's locus_sets sweep used
              (datasets.loader.drug_loci) plus every locus carrying a WHO
              Table-22 region for those genes.
    size:N    19r plus a seeded random draw from the rest, N loci in all.
              Nested across N for one --ladder-seed; size:6776 is the genome.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paths import GF_TOKENS, GF_WEIGHTS_DIR, REAL_PHENOTYPE_CSV, SPLIT_DIR
from training.core import EarlyStopper, masked_weighted_bce
from training.multidrug import _alpha_matrix
from training.multimodal import LR, _metrics, build_optimizer

from genomefusion.data import (BatchBuilder, Labels, TokenStore, budget_batches,
                               length_grouped_batches, split_rows)
from genomefusion.layout import channel_names
from genomefusion.partition import CACHE, PROMOTER_FLANK, load_or_build
from models.genomefusion import GF_DEFAULTS, GenomeFusionNet

#: gene lists shared with LocusFusion (datasets.loader.drug_loci), so GenomeFusion
#: and the locus_sets baselines start from the same genes, plus the loci carrying
#: those genes' WHO Table-22 regions, as `19r` does for the 19.
NAMED_SETS = {"who_t12r": "who_t12", "lab64r": "lab64"}
LOCUS_SETS = ("genome", "19", "19r", "19p", "minus19p", *NAMED_SETS)


def _is_size(which):
    return isinstance(which, str) and which.startswith("size:")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drugs", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="results dir, results/genomefusion/<arm>")
    ap.add_argument("--split-dir", default=SPLIT_DIR,
                    help="lineage splits, one {DRUG}.csv each")
    ap.add_argument("--locus-set", default="genome",
                    help="one of " + ", ".join(LOCUS_SETS) + ", or size:N for a ladder rung")
    ap.add_argument("--ladder-seed", type=int, default=0,
                    help="fixes the draw order for --locus-set size:N. The SAME seed must be "
                         "used for every rung of a ladder, or the rungs stop being nested.")
    ap.add_argument("--no-regulated-features", action="store_true",
                    help="drop the regulated-locus half of the locus feature vector, i.e. a "
                         "locus no longer knows WHAT it regulates. The control for the "
                         "known-relationship change.")
    ap.add_argument("--tokens", default=str(GF_TOKENS / "tokens.npz"))
    ap.add_argument("--partition-cache", default=str(CACHE))
    ap.add_argument("--weights-dir", default=str(GF_WEIGHTS_DIR),
                    help="best.pt goes to <weights-dir>/<basename of --out>/<drug>/")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    args.drugs = [d.upper() for d in args.drugs]
    if len(args.drugs) > 1:
        ap.error("single-drug only; pass one drug")
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
    """A NAMED_SETS locus set for one drug: `who_t12r` is per-drug, `lab64r` is not.

    A gene inside the RLC/PE-PPE mask has no locus and is reported, not fatal, but
    its regulatory region is still added if it is unmasked.
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


def locus_subset(partition, which, ladder_seed=0, drug=None):
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
        prom = np.asarray(partition.promoter_loci(), np.int64)
        return np.union1d(np.flatnonzero(is19), prom)
    # minus19p, the complement of 19p: drop the 19 AND the promoter loci that
    # carry their regulatory variants, so what is left is genuinely elsewhere
    keep = ~is19
    keep[np.asarray(partition.promoter_loci(), np.int64)] = False
    return np.flatnonzero(keep)


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
    """{drug: LocusFusion's metric dict}, macro AUC. Predictions are P(susceptible), as there."""
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
    loci = locus_subset(partition, args.locus_set, args.ladder_seed, drug=tag)
    bb = BatchBuilder(store, partition, loci=loci, device=args.device)
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
        "promoter_flank": PROMOTER_FLANK if args.locus_set == "19p" else None,
        "promoter_loci": ([int(x) for x in loci
                           if partition.loci[int(x)].kind == "intergenic"]
                          if args.locus_set == "19p" else None),
        "n_params": n_params, "n_isolates": len(store),
        "token_width": int(bb.c_tok), "channels": channel_names(),
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

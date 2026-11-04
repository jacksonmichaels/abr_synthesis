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
    genome    every locus the partition keeps
    19        the LocusFusion v3/v5 loci (the parity arm)
    minus19   every locus except those (is there signal outside the known loci?)
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

from genomefusion.data import (BatchBuilder, Labels, TokenStore, length_grouped_batches,  # noqa: E402
                               split_rows)
from genomefusion.layout import c_base, channel_names  # noqa: E402
from genomefusion.model import GF_DEFAULTS, GenomeFusionNet  # noqa: E402
from genomefusion.partition import CACHE, WORKSPACE, load_or_build  # noqa: E402

ALL_DRUGS = ["AMIKACIN", "CAPREOMYCIN", "ETHAMBUTOL", "ETHIONAMIDE", "ISONIAZID", "KANAMYCIN",
             "LEVOFLOXACIN", "MOXIFLOXACIN", "PYRAZINAMIDE", "RIFAMPICIN", "STREPTOMYCIN"]
LOCUS_SETS = ("genome", "19", "minus19")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drugs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--locus-set", choices=LOCUS_SETS, default="genome")
    ap.add_argument("--tokens", default=str(WORKSPACE / "genomefusion_tokens" / "tokens.npz"))
    ap.add_argument("--partition-cache", default=str(CACHE))
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--population", action="store_true",
                    help="the homoplasy / lineage-marker channels; needs a phylogeny, which we "
                         "do not have yet, so this currently refuses rather than training on "
                         "two columns of zeros")
    ap.add_argument("--max-tokens-per-locus", type=int, default=None)
    # protocol
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--min-epochs", type=int, default=50)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    for k, v in GF_DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    args = ap.parse_args(argv)
    args.drugs = ALL_DRUGS if [d.lower() for d in args.drugs] == ["all"] else \
        [d.upper() for d in args.drugs]
    if len(args.drugs) > 1:
        ap.error("multi-drug is not wired up yet; pass one drug")
    if args.population:
        ap.error("--population needs the homoplasy channel, which needs a phylogeny of the "
                 "cohort; it is not built. Rerun without it.")
    return args


def locus_subset(partition, which):
    is19 = np.asarray([L.is_19 for L in partition.loci])
    if which == "genome":
        return np.arange(len(partition.loci))
    if which == "19":
        return np.flatnonzero(is19)
    return np.flatnonzero(~is19)


def predict(model, bb, rows, bs, n_drugs):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), bs):
            out.append(model(bb.build(rows[s:s + bs])).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, n_drugs))


def per_drug(Y, logits, drugs):
    """{drug: v3's metric dict}, macro AUC. Predictions are P(susceptible), as v3's."""
    P = 1.0 / (1.0 + np.exp(-logits))
    per = {d: _metrics(Y[:, j], P[:, j]) for j, d in enumerate(drugs)}
    aucs = [m["auc"] for m in per.values() if m["auc"] == m["auc"]]
    return per, (float(np.mean(aucs)) if aucs else float("nan"))


def train_fold(model, bb, Y, alpha, tr, va, args, fold):
    torch.manual_seed(args.seed + fold)
    rng = np.random.default_rng(args.seed + fold)
    opt = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopper(args.patience, args.min_delta, mode="max", min_epochs=args.min_epochs)
    alpha_t = torch.as_tensor(alpha, device=args.device)
    hist = {"train_loss": [], "val_auc": []}
    for ep in range(args.epochs):
        model.train()
        tot = seen = 0.0
        for b in length_grouped_batches(bb.n_active, tr, args.batch_size, rng):
            x = bb.build(b)
            a = alpha_t[torch.as_tensor(b, device=args.device)]
            loss = masked_weighted_bce(model(x), a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
            seen += len(b)
        hist["train_loss"].append(tot / max(seen, 1))
        _, val = per_drug(Y[va], predict(model, bb, va, args.eval_batch_size, len(args.drugs)),
                          args.drugs)
        hist["val_auc"].append(val)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  fold {fold} ep {ep + 1:3d} loss {tot / max(seen, 1):.4f} "
                  f"val_auc {val:.4f}", flush=True)
        if val != val:
            continue
        if stopper.step(ep + 1, val, model):
            break
    stopper.restore(model)
    return stopper.best_epoch, hist


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
    loci = locus_subset(partition, args.locus_set)
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

    feats = partition.feature_matrix()[loci]
    capacity = {k: getattr(args, k) for k in GF_DEFAULTS}

    def build():
        return GenomeFusionNet(feats, args.drugs, c_tok=bb.c_tok, **capacity).to(args.device)

    alpha = _alpha_matrix(Y, train, len(args.drugs))
    run_folds = args.folds if args.folds is not None else range(len(folds))
    fold_recs, best, n_params = [], (-np.inf, None, None), None
    for fold in run_folds:
        tr, va = folds[fold]
        model = build()
        if n_params is None:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[gf] {n_params:,} parameters, token width {model.c_in}", flush=True)
        best_epoch, hist = train_fold(model, bb, Y, alpha, tr, va, args, fold)
        per, macro = per_drug(Y[va], predict(model, bb, va, args.eval_batch_size,
                                             len(args.drugs)), args.drugs)
        rec = {"fold": fold, "best_epoch": best_epoch, "macro_auc": macro, "per_drug": per,
               "history": hist,
               "val_ids_sha1": hashlib.sha1(
                   "\n".join(sorted(store.ids[i] for i in va)).encode()).hexdigest()[:12]}
        rec.update(per[tag])
        fold_recs.append(rec)
        print(f"[gf] {tag} fold {fold}: val AUC {macro:.4f} (best epoch {best_epoch})", flush=True)
        if macro == macro and macro > best[0]:
            best = (macro, fold, model)
    _, best_fold, best_model = best
    if best_model is None:
        best_fold, best_model = fold_recs[-1]["fold"], model

    test_logits = predict(best_model, bb, test, args.eval_batch_size, len(args.drugs))
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
        "n_params": n_params, "n_isolates": len(store),
        "token_width": int(bb.c_tok), "channels": channel_names((), args.population),
        "active_loci_mean": float(bb.n_active.mean()), "active_loci_max": int(bb.n_active.max()),
        "tokens": store.stats, "partition": partition.meta,
        "capacity": capacity,
        "protocol": {"epochs": args.epochs, "patience": args.patience,
                     "min_epochs": args.min_epochs, "min_delta": args.min_delta,
                     "batch_size": args.batch_size, "monitor": "auc",
                     "lr": LR if args.lr is None else args.lr,
                     "weight_decay": args.weight_decay, "seed": args.seed},
        "split_file": split_file, "cv_folds": fold_recs,
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

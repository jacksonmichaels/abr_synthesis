"""
Train and evaluate LocusFusion v5_fixed: one drug, or all eleven jointly.

    # one drug, 19 loci, lineage 2 held out (results/splits/heldout_2/{DRUG}.csv)
    python -m locusfusion.train --drugs PYRAZINAMIDE --split-dir results/splits/heldout_2 --run v5_fixed_L2
    # the multi-drug model (one query per drug, one proximity channel per drug;
    # <split-dir>/MULTIDRUG.csv)
    python -m locusfusion.train --drugs all --split-dir results/splits/heldout_2 --run md_v5_fixed_L2

Run it through sbatch, not in an interactive session. The token cache for the
drug(s) is built on first use (a few minutes, ~4 GB RAM) and written atomically,
so concurrent jobs at worst build it twice.

Tokens: synonymous changes dropped, residue one-hot kept, the site-proximity
channel on; genes from paths.REAL_GENOTYPE_DIR, promoter windows from
paths.ATG_REGULATORY_DIR (the start-codon-anchored windows).

Protocol: the split file's TEST rows are held out, its folds 0-4 are the CV
folds (splits/build_holdout_splits.py); alpha class weights fit on the training rows; Adam at
exp(-9); early stopping on validation AUC (macro over drugs for the multi-drug
model) with best-weight restore; TEST = the best CV fold's model scored once.
Single-drug and multi-drug are the same code: a single-drug run is a one-column
label matrix. A single-drug run keeps the isolates with that drug's label; the
multi-drug run keeps every isolate with at least one, and each drug's loss and
metrics use only its labelled rows.

Writes <out>/<TAG>.json (TAG = the drug, or MULTIDRUG) and
<out>/<TAG>_test_predictions.csv, with <out> = paths.RESULTS_DIR/locusfusion/<run>,
and the best fold's weights to <weights-dir>/<run>/<TAG>/best.pt.
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

import paths
from locusfusion.binding_sites.build_distances import load_distances
from locusfusion.layout import C_BASE, channel_names
from locusfusion.tokenizer import ALL_DRUGS, MAX_VARIANTS, load_or_build
from models.locusfusion import LF_DEFAULTS, LocusFusionNet, proximity_table
from training.core import EarlyStopper, masked_weighted_bce
from training.multidrug import _alpha_matrix
from training.multimodal import LR, _metrics, build_optimizer

MODEL_KEYS = ("feat", "coord", "valid", "site_pos", "n_occ")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drugs", nargs="+", required=True, help="one drug, several, or 'all'")
    sp = ap.add_mutually_exclusive_group()
    sp.add_argument("--split-dir", default=paths.SPLIT_DIR,
                    help="folder of split files {TAG}.csv (TAG = the drug, or MULTIDRUG)")
    sp.add_argument("--split-file", help="one split file (New_ID, role, fold)")
    ap.add_argument("--run", default="v5_fixed",
                    help="results go to paths.RESULTS_DIR/locusfusion/<run>")
    ap.add_argument("--out", default=None, help="results folder (overrides --run)")
    ap.add_argument("--weights-dir", default=str(paths.ROOT / "model_weights_lf"),
                    help="the best fold's weights go to <weights-dir>/<out folder name>/<TAG>/")
    ap.add_argument("--token-cache", default=str(paths.ROOT / "token_cache_v4"))
    ap.add_argument("--genotype-dir", default=paths.REAL_GENOTYPE_DIR,
                    help="aligned gene FASTAs. The token cache keys on this path string, so a"
                         " different path gets its own cache")
    ap.add_argument("--regulatory-dir", default=paths.ATG_REGULATORY_DIR,
                    help="aligned promoter FASTAs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # protocol (the lineage-split v3 cells)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--min-epochs", type=int, default=50)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None, help="default exp(-9), as v3")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="+", default=None,
                    help="run only these CV folds (a canary); default all")
    # capacity
    for k, v in LF_DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    args = ap.parse_args(argv)
    args.drugs = ALL_DRUGS if [d.lower() for d in args.drugs] == ["all"] else [d.upper() for d in args.drugs]
    return args


class Data:
    """The kept isolates' tokens, on the device, plus labels and split."""

    def __init__(self, ts, drugs, split_file, device):
        cols = [ts.drugs.index(d) for d in drugs]
        Y = ts.Y[:, cols]
        keep = np.nonzero((Y != -1).any(1))[0]
        self.ids = [ts.isolate_ids[i] for i in keep]
        self.Y = Y[keep]
        self.t = {k: torch.as_tensor(getattr(ts, k)[keep], device=device) for k in MODEL_KEYS}
        self.t["coord"] = self.t["coord"].float()
        sp = pd.read_csv(split_file, dtype={"New_ID": str}).set_index("New_ID")
        missing = [i for i in self.ids if i not in sp.index]
        if missing:
            raise ValueError(f"{split_file}: {len(missing)} of {len(self.ids)} isolates have no "
                             f"split (e.g. {missing[:3]})")
        sp = sp.loc[self.ids]
        self.test_idx = np.nonzero((sp.role == "test").values)[0]
        self.train_idx = np.nonzero((sp.role == "train").values)[0]
        fold_of = sp.fold.values[self.train_idx]
        self.folds = [(self.train_idx[fold_of != k], self.train_idx[fold_of == k])
                      for k in sorted(set(fold_of))]
        print(f"[data] {len(self.ids)} isolates, split {split_file}: train={len(self.train_idx)} "
              f"test={len(self.test_idx)} folds={[len(v) for _, v in self.folds]}", flush=True)

    def batch(self, idx):
        idx = torch.as_tensor(idx, device=self.t["feat"].device)
        return {k: v[idx] for k, v in self.t.items()}


def predict(model, data, idx, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            b = data.batch(idx[s:s + bs])
            out.append(model(**{k: b[k] for k in MODEL_KEYS}).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.fc_out.out_features))


def per_drug(Y, logits, drugs):
    """{drug: v3's metric dict}, macro AUC. Predictions are P(susceptible), as v3's."""
    P = 1.0 / (1.0 + np.exp(-logits))
    per = {d: _metrics(Y[:, j], P[:, j]) for j, d in enumerate(drugs)}
    aucs = [m["auc"] for m in per.values() if m["auc"] == m["auc"]]
    return per, (float(np.mean(aucs)) if aucs else float("nan"))


def train_fold(model, data, alpha, tr, va, args, fold):
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    opt = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopper(args.patience, args.min_delta, mode="max", min_epochs=args.min_epochs)
    alpha_t = torch.as_tensor(alpha, device=args.device)
    # train_consistency / aug_inserted are always 0 (no augmentation); kept so the
    # history, and the result JSON, has the same fields as every earlier run
    hist = {"train_loss": [], "train_consistency": [], "aug_inserted": [], "val_auc": []}
    for ep in range(args.epochs):
        model.train()
        perm = tr[np.random.permutation(len(tr))]
        tot = seen = 0.0
        for s in range(0, len(perm), args.batch_size):
            b = perm[s:s + args.batch_size]
            x = data.batch(b)
            a = alpha_t[torch.as_tensor(b, device=args.device)]
            logits = model(**{k: x[k] for k in MODEL_KEYS})
            loss = masked_weighted_bce(logits, a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
            seen += len(b)
        hist["train_loss"].append(tot / seen)
        hist["train_consistency"].append(0.0)
        hist["aug_inserted"].append(0.0)
        _, val = per_drug(data.Y[va], predict(model, data, va), args.drugs)
        hist["val_auc"].append(val)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  fold {fold} ep {ep + 1:3d} loss {tot / seen:.4f} val_auc {val:.4f}", flush=True)
        if val != val:
            continue
        if stopper.step(ep + 1, val, model):
            break
    stopper.restore(model)
    return stopper.best_epoch, hist


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    multi = len(args.drugs) > 1
    tag = "MULTIDRUG" if multi else args.drugs[0]
    out_dir = Path(args.out) if args.out else paths.RESULTS_DIR / "locusfusion" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    split_file = args.split_file or str(Path(args.split_dir) / f"{tag}.csv")
    print(f"[lf] {tag} drugs={args.drugs} device={args.device} out={out_dir}", flush=True)
    print(f"[lf] data: geno={args.genotype_dir} pheno={paths.REAL_PHENOTYPE_CSV} "
          f"reg={args.regulatory_dir}", flush=True)

    ts = load_or_build(args.token_cache, args.drugs, args.genotype_dir, paths.REAL_PHENOTYPE_CSV,
                       args.regulatory_dir)
    data = Data(ts, args.drugs, split_file, args.device)
    dist_drugs = list(args.drugs)
    prox = proximity_table(dist_drugs, ts.loci, load_distances())
    capacity = {k: getattr(args, k) for k in LF_DEFAULTS}

    def build():
        return LocusFusionNet(ts.loci, ts.kept, args.drugs, prox=prox, dist_drugs=dist_drugs,
                              max_variants=MAX_VARIANTS, c_base=C_BASE,
                              **capacity).to(args.device)

    alpha = _alpha_matrix(data.Y, data.train_idx, len(args.drugs))
    folds, best = [], (-np.inf, None, None)
    run_folds = args.folds if args.folds is not None else range(len(data.folds))
    n_params = None
    for fold in run_folds:
        tr, va = data.folds[fold]
        model = build()
        if n_params is None:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[lf] {n_params:,} parameters, token width {model.c_tok}", flush=True)
        best_epoch, hist = train_fold(model, data, alpha, tr, va, args, fold)
        per, macro = per_drug(data.Y[va], predict(model, data, va), args.drugs)
        rec = {"fold": fold, "best_epoch": best_epoch, "macro_auc": macro, "per_drug": per,
               "history": hist,
               "val_ids_sha1": hashlib.sha1("\n".join(sorted(data.ids[i] for i in va))
                                            .encode()).hexdigest()[:12]}
        if not multi:
            rec.update(per[tag])
        folds.append(rec)
        print(f"[lf] {tag} fold {fold}: val AUC {macro:.4f} (best epoch {best_epoch})", flush=True)
        if macro == macro and macro > best[0]:
            best = (macro, fold, model)
    best_auc, best_fold, best_model = best
    if best_model is None:
        best_fold, best_model = folds[-1]["fold"], model

    test_logits = predict(best_model, data, data.test_idx)
    test_per, test_macro = per_drug(data.Y[data.test_idx], test_logits, args.drugs)
    cv_per = {d: {"cv_auc_mean": float(np.nanmean([f["per_drug"][d]["auc"] for f in folds])),
                  "cv_auc_std": float(np.nanstd([f["per_drug"][d]["auc"] for f in folds])),
                  "test": test_per[d]} for d in args.drugs}
    macro_cv = [f["macro_auc"] for f in folds if f["macro_auc"] == f["macro_auc"]]
    print(f"[lf] {tag} CV macro AUC {np.mean(macro_cv):.4f} | TEST macro AUC {test_macro:.4f} "
          f"(best fold {best_fold})", flush=True)

    pred = pd.DataFrame({"New_ID": [data.ids[i] for i in data.test_idx]})
    for j, d in enumerate(args.drugs):
        pred[f"P_S_{d}"] = 1.0 / (1.0 + np.exp(-test_logits[:, j]))
        pred[f"y_{d}"] = data.Y[data.test_idx, j]
    pred.to_csv(out_dir / f"{tag}_test_predictions.csv", index=False)

    weights = Path(args.weights_dir) / out_dir.name / tag / "best.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": best_model.config, "kept": ts.kept.tolist(),
               "state_dict": best_model.state_dict(), "fold": best_fold,
               "token_config": ts.config, "args": vars(args)}, weights)

    result = {
        "arch": "locusfusion_v4", "tag": tag, "drugs": args.drugs, "multidrug": multi,
        "n_params": n_params, "n_isolates": len(data.ids),
        "token_width": C_BASE + len(dist_drugs),
        "keep_residue": True,
        "channels": channel_names(dist_drugs),
        "tokens": {k: v for k, v in ts.config.items() if k != "frames"},
        "capacity": capacity, "max_variants": MAX_VARIANTS,
        "distance": {"drugs": dist_drugs, "length_A": 10.0, "missing": -1.0},
        "augmentation": None,
        "drop_synonymous": True,
        "protocol": {"epochs": args.epochs, "patience": args.patience,
                     "min_epochs": args.min_epochs, "min_delta": args.min_delta,
                     "batch_size": args.batch_size, "monitor": "auc",
                     "lr": LR if args.lr is None else args.lr,
                     "weight_decay": args.weight_decay, "seed": args.seed},
        "split_file": split_file, "cv_folds": folds,
        "cv_auc_mean": float(np.mean(macro_cv)) if macro_cv else float("nan"),
        "cv_auc_std": float(np.std(macro_cv)) if macro_cv else float("nan"),
        "test_model_fold": best_fold, "test_macro_auc": test_macro,
        "per_drug": cv_per,
        "weights": str(weights),
        "seconds": round(time.time() - t0, 1), "host": os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    if not multi:
        result.update(drug=tag, test=test_per[tag])
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(f"[lf] wrote {out_dir / f'{tag}.json'} ({result['seconds']} s)", flush=True)
    return result


if __name__ == "__main__":
    main()

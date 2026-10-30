"""
Train and evaluate LocusFusion v4: one drug, or all eleven jointly.

    cd biophysical-fusion
    # one drug, 19 loci, lineage split, synonymous augmentation on
    python -m locusfusion_v4.train --drugs PYRAZINAMIDE \
        --split-dir results/experiments/lineage_split/splits \
        --aug-frac 0.5 --out results/experiments/locusfusion_v4/sd19_aug
    # the multi-drug model (one query per drug, one proximity channel per drug)
    python -m locusfusion_v4.train --drugs all \
        --split-file results/experiments/locusfusion_v4/splits/MULTIDRUG.csv \
        --aug-frac 0.5 --out results/experiments/locusfusion_v4/md_aug

Run it through sbatch (locusfusion_v4/sbatch.py), not in an interactive session.

Protocol = the lineage-split LocusFusion v3 cells
(results/experiments/lineage_split/sd19_*_unk__locusfusion_v2), unchanged:
the split file's TEST rows are held out, its folds 0-4 are the CV folds (fold 4
= Beijing); alpha class weights fit on the training rows; Adam at exp(-9);
early stopping on validation AUC (macro over drugs for the multi-drug model)
with best-weight restore; TEST = the best CV fold's model scored once.
Single-drug and multi-drug are the same code: a single-drug run is a one-column
label matrix. A single-drug run keeps the isolates with that drug's label; the
multi-drug run keeps every isolate with at least one, and each drug's loss and
metrics use only its labelled rows.

Writes <out>/<TAG>.json (TAG = the drug, or MULTIDRUG), <out>/<TAG>_test_predictions.csv,
<out>/<TAG>_synonymous_scan.csv.gz, and with --weights-dir the best fold's weights.

After training, two synonymous-robustness probes run on the best model (both
inference only, recorded under "probe" in the JSON):
  * test isolates + k synthetic synonymous changes (k = 1, 2): mean |delta logit|
    and mean delta P(R), per drug;
  * the pncA scan generalized: every synonymous candidate alone on an H37Rv
    background, P(R) per drug against the empty background's.
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

from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR  # noqa: E402
from training.core import EarlyStopper, masked_weighted_bce  # noqa: E402
from training.multidrug import _alpha_matrix  # noqa: E402
from training.multimodal import LR, _metrics, build_optimizer  # noqa: E402

from locusfusion_v4.augment import BATCH_KEYS, SynonymousAugmenter  # noqa: E402
from locusfusion_v4.binding_sites.build_distances import load_distances  # noqa: E402
from locusfusion_v4.layout import c_base, channel_names  # noqa: E402
from locusfusion_v4.model import V4_DEFAULTS, LocusFusionV4Net, proximity_table  # noqa: E402
from locusfusion_v4.tokenizer import ALL_DRUGS, LOCI_19, load_or_build  # noqa: E402

WORKSPACE = PROJECT.parent
MODEL_KEYS = ("feat", "coord", "valid", "site_pos", "n_occ")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drugs", nargs="+", required=True, help="one drug, several, or 'all'")
    sp = ap.add_mutually_exclusive_group(required=True)
    sp.add_argument("--split-dir", help="per-drug split files {DRUG}.csv (single-drug runs)")
    sp.add_argument("--split-file", help="one split file (New_ID, role, fold)")
    ap.add_argument("--out", required=True, help="results folder")
    ap.add_argument("--weights-dir", default=None, help="save the best fold's weights under here")
    ap.add_argument("--token-cache", default=str(WORKSPACE / "token_cache_v4"))
    ap.add_argument("--loci", nargs="+", default=LOCI_19)
    ap.add_argument("--genotype-dir", default=None,
                    help="aligned gene FASTAs; default REAL_GENOTYPE_DIR. The token cache keys on"
                         " this path, so a different directory gets its own cache")
    ap.add_argument("--regulatory-dir", default=None,
                    help="aligned promoter FASTAs; default REAL_REGULATORY_DIR")
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
    # tokens
    ap.add_argument("--max-variants", type=int, default=16)
    ap.add_argument("--no-distance", action="store_true", help="ablation: no site-proximity channels")
    ap.add_argument("--keep-residue", action="store_true",
                    help="also carry v3's 23-channel residue one-hot (v3's token + consequence"
                         " + distance)")
    ap.add_argument("--drop-synonymous", action="store_true",
                    help="ablation: synonymous changes are not tokens at all")
    # augmentation
    ap.add_argument("--aug-frac", type=float, default=0.0,
                    help="augmented copies per batch, as a fraction of it (0 = off)")
    ap.add_argument("--aug-k", type=int, default=2, help="synonymous changes per copy, ~U{1..k}")
    ap.add_argument("--aug-lambda", type=float, default=0.5, help="weight of the consistency term")
    ap.add_argument("--aug-locus-weight", choices=["uniform", "length"], default="uniform")
    ap.add_argument("--no-probe", action="store_true")
    # capacity (v3 defaults)
    for k, v in V4_DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    args = ap.parse_args(argv)
    args.drugs = ALL_DRUGS if [d.lower() for d in args.drugs] == ["all"] else [d.upper() for d in args.drugs]
    if args.drop_synonymous and args.aug_frac:
        ap.error("--drop-synonymous removes synonymous tokens; augmenting them back in is contradictory")
    if len(args.drugs) > 1 and args.split_dir:
        ap.error("a multi-drug run needs one --split-file")
    return args


def _p_resistant(logits):
    return 1.0 - torch.sigmoid(logits)


class Data:
    """The kept isolates' tokens, on the device, plus labels and split."""

    def __init__(self, ts, drugs, split_file, device):
        cols = [ts.drugs.index(d) for d in drugs]
        Y = ts.Y[:, cols]
        keep = np.nonzero((Y != -1).any(1))[0]
        self.ids = [ts.isolate_ids[i] for i in keep]
        self.Y = Y[keep]
        self.t = {k: torch.as_tensor(getattr(ts, k)[keep], device=device) for k in BATCH_KEYS}
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


def train_fold(model, data, alpha, tr, va, args, aug, fold):
    torch.manual_seed(args.seed + fold)
    np.random.seed(args.seed + fold)
    gen = torch.Generator(device=args.device).manual_seed(10_000 + args.seed + fold)
    opt = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopper(args.patience, args.min_delta, mode="max", min_epochs=args.min_epochs)
    alpha_t = torch.as_tensor(alpha, device=args.device)
    hist = {"train_loss": [], "train_consistency": [], "aug_inserted": [], "val_auc": []}
    for ep in range(args.epochs):
        model.train()
        perm = tr[np.random.permutation(len(tr))]
        tot = cons_tot = ins_tot = seen = 0.0
        for s in range(0, len(perm), args.batch_size):
            b = perm[s:s + args.batch_size]
            x = data.batch(b)
            a = alpha_t[torch.as_tensor(b, device=args.device)]
            m = int(round(args.aug_frac * len(b))) if aug is not None else 0
            if m:
                xa, ins = aug({k: x[k][:m] for k in BATCH_KEYS}, k_max=args.aug_k, gen=gen)
                x = {k: torch.cat([x[k], xa[k]]) for k in MODEL_KEYS}
                a = torch.cat([a, a[:m]])
            logits = model(**{k: x[k] for k in MODEL_KEYS})
            loss = masked_weighted_bce(logits, a)
            if m:
                w = (ins > 0).to(logits.dtype).unsqueeze(1)
                gap = (logits[len(b):] - logits[:m].detach()).abs() * w
                cons = gap.sum() / (w.sum() * logits.shape[1]).clamp_min(1)
                loss = loss + args.aug_lambda * cons
                cons_tot += float(cons) * len(b)
                ins_tot += float(ins.float().sum())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
            seen += len(b)
        hist["train_loss"].append(tot / seen)
        hist["train_consistency"].append(cons_tot / seen)
        hist["aug_inserted"].append(ins_tot)
        _, val = per_drug(data.Y[va], predict(model, data, va), args.drugs)
        hist["val_auc"].append(val)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  fold {fold} ep {ep + 1:3d} loss {tot / seen:.4f} cons {cons_tot / seen:.4f} "
                  f"val_auc {val:.4f}", flush=True)
        if val != val:
            continue
        if stopper.step(ep + 1, val, model):
            break
    stopper.restore(model)
    return stopper.best_epoch, hist


@torch.no_grad()
def probes(model, data, ts, aug, idx, drugs, device):
    """Synonymous robustness of a trained model (see the module docstring)."""
    model.eval()
    out = {"test_plus_k": {}}
    gen = torch.Generator(device=device).manual_seed(1234)
    for k in (1, 2):
        d_logit, d_pr, n = [], [], 0
        for s in range(0, len(idx), 512):
            b = data.batch(idx[s:s + 512])
            xa, ins = aug({kk: b[kk] for kk in BATCH_KEYS}, gen=gen, exact_k=k)
            ok = ins == k
            if not ok.any():
                continue
            l0 = model(**{kk: b[kk] for kk in MODEL_KEYS})[ok]
            l1 = model(**{kk: xa[kk] for kk in MODEL_KEYS})[ok]
            d_logit.append((l1 - l0).cpu())
            d_pr.append((_p_resistant(l1) - _p_resistant(l0)).cpu())
            n += int(ok.sum())
        dl, dp = torch.cat(d_logit), torch.cat(d_pr)
        out["test_plus_k"][str(k)] = {
            "n": n, **{d: {"mean_abs_dlogit": float(dl[:, j].abs().mean()),
                           "mean_dP_R": float(dp[:, j].mean()),
                           "mean_abs_dP_R": float(dp[:, j].abs().mean())}
                       for j, d in enumerate(drugs)}}
    # every candidate alone on an H37Rv background
    L, V = ts.feat.shape[1], ts.feat.shape[2]
    M = len(aug.locus)
    empty = {"feat": torch.zeros(1, L, V, ts.feat.shape[3], device=device),
             "coord": torch.zeros(1, L, V, device=device),
             "valid": torch.zeros(1, L, V, dtype=torch.bool, device=device),
             "site_pos": torch.full((1, L, V), -1, dtype=torch.int16, device=device),
             "n_occ": torch.zeros(1, L, dtype=torch.int32, device=device)}
    base = _p_resistant(model(**empty))[0].cpu().numpy()
    pr = []
    for s in range(0, M, 1024):
        c = torch.arange(s, min(s + 1024, M), device=device)
        x = {k: v.expand(len(c), *v.shape[1:]).clone() for k, v in empty.items()}
        r = torch.arange(len(c), device=device)
        x["feat"][r, aug.locus[c], 0] = aug.feat[c]
        x["coord"][r, aug.locus[c], 0] = aug.coord[c]
        x["valid"][r, aug.locus[c], 0] = True
        x["site_pos"][r, aug.locus[c], 0] = aug.site_pos[c].to(torch.int16)
        x["n_occ"][r, aug.locus[c]] = 1
        pr.append(_p_resistant(model(**x)).cpu().numpy())
    pr = np.concatenate(pr)
    scan = pd.DataFrame({"locus": [ts.loci[i] for i in aug.locus.cpu().numpy()],
                         "codon": aug.codon.cpu().numpy() + 1,
                         **{f"P_R_{d}": pr[:, j] for j, d in enumerate(drugs)}})
    out["h37rv_background"] = {
        d: {"P_R_empty": float(base[j]),
            "by_locus": {l: {"n": int(len(g)), "mean_P_R": float(g[f"P_R_{d}"].mean()),
                             "frac_above_0.5": float((g[f"P_R_{d}"] > 0.5).mean())}
                         for l, g in scan.groupby("locus")}}
        for j, d in enumerate(drugs)}
    return out, scan


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    multi = len(args.drugs) > 1
    tag = "MULTIDRUG" if multi else args.drugs[0]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_file = args.split_file or str(Path(args.split_dir) / f"{tag}.csv")
    print(f"[v4] {tag} drugs={args.drugs} device={args.device} out={out_dir}", flush=True)
    print(f"[v4] data: geno={REAL_GENOTYPE_DIR} pheno={REAL_PHENOTYPE_CSV} reg={REAL_REGULATORY_DIR}",
          flush=True)

    geno = args.genotype_dir or REAL_GENOTYPE_DIR
    reg = args.regulatory_dir or REAL_REGULATORY_DIR
    ts = load_or_build(args.token_cache, args.drugs, geno, REAL_PHENOTYPE_CSV,
                       reg, loci=args.loci, max_variants=args.max_variants,
                       drop_synonymous=args.drop_synonymous,
                       keep_residue=args.keep_residue)
    data = Data(ts, args.drugs, split_file, args.device)
    dist_drugs = [] if args.no_distance else list(args.drugs)
    prox = proximity_table(dist_drugs, ts.loci, load_distances())
    aug = (SynonymousAugmenter(ts.cand, args.device, args.aug_locus_weight)
           if (args.aug_frac > 0 or not args.no_probe) and len(ts.cand["locus"]) else None)
    capacity = {k: getattr(args, k) for k in V4_DEFAULTS}

    def build():
        return LocusFusionV4Net(ts.loci, ts.kept, args.drugs, prox=prox, dist_drugs=dist_drugs,
                                max_variants=args.max_variants,
                                c_base=c_base(args.keep_residue),
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
            print(f"[v4] {n_params:,} parameters, token width {model.c_tok}", flush=True)
        best_epoch, hist = train_fold(model, data, alpha, tr, va, args,
                                      aug if args.aug_frac > 0 else None, fold)
        per, macro = per_drug(data.Y[va], predict(model, data, va), args.drugs)
        rec = {"fold": fold, "best_epoch": best_epoch, "macro_auc": macro, "per_drug": per,
               "history": hist,
               "val_ids_sha1": hashlib.sha1("\n".join(sorted(data.ids[i] for i in va))
                                            .encode()).hexdigest()[:12]}
        if not multi:
            rec.update(per[tag])
        folds.append(rec)
        print(f"[v4] {tag} fold {fold}: val AUC {macro:.4f} (best epoch {best_epoch})", flush=True)
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
    print(f"[v4] {tag} CV macro AUC {np.mean(macro_cv):.4f} | TEST macro AUC {test_macro:.4f} "
          f"(best fold {best_fold})", flush=True)

    pred = pd.DataFrame({"New_ID": [data.ids[i] for i in data.test_idx]})
    for j, d in enumerate(args.drugs):
        pred[f"P_S_{d}"] = 1.0 / (1.0 + np.exp(-test_logits[:, j]))
        pred[f"y_{d}"] = data.Y[data.test_idx, j]
    pred.to_csv(out_dir / f"{tag}_test_predictions.csv", index=False)

    probe = None
    if not args.no_probe and aug is not None:
        probe, scan = probes(best_model, data, ts, aug, data.test_idx, args.drugs, args.device)
        scan.to_csv(out_dir / f"{tag}_synonymous_scan.csv.gz", index=False)

    weights = None
    if args.weights_dir:
        weights = Path(args.weights_dir) / out_dir.name / tag / "best.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": best_model.config, "kept": ts.kept.tolist(),
                    "state_dict": best_model.state_dict(), "fold": best_fold,
                    "token_config": ts.config, "args": vars(args)}, weights)

    result = {
        "arch": "locusfusion_v4", "tag": tag, "drugs": args.drugs, "multidrug": multi,
        "n_params": n_params, "n_isolates": len(data.ids),
        "token_width": c_base(args.keep_residue) + len(dist_drugs),
        "keep_residue": bool(args.keep_residue),
        "channels": channel_names(dist_drugs, args.keep_residue),
        "tokens": {k: v for k, v in ts.config.items() if k != "frames"},
        "capacity": capacity, "max_variants": args.max_variants,
        "distance": {"drugs": dist_drugs, "length_A": 10.0, "missing": -1.0},
        "augmentation": ({"frac": args.aug_frac, "k_max": args.aug_k, "lambda": args.aug_lambda,
                          "locus_weight": args.aug_locus_weight,
                          "n_candidates": int(len(ts.cand["locus"]))}
                         if args.aug_frac > 0 else None),
        "drop_synonymous": args.drop_synonymous,
        "protocol": {"epochs": args.epochs, "patience": args.patience,
                     "min_epochs": args.min_epochs, "min_delta": args.min_delta,
                     "batch_size": args.batch_size, "monitor": "auc",
                     "lr": LR if args.lr is None else args.lr,
                     "weight_decay": args.weight_decay, "seed": args.seed},
        "split_file": split_file, "cv_folds": folds,
        "cv_auc_mean": float(np.mean(macro_cv)) if macro_cv else float("nan"),
        "cv_auc_std": float(np.std(macro_cv)) if macro_cv else float("nan"),
        "test_model_fold": best_fold, "test_macro_auc": test_macro,
        "per_drug": cv_per, "probe": probe,
        "weights": str(weights) if weights else None,
        "seconds": round(time.time() - t0, 1), "host": os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    if not multi:
        result.update(drug=tag, test=test_per[tag])
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(f"[v4] wrote {out_dir / f'{tag}.json'} ({result['seconds']} s)", flush=True)
    return result


if __name__ == "__main__":
    main()

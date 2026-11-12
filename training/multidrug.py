"""
Multi-drug ("MD-CNN") training/eval engine for the stacked CNN.

Trains one ``models.stacked_cnn.MDCNNNet`` to predict ALL drugs at once,
following the same BIG-TB protocol as the single-drug
``training.multimodal.run_modal_cv`` but with a label MATRIX:

  * drop isolates with no phenotype for ANY drug,
  * random held-out split (seed 42) — multi-label, so not stratified — or a
    fixed split file (lineage-grouped MULTIDRUG.csv),
  * per-DRUG alpha class-weights (tb.alpha_mat per column) fit on the train split,
  * KFold(5) CV on the train split; each fold early-stops on the macro-mean
    per-drug validation AUC with best-weight restore,
  * masked multi-drug weighted BCE (core.masked_weighted_bce reduces over the
    drug axis and masks each isolate's missing drugs),
  * per-drug AND macro-mean AUC / AUC-PR reported for CV and the held-out TEST
    (the best-macro-val-AUC fold model), mirroring the MD-CNN reference.
"""
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split

import bigtb as tb
from models.stacked_cnn import MDCNNNet
from .checkpoint import RunCheckpointer, model_config
from .core import EarlyStopper, masked_weighted_bce
from .multimodal import (CV_SEED, DROPOUT, HIDDEN, LR, MIN_DELTA, MONITOR, OUT_BIAS,
                         PER_DRUG_HIDDEN, WEIGHT_DECAY, _batch, _set_seed,
                         build_optimizer)


def _predict(model, arrays, idx, batch_size, device):
    """(len(idx), n_drugs) sigmoid predictions."""
    idx = np.asarray(idx)
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            out.append(torch.sigmoid(model(_batch(arrays, b, device))).cpu().numpy())
    if out:
        return np.concatenate(out, axis=0)
    return np.zeros((0, model.n_drugs))


def _per_drug_metrics(Y_idx, P_idx, drugs):
    """Per-drug AUC / AUC-PR / operating point over each drug's non-missing rows,
    plus macro means over the drugs that had both classes present."""
    per, aucs, prs = {}, [], []
    for j, d in enumerate(drugs):
        yt, pr = Y_idx[:, j], P_idx[:, j]
        valid = yt != -1
        ytv, prv = yt[valid], pr[valid]
        m = {"n": int(valid.sum()), "n_R": int((ytv == 0).sum()),
             "n_S": int((ytv == 1).sum())}
        if len(np.unique(ytv)) < 2:
            m.update(auc=float("nan"), auc_pr=float("nan"),
                     sens=float("nan"), spec=float("nan"))
        else:
            m["auc"] = float(roc_auc_score(ytv, prv))
            m["auc_pr"] = float(average_precision_score(1 - ytv, 1 - prv))  # R = positive
            th = tb.get_threshold_val(ytv, prv)
            m.update(sens=float(th["sens"]), spec=float(th["spec"]))
            aucs.append(m["auc"])
            prs.append(m["auc_pr"])
        per[d] = m
    macro_auc = float(np.mean(aucs)) if aucs else float("nan")
    macro_pr = float(np.mean(prs)) if prs else float("nan")
    return per, macro_auc, macro_pr


def _macro_val_auc(model, arrays, Y, idx, batch_size, device):
    P = _predict(model, arrays, idx, batch_size, device)
    _, macro_auc, _ = _per_drug_metrics(Y[idx], P, [str(i) for i in range(Y.shape[1])])
    return macro_auc


def _train(model, arrays, alpha, Y, tr_idx, va_idx, epochs, batch_size, device,
           seed, patience=15, min_delta=1e-4, min_epochs=0):
    """Mini-batched training with early stopping on the macro val AUC (mean
    per-drug val AUC, maximised). Returns (model, best_epoch, history);
    `history` is the per-epoch train loss + val AUC (curves.save_curves)."""
    _set_seed(seed)
    opt = build_optimizer(model)
    tr_idx = np.asarray(tr_idx)
    stopper = EarlyStopper(patience, min_delta, mode="max", min_epochs=min_epochs)
    history = {"train_loss": [], "val_auc": []}
    for ep in range(epochs):
        model.train()
        perm = tr_idx[np.random.permutation(len(tr_idx))]
        run_loss, seen = 0.0, 0
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            ab = torch.from_numpy(alpha[b]).to(device, non_blocking=True)
            opt.zero_grad()
            loss = masked_weighted_bce(model(_batch(arrays, b, device)), ab)
            loss.backward()
            opt.step()
            run_loss += float(loss) * len(b)
            seen += len(b)
        history["train_loss"].append(run_loss / max(seen, 1))
        metric = _macro_val_auc(model, arrays, Y, va_idx, batch_size, device)
        history["val_auc"].append(metric)  # NaN keeps epochs aligned
        if metric != metric:      # NaN (no scorable drug): skip epoch
            continue
        if stopper.step(ep + 1, metric, model):
            break
    stopper.restore(model)
    return model, stopper.best_epoch, history


def _alpha_matrix(Y, train_idx, n_drugs):
    """Per-drug alpha (N, n_drugs): tb.alpha_mat fit per column on the train
    split, scattered back so val/test rows stay 0 (never touched by the loss).
    A drug with no valid train label stays all-zero (masked out)."""
    n = Y.shape[0]
    alpha = np.zeros((n, n_drugs), dtype=np.float32)
    for j in range(n_drugs):
        col = Y[train_idx, j]
        if int((col == 0).sum()) + int((col == 1).sum()) == 0:
            continue                       # all missing on train -> masked
        alpha[train_idx, j] = tb.alpha_mat(
            col.reshape(-1, 1), None, weight=1.0).astype(np.float32).ravel()
    return alpha


def run_multidrug_cv(data, epochs=60, n_splits=5, batch_size=128, device="cpu",
                     seed=0, patience=15, min_epochs=0, run_name=None,
                     save_weights="best", weights_dir=None, data_config=None,
                     split_file=None):
    """Train/eval the stacked CNN on a MultiDrugData bundle. Returns a result
    dict with per-drug and macro CV/TEST metrics.

    split_file : optional CSV (New_ID, role in {train,test}, fold 0..k-1) that
                fixes the held-out test split AND the CV folds, e.g. the
                lineage-grouped MULTIDRUG.csv. Every isolate kept after the
                all-missing filter must appear in it (extra rows are ignored).
                Overrides the random split and KFold."""
    t0 = time.time()
    drugs = data.drugs
    n_drugs = len(drugs)
    tag = data.modality_tag()
    specs = data.branch_specs()
    encoder_types = ["cnn"] * len(data.blocks)

    # --- drop isolates with NO phenotype for any drug -------------------------
    keep = np.nonzero((data.Y != -1).any(axis=1))[0]
    Y = data.Y[keep]
    arrays = [a[keep] for a in data.arrays()]
    n = len(keep)
    print(f"[multidrug/{tag}] arch=mdcnn drugs={n_drugs} blocks={len(data.blocks)} "
          f"specs~{specs[:3]}{'...' if len(specs) > 3 else ''} "
          f"n_valid={n} (dropped {data.n - n} all-missing)", flush=True)

    # --- held-out split: random (multi-label -> not stratified), seed 42, or
    # fixed by split_file (test split + CV folds) -----------------------------
    kept_ids = [str(data.isolate_ids[i]) for i in keep]
    if split_file is None:
        train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=42)
        split_folds = None
    else:
        sp = pd.read_csv(split_file, dtype={"New_ID": str}).set_index("New_ID")
        missing = [i for i in kept_ids if i not in sp.index]
        if missing:
            raise ValueError(f"{split_file}: {len(missing)} of {n} isolates have no split "
                             f"(e.g. {missing[:3]})")
        sp = sp.loc[kept_ids]
        test_idx = np.nonzero((sp.role == "test").values)[0]
        train_idx = np.nonzero((sp.role == "train").values)[0]
        fold_of = sp.fold.values[train_idx]
        split_folds = [(np.nonzero(fold_of != k)[0], np.nonzero(fold_of == k)[0])
                       for k in sorted(set(fold_of))]
        assert len(split_folds) == n_splits, (len(split_folds), n_splits)
        print(f"[multidrug/{tag}] split from {split_file}: train={len(train_idx)} "
              f"test={len(test_idx)} folds={[len(v) for _, v in split_folds]}", flush=True)

    alpha = _alpha_matrix(Y, train_idx, n_drugs)
    head = {"hidden": HIDDEN, "dropout": DROPOUT, "per_drug_hidden": PER_DRUG_HIDDEN}
    ckpt = RunCheckpointer(run_name, f"multidrug__{tag}", mode=save_weights,
                           weights_dir=weights_dir)
    kept_isolates = [data.isolate_ids[i] for i in keep]

    fold_splits = (split_folds if split_folds is not None else
                   KFold(n_splits=n_splits, shuffle=True, random_state=CV_SEED).split(train_idx))
    folds, best_macro, best_model, best_fold = [], -np.inf, None, None
    for fold, (tr, va) in enumerate(fold_splits):
        model = MDCNNNet.from_blocks(data.blocks, drug_names=drugs, hidden=HIDDEN).to(device)
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[multidrug/{tag}] mdcnn: {n_params:,} parameters", flush=True)
        _, best_epoch, history = _train(model, arrays, alpha, Y, train_idx[tr], train_idx[va],
                                        epochs, batch_size, device, seed + fold,
                                        patience, MIN_DELTA, min_epochs)
        P = _predict(model, arrays, train_idx[va], batch_size, device)
        per, macro_auc, macro_pr = _per_drug_metrics(Y[train_idx[va]], P, drugs)
        folds.append({"fold": fold, "best_epoch": best_epoch, "macro_auc": macro_auc,
                      "macro_auc_pr": macro_pr, "per_drug": per,
                      "history": history})   # per-epoch curves (training.curves)
        # `model` here already carries the best-epoch weights (stopper.restore)
        ckpt.add_fold(fold, model, best_epoch, macro_auc)
        print(f"[multidrug/{tag}] CV fold {fold}: macro-AUC={macro_auc:.4f} "
              f"macro-AUC_PR={macro_pr:.4f} (best_epoch={best_epoch})", flush=True)
        if macro_auc == macro_auc and macro_auc > best_macro:
            best_macro, best_model, best_fold = macro_auc, model, fold

    if best_model is None:
        best_model = model

    # --- held-out TEST: best-macro-val-AUC fold model, scored once ------------
    P_test = _predict(best_model, arrays, test_idx, batch_size, device)
    test_per, test_macro_auc, test_macro_pr = _per_drug_metrics(Y[test_idx], P_test, drugs)

    macro_aucs = [f["macro_auc"] for f in folds if f["macro_auc"] == f["macro_auc"]]
    cv_macro_mean = float(np.mean(macro_aucs)) if macro_aucs else float("nan")
    cv_macro_std = float(np.std(macro_aucs)) if macro_aucs else float("nan")

    # per-drug CV mean AUC across folds (each drug averaged over folds it scored)
    cv_per_drug = {}
    for d in drugs:
        vals = [f["per_drug"][d]["auc"] for f in folds
                if f["per_drug"][d]["auc"] == f["per_drug"][d]["auc"]]
        cv_per_drug[d] = float(np.mean(vals)) if vals else float("nan")

    # --- persist the weights + everything needed to rebuild them --------------
    ckpt_cfg = {
        "run_name": run_name, "scope": "joint", "tag": tag, "drug": None,
        "model": model_config(blocks=data.blocks, encoder_types=encoder_types,
                              drug_names=drugs, out_bias=OUT_BIAS, head=head,
                              n_params=n_params),
        "data": {**(data_config or {}),
                 "modalities_used": data.modalities,
                 "modalities_requested": data.requested,
                 "dropped": data.dropped,
                 "loci": data.loci, "gene_order": data.gene_order,
                 "n_isolates": data.n, "n_valid": n,
                 "dropped_all_missing": data.n - n},
        "split": {**({"test_size": 0.2, "split_seed": 42, "stratified": False,
                     "kfold": "KFold", "kfold_seed": CV_SEED, "kfold_shuffle": True}
                    if split_file is None else {}),
                  "n_splits": n_splits,
                  "split_file": None if split_file is None else str(split_file),
                  "isolate_order": "isolates.txt (post all-missing filter)",
                  "test_isolate_ids": [kept_isolates[i] for i in test_idx]},
        "training": {"epochs": epochs, "batch_size": batch_size, "seed": seed,
                     "lr": LR, "weight_decay": WEIGHT_DECAY,
                     "patience": patience, "min_delta": MIN_DELTA,
                     "monitor": MONITOR, "min_epochs": min_epochs},
    }
    weights_path = ckpt.write(ckpt_cfg, best_fold, isolate_ids=kept_isolates)

    print(f"[multidrug/{tag}] CV macro-AUC = {cv_macro_mean:.4f} +/- {cv_macro_std:.4f} "
          f"| TEST macro-AUC = {test_macro_auc:.4f} (best fold {best_fold})", flush=True)
    for d in drugs:
        print(f"    {d:14s} CV={cv_per_drug[d]:.3f}  TEST={test_per[d]['auc']:.3f} "
              f"(R={test_per[d]['n_R']} S={test_per[d]['n_S']})", flush=True)

    return {
        "drugs": drugs, "modalities": data.modalities, "tag": tag,
        "loci": data.gene_order, "blocks": [b.name for b in data.blocks],
        "branch_specs": specs, "encoder_types": encoder_types,
        "arch": "mdcnn", "n_params": n_params,
        "dropped": data.dropped, "n_isolates": data.n, "n_valid": n,
        "epochs": epochs, "batch_size": batch_size, "n_splits": n_splits,
        "patience": patience, "min_delta": MIN_DELTA, "out_bias": OUT_BIAS,
        "monitor": MONITOR, "min_epochs": min_epochs,
        "lr": LR, "weight_decay": WEIGHT_DECAY,
        "hidden": HIDDEN, "dropout": DROPOUT, "per_drug_hidden": PER_DRUG_HIDDEN,
        "split_file": None if split_file is None else str(split_file),
        "weights_dir": str(weights_path) if weights_path else None,
        "save_weights": save_weights,
        "cv_macro_auc_mean": cv_macro_mean, "cv_macro_auc_std": cv_macro_std,
        "cv_per_drug_auc": cv_per_drug,
        "test_macro_auc": test_macro_auc, "test_macro_auc_pr": test_macro_pr,
        "test_model_fold": best_fold,
        "test_per_drug": test_per,
        "cv_folds": folds,
        "seconds": round(time.time() - t0, 1),
    }

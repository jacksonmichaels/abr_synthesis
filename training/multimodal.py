"""
Single-drug training/eval engine for the stacked CNN (models.stacked_cnn.MDCNNNet).

Given a ``datasets.loader.DrugData`` bundle (one block per locus/region), runs
BIG-TB's SD-CNN protocol (run_SDCNN_ccp_crossval / _assess):

  * drop missing-phenotype isolates before anything else (TODO #1),
  * stratified held-out split (test_size=0.2, seed 42) (TODO #5), or a fixed
    split file (lineage-grouped splits),
  * alpha class-weights (tb.alpha_mat) fit on the training split only (TODO #2),
  * stratified 5-fold CV on the training portion (seed 42) with per-fold
    early stopping on val AUC + best-weight restore (TODO #4),
  * masked weighted BCE loss, tb.get_threshold_val operating point,
  * TEST metric = the best-val-AUC CV fold model scored once on the held-out
    split (baseline's sd-cnn_model_best.h5).

Early stopping deviates from the strict baseline (monitor='val_loss',
out_bias=log-odds), which collapsed to the all-susceptible majority on the
imbalanced drugs (MOXI/ETO): val_loss bottoms out at the degenerate solution,
so it stopped in ~1-9 epochs at CV AUC ~0.54. Monitoring val AUC with
patience 15 and PyTorch's default output bias recovers CV AUC 0.59->0.85 on
MOXIFLOXACIN DNA and a usable operating point.

CV uses StratifiedKFold(seed=42) where the baseline crossval script used plain
KFold(seed=1). Mini-batched throughout (real data doesn't fit one GPU).
"""
import hashlib
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import bigtb as tb
from models.stacked_cnn import MDCNNNet
from .checkpoint import RunCheckpointer, model_config
from .core import EarlyStopper, masked_weighted_bce

LR = float(np.exp(-9.0))  # matches BIG-TB SD-CNN (run_SDCNN_ccp_crossval)


def build_optimizer(model, lr=None, weight_decay=0.0):
    """Adam at the BIG-TB learning rate by default; ``weight_decay > 0``
    switches to AdamW (Adam's ``weight_decay`` couples the penalty into the
    adaptive step). The stacked CNN uses the defaults."""
    lr = LR if lr is None else float(lr)
    if weight_decay:
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    return optim.Adam(model.parameters(), lr=lr)


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batch(arrays, idx, device):
    # .float() casts uint8 one-hot storage (the memory-lean layout the multi-drug
    # loader uses for its many locus blocks) up to float for the conv branches;
    # it is a no-op for the float32 blocks the single-drug path already uses.
    return [torch.from_numpy(a[idx]).to(device, non_blocking=True).float() for a in arrays]


def _val_auc(model, arrays, y, idx, batch_size, device):
    """Val-fold ROC-AUC on the resistant/susceptible ranking (NaN if the fold is
    single-class) — the early-stopping metric."""
    idx = np.asarray(idx)
    yt = y[idx]
    valid = yt != -1
    if valid.sum() == 0 or len(np.unique(yt[valid])) < 2:
        return float("nan")
    pred = _predict(model, arrays, idx, batch_size, device)
    return float(roc_auc_score(yt[valid], pred[valid]))


def _train(model, arrays, alpha, y, tr_idx, va_idx, epochs, batch_size, device, seed,
           patience, min_delta=1e-4, min_epochs=0):
    """Mini-batched training with early stopping on val AUC + best-weight
    restore (TODO #4). ``min_epochs`` holds patience at zero for that many
    epochs (see EarlyStopper). Returns (model, best_epoch, history); `history`
    holds the per-epoch train loss and val AUC (curves.save_curves plots them)."""
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
            xb = _batch(arrays, b, device)
            ab = torch.from_numpy(alpha[b]).to(device, non_blocking=True)
            opt.zero_grad()
            loss = masked_weighted_bce(model(xb), ab)
            loss.backward()
            opt.step()
            run_loss += float(loss) * len(b)
            seen += len(b)
        history["train_loss"].append(run_loss / max(seen, 1))
        val_metric = _val_auc(model, arrays, y, va_idx, batch_size, device)
        history["val_auc"].append(val_metric)  # NaN keeps epochs aligned
        if val_metric != val_metric:  # NaN fold: skip, don't count as bad
            continue
        if stopper.step(ep + 1, val_metric, model):
            break
    stopper.restore(model)   # restore_best_weights=True: best epoch, always
    return model, stopper.best_epoch, history


def _predict(model, arrays, idx, batch_size, device):
    idx = np.asarray(idx)
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            out.append(torch.sigmoid(model(_batch(arrays, b, device))).cpu().numpy().reshape(-1))
    return np.concatenate(out)


def _metrics(y_true, pred):
    """AUC / AUC_PR / operating point on the valid (non-missing) subset."""
    valid = y_true != -1
    yt, pr = y_true[valid], pred[valid]
    m = {"n": int(valid.sum()), "n_R": int((yt == 0).sum()), "n_S": int((yt == 1).sum())}
    if m["n_R"] == 0 or m["n_S"] == 0:
        m.update(auc=float("nan"), auc_pr=float("nan"),
                 sens=float("nan"), spec=float("nan"), threshold=float("nan"))
        return m
    m["auc"] = float(roc_auc_score(yt, pr))
    m["auc_pr"] = float(average_precision_score(1 - yt, 1 - pr))  # resistant = positive
    th = tb.get_threshold_val(yt, pr)
    m.update(sens=float(th["sens"]), spec=float(th["spec"]), threshold=float(th["threshold"]))
    return m


# Fixed protocol values, recorded in every result and checkpoint config.
HIDDEN, DROPOUT, PER_DRUG_HIDDEN, OUT_BIAS = 256, 0.0, 0, None
MONITOR, MIN_DELTA, WEIGHT_DECAY, CV_SEED = "auc", 1e-4, 0.0, 42


def run_modal_cv(data, epochs=60, n_splits=5, batch_size=128, device="cpu", seed=0,
                 patience=15, min_epochs=0, run_name=None, save_weights="best",
                 weights_dir=None, data_config=None, split_file=None):
    """Train/eval the stacked CNN on a DrugData bundle, following BIG-TB's
    SD-CNN protocol (see the module docstring). Returns a result dict.

    split_file : optional CSV (New_ID, role in {train,test}, fold 0..k-1) that
                 fixes the held-out test split AND the CV folds, e.g. the
                 held-out-lineage splits of splits/build_holdout_splits.py.
                 Every isolate kept after the missing-phenotype filter must
                 appear in it. Overrides the stratified split.
    """
    t0 = time.time()
    head = {"hidden": HIDDEN, "dropout": DROPOUT, "per_drug_hidden": PER_DRUG_HIDDEN}
    drug, tag = data.drug, data.modality_tag()
    ckpt = RunCheckpointer(run_name, f"{drug}__{tag}", mode=save_weights,
                           weights_dir=weights_dir)
    specs = data.branch_specs()
    encoder_types = ["cnn"] * len(data.blocks)
    enc_by_modality = {b.modality: "cnn" for b in data.blocks}

    # --- #1: drop missing-phenotype isolates BEFORE splitting -----------------
    # Baseline run_SDCNN_ccp_crossval applies mask_valid=(y!=-1) before CV.
    # Carrying -1 rows through (alpha 0, so masked out of the loss) still filled
    # ~85% of every MOXI batch with dead padding, shrinking the effective
    # gradient — the primary source of an earlier ~0.1 AUC gap.
    keep = np.nonzero(data.y != -1)[0]
    y = data.y[keep]
    arrays = [a[keep] for a in data.arrays()]
    n = len(keep)
    assert (y != -1).all(), "missing phenotypes leaked past the filter"
    n_R, n_S = int((y == 0).sum()), int((y == 1).sum())
    print(f"[{drug}/{tag}] arch=mdcnn blocks={[b.name for b in data.blocks]} "
          f"specs={specs} n_valid={n} R={n_R} S={n_S} "
          f"(dropped {data.n - n} missing)", flush=True)

    # --- #5: stratified held-out split, seed 42 (paper's seed; also matches the
    # baseline train_test_split random_state=42, and adds stratification as the
    # baseline assess script does) --------------------------------------------
    kept_ids = [str(data.isolate_ids[i]) for i in keep]
    if split_file is None:
        train_idx, test_idx = train_test_split(
            np.arange(n), test_size=0.2, random_state=42, stratify=y)
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
        print(f"[{drug}/{tag}] split from {split_file}: train={len(train_idx)} "
              f"test={len(test_idx)} folds={[len(v) for _, v in split_folds]}", flush=True)

    # --- #2: alpha fit on the TRAINING split only -----------------------------
    # tb.alpha_mat's magnitude is n_R/(n_R+n_S) over the rows it sees; the
    # baseline fits it on train_df. Scattering into a full-length array keeps
    # global indices valid; test rows keep 0 (never touched by the loss).
    alpha = np.zeros((n, 1), dtype=np.float32)
    alpha[train_idx] = tb.alpha_mat(
        y[train_idx].reshape(-1, 1), None, weight=1.0).astype(np.float32)

    # --- #5: stratified 5-fold CV on the training split -----------------------
    if split_folds is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CV_SEED)
        split_folds = list(skf.split(train_idx, y[train_idx]))
    folds = []
    best_auc, best_model, best_fold = -np.inf, None, None
    for fold, (tr, va) in enumerate(split_folds):
        model = MDCNNNet.from_blocks(data.blocks, n_drugs=1, hidden=HIDDEN).to(device)
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[{drug}/{tag}] mdcnn: {n_params:,} parameters", flush=True)
        _, best_epoch, history = _train(model, arrays, alpha, y, train_idx[tr],
                                        train_idx[va], epochs, batch_size, device,
                                        seed + fold, patience, MIN_DELTA,
                                        min_epochs=min_epochs)
        m = _metrics(y[train_idx[va]], _predict(model, arrays, train_idx[va], batch_size, device))
        m["fold"] = fold
        m["best_epoch"] = best_epoch
        m["history"] = history       # per-epoch train loss + val metric (training.curves)
        # which isolates this fold held out, compactly: two results are fold-paired
        # only if these match, so a paired test can check rather than assume it
        m["val_ids_sha1"] = hashlib.sha1("\n".join(sorted(
            data.isolate_ids[keep[i]] for i in train_idx[va])).encode()).hexdigest()[:12]
        folds.append(m)
        # `model` already carries the best-epoch weights (stopper.restore)
        ckpt.add_fold(fold, model, best_epoch, m["auc"])
        print(f"[{drug}/{tag}] CV fold {fold}: AUC={m['auc']:.4f} AUC_PR={m['auc_pr']:.4f} "
              f"sens={m['sens']:.3f} spec={m['spec']:.3f} (n_val={m['n']}, "
              f"best_epoch={best_epoch})", flush=True)
        # baseline sd-cnn_model_best.h5 = the fold with the highest val AUC
        if m["auc"] == m["auc"] and m["auc"] > best_auc:
            best_auc, best_model, best_fold = m["auc"], model, fold

    # --- held-out TEST: score the best CV-fold model once (baseline protocol) --
    if best_model is None:            # every fold degenerate (single-class val)
        best_model = model
    test_m = _metrics(y[test_idx], _predict(best_model, arrays, test_idx, batch_size, device))

    aucs = [f["auc"] for f in folds if f["auc"] == f["auc"]]
    prs = [f["auc_pr"] for f in folds if f["auc_pr"] == f["auc_pr"]]
    cv_mean = float(np.mean(aucs)) if aucs else float("nan")
    cv_std = float(np.std(aucs)) if aucs else float("nan")

    # --- persist the weights + everything needed to rebuild them --------------
    kept_isolates = [data.isolate_ids[i] for i in keep]
    weights_path = ckpt.write({
        "run_name": run_name, "scope": "single", "tag": tag, "drug": drug,
        "model": model_config(blocks=data.blocks, encoder_types=encoder_types,
                              drug_names=[drug], out_bias=OUT_BIAS, head=head,
                              n_params=n_params),
        "data": {**(data_config or {}),
                 "modalities_used": data.modalities,
                 "modalities_requested": data.requested,
                 "dropped": data.dropped,
                 "loci": data.loci, "gene_order": data.gene_order,
                 "n_isolates": data.n, "n_valid": n,
                 "n_resistant": n_R, "n_susceptible": n_S},
        "split": {"test_size": 0.2, "split_seed": 42, "stratified": True,
                  "n_splits": n_splits, "kfold": "StratifiedKFold",
                  "kfold_seed": CV_SEED, "kfold_shuffle": True,
                  "split_file": None if split_file is None else str(split_file),
                  "isolate_order": "isolates.txt (post missing-phenotype filter)",
                  "test_isolate_ids": [kept_isolates[i] for i in test_idx]},
        "training": {"epochs": epochs, "batch_size": batch_size, "seed": seed,
                     "lr": LR, "weight_decay": WEIGHT_DECAY,
                     "patience": patience, "min_delta": MIN_DELTA,
                     "monitor": MONITOR, "min_epochs": min_epochs},
    }, best_fold, isolate_ids=kept_isolates)

    # --- #9: surface the per-fold spread, and label CV vs TEST distinctly -----
    print(f"[{drug}/{tag}] CV  AUC = {cv_mean:.4f} +/- {cv_std:.4f} "
          f"(mean +/- std over {len(aucs)} folds)", flush=True)
    print(f"[{drug}/{tag}] TEST AUC = {test_m['auc']:.4f} AUC_PR={test_m['auc_pr']:.4f} "
          f"sens={test_m['sens']:.3f} spec={test_m['spec']:.3f} "
          f"(best CV fold {best_fold}, n_test={test_m['n']})", flush=True)

    return {
        "drug": drug, "modalities": data.modalities, "tag": tag,
        "genes": data.gene_order, "blocks": [b.name for b in data.blocks],
        "branch_specs": specs, "encoders": enc_by_modality,
        "encoder_types": encoder_types, "dropped": data.dropped,
        "cv_seed": CV_SEED,
        "arch": "mdcnn", "n_params": n_params,
        "n_isolates": data.n, "n_valid": n,
        "n_resistant": n_R, "n_susceptible": n_S,
        "epochs": epochs, "batch_size": batch_size,
        "patience": patience, "min_delta": MIN_DELTA, "out_bias": OUT_BIAS,
        "monitor": MONITOR, "min_epochs": min_epochs,
        "lr": LR, "weight_decay": WEIGHT_DECAY,
        "hidden": HIDDEN, "dropout": DROPOUT, "per_drug_hidden": PER_DRUG_HIDDEN,
        "weights_dir": str(weights_path) if weights_path else None,
        "save_weights": save_weights,
        "cv_folds": folds,
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        "cv_auc_pr_mean": float(np.mean(prs)) if prs else float("nan"),
        "test_model_fold": best_fold,
        "split_file": None if split_file is None else str(split_file),
        "test": test_m,
        "seconds": round(time.time() - t0, 1),
    }

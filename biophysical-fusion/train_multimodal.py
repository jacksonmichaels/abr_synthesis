"""
Generic mini-batched training/eval engine for any set of modalities.

Given a ``datasets.DrugData`` bundle (any combination of DNA / protein /
biophysical / regulatory blocks), builds a ``models.MultiModalNet`` with one
branch per block and runs BIG-TB's SD-CNN protocol
(run_SDCNN_ccp_crossval / _assess):

  * drop missing-phenotype isolates before anything else (TODO #1),
  * stratified held-out split (test_size=0.2, seed 42) (TODO #5),
  * alpha class-weights (tb.alpha_mat) fit on the training split only (TODO #2),
  * output-bias init to the training-split log-odds (TODO #6),
  * stratified 5-fold CV on the training portion (seed 42) with per-fold
    val-loss early stopping + best-weight restore (TODO #4),
  * masked weighted BCE loss, tb.get_threshold_val operating point,
  * TEST metric = the best-val-AUC CV fold model scored once on the held-out
    split (baseline's sd-cnn_model_best.h5).

Everything is mini-batched (real data doesn't fit one GPU). This generalizes
eval_dna_cnn.py (DNA-only) to arbitrary modality sets; the two share the same
metric conventions. Optional TensorBoard logging mirrors eval_dna_cnn's layout.

NB: two knobs deviate from the baseline *crossval* script in favour of the
paper's stated protocol — CV uses StratifiedKFold(seed=42) where the baseline
used plain KFold(seed=1), and the output bias is initialised to the log-odds
where Keras leaves it at 0. Neither changes AUC ranking materially; see the
inline notes at each step.
"""
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from bigtb_ref import tb
from models import MultiModalNet
from train import EarlyStopper, masked_weighted_bce

LR = float(np.exp(-9.0))  # matches BIG-TB SD-CNN / train.run_cv / eval_dna_cnn


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batch(arrays, idx, device):
    return [torch.from_numpy(a[idx]).to(device, non_blocking=True) for a in arrays]


def _eval_loss(model, arrays, alpha, idx, batch_size, device):
    """Mean masked-weighted BCE over `idx` (the validation fold). This is the
    quantity BIG-TB's EarlyStopping monitors as val_loss."""
    idx = np.asarray(idx)
    model.eval()
    tot, seen = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            ab = torch.from_numpy(alpha[b]).to(device, non_blocking=True)
            tot += float(masked_weighted_bce(model(_batch(arrays, b, device)), ab)) * len(b)
            seen += len(b)
    return tot / max(seen, 1)


def _train(model, arrays, alpha, tr_idx, va_idx, epochs, batch_size, device, seed,
           writer=None, patience=5, min_delta=1e-4):
    """Mini-batched training with val-loss early stopping + best-weight restore
    (TODO #4). `va_idx=None` disables early stopping (fixed epoch loop). Returns
    (model, best_epoch); best_epoch is None when no validation fold is given."""
    _set_seed(seed)
    opt = optim.Adam(model.parameters(), lr=LR)
    tr_idx = np.asarray(tr_idx)
    stopper = EarlyStopper(patience, min_delta) if va_idx is not None else None
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
        if writer is not None:
            writer.add_scalar("loss/train", run_loss / max(seen, 1), ep + 1)
        if stopper is not None:
            val_loss = _eval_loss(model, arrays, alpha, va_idx, batch_size, device)
            if writer is not None:
                writer.add_scalar("loss/val", val_loss, ep + 1)
            if stopper.step(ep + 1, val_loss, model):
                break
    if stopper is not None:
        stopper.restore(model)   # restore_best_weights=True: best epoch, always
    if writer is not None:
        writer.flush()
    return model, (stopper.best_epoch if stopper is not None else None)


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


def _log_final(writer, prefix, m, step):
    for k in ("auc", "auc_pr", "sens", "spec"):
        if m[k] == m[k]:  # skip NaN
            writer.add_scalar(f"{prefix}/{k}", m[k], step)
    writer.flush()


def _new_writer(tb_dir, *parts):
    if tb_dir is None:
        return None
    from torch.utils.tensorboard import SummaryWriter
    path = tb_dir
    for p in parts:
        path = path / p
    return SummaryWriter(path, flush_secs=10)


def run_modal_cv(data, epochs=60, n_splits=5, batch_size=128, device="cpu",
                 tb_dir=None, seed=0, branch_models=None, default_encoder="cnn",
                 patience=5, min_delta=1e-4):
    """Train/eval a MultiModalNet on a DrugData bundle, following BIG-TB's
    SD-CNN protocol (run_SDCNN_ccp_crossval / _assess). Returns a result dict.

    Protocol (see the per-step TODO notes inline):
      #1 drop missing-phenotype isolates up front,
      #2 alpha class-weights fit on the training split only,
      #5 stratified held-out split (seed 42) + stratified 5-fold CV,
      #6 output-bias init to the training-split log-odds,
      #4 per-fold val-loss early stopping with best-weight restore,
         TEST = the best CV-fold model scored once on the held-out split
         (baseline uses sd-cnn_model_best.h5, the best-val-AUC fold).

    branch_models    : {modality: encoder_type} — e.g. {'protein': 'transformer'}.
                       Each block inherits its modality's encoder; modalities not
                       named fall back to default_encoder.
    default_encoder  : encoder for any modality not in branch_models (default cnn).
    """
    t0 = time.time()
    branch_models = branch_models or {}
    drug, tag = data.drug, data.modality_tag()
    specs = data.branch_specs()
    encoder_types = [branch_models.get(b.modality, default_encoder) for b in data.blocks]
    # per-modality view for logging/results (all blocks of a modality share a type)
    enc_by_modality = {b.modality: e for b, e in zip(data.blocks, encoder_types)}

    # --- #1: drop missing-phenotype isolates BEFORE splitting -----------------
    # Baseline run_SDCNN_ccp_crossval applies mask_valid=(y!=-1) before CV. The
    # earlier code carried -1 rows all the way through: their alpha is 0 so the
    # loss masks them out, but they still filled ~85% of every MOXI batch as
    # dead padding (2,868 valid of 17,942) — shrinking the effective gradient
    # and diluting each minibatch. This was the primary source of the ~0.1 AUC
    # gap. We drop them from y and every feature block up front.
    keep = np.nonzero(data.y != -1)[0]
    y = data.y[keep]
    arrays = [a[keep] for a in data.arrays()]
    n = len(keep)
    assert (y != -1).all(), "missing phenotypes leaked past the filter"
    n_R, n_S = int((y == 0).sum()), int((y == 1).sum())
    print(f"[{drug}/{tag}] blocks={[b.name for b in data.blocks]} specs={specs} "
          f"encoders={enc_by_modality} n_valid={n} R={n_R} S={n_S} "
          f"(dropped {data.n - n} missing)", flush=True)

    # --- #5: stratified held-out split, seed 42 (paper's seed; also matches the
    # baseline train_test_split random_state=42, and adds stratification as the
    # baseline assess script does) --------------------------------------------
    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=0.2, random_state=42, stratify=y)

    # --- #2: alpha fit on the TRAINING split only -----------------------------
    # tb.alpha_mat's magnitude is n_R/(n_R+n_S) over the rows it sees; the
    # baseline fits it on train_df. Scattering into a full-length array keeps
    # global indices valid; test rows keep 0 (never touched by the loss). With
    # missing rows already dropped, np.unique(alpha[train_idx]) is {-a, +a}
    # (a = R/(R+S) on train), e.g. ~{-0.135, +0.135} for MOXIFLOXACIN.
    alpha = np.zeros((n, 1), dtype=np.float32)
    alpha[train_idx] = tb.alpha_mat(
        y[train_idx].reshape(-1, 1), None, weight=1.0).astype(np.float32)

    # --- #6: output-bias init = log(n_pos / n_neg) on the training split ------
    # The sigmoid target is y==1 (susceptible), so n_pos=#S, n_neg=#R. For MOXI
    # this is +log(2480/388) ≈ +1.855 (a positive bias: susceptible is the
    # majority class). A constant bias shift can't change AUC ranking; it only
    # calibrates the initial loss / operating point.
    tr_pos = int((y[train_idx] == 1).sum())
    tr_neg = int((y[train_idx] == 0).sum())
    out_bias = float(np.log(tr_pos / tr_neg)) if tr_pos and tr_neg else 0.0

    tb_root = (tb_dir / f"{drug}__{tag}") if tb_dir is not None else None

    # --- #5: stratified 5-fold CV on the training split -----------------------
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []
    best_auc, best_model, best_fold = -np.inf, None, None
    for fold, (tr, va) in enumerate(skf.split(train_idx, y[train_idx])):
        writer = _new_writer(tb_root, f"cv_fold{fold}")
        model = MultiModalNet(specs, encoder_types, n_drugs=1, out_bias=out_bias).to(device)
        _, best_epoch = _train(model, arrays, alpha, train_idx[tr], train_idx[va],
                               epochs, batch_size, device, seed + fold, writer,
                               patience, min_delta)
        m = _metrics(y[train_idx[va]], _predict(model, arrays, train_idx[va], batch_size, device))
        m["fold"] = fold
        m["best_epoch"] = best_epoch
        folds.append(m)
        if writer is not None:
            _log_final(writer, "val", m, epochs)
            writer.close()
        print(f"[{drug}/{tag}] CV fold {fold}: AUC={m['auc']:.4f} AUC_PR={m['auc_pr']:.4f} "
              f"sens={m['sens']:.3f} spec={m['spec']:.3f} (n_val={m['n']}, "
              f"best_epoch={best_epoch})", flush=True)
        # baseline sd-cnn_model_best.h5 = the fold with the highest val AUC
        if m["auc"] == m["auc"] and m["auc"] > best_auc:
            best_auc, best_model, best_fold = m["auc"], model, fold

    # --- held-out TEST: score the best CV-fold model once (baseline protocol) --
    if best_model is None:            # every fold degenerate (single-class val)
        best_model = model
    writer = _new_writer(tb_root, "test")
    test_m = _metrics(y[test_idx], _predict(best_model, arrays, test_idx, batch_size, device))
    if writer is not None:
        _log_final(writer, "test", test_m, epochs)
        writer.close()

    aucs = [f["auc"] for f in folds if f["auc"] == f["auc"]]
    prs = [f["auc_pr"] for f in folds if f["auc_pr"] == f["auc_pr"]]
    cv_mean = float(np.mean(aucs)) if aucs else float("nan")
    cv_std = float(np.std(aucs)) if aucs else float("nan")

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
        "n_isolates": data.n, "n_valid": n,
        "n_resistant": n_R, "n_susceptible": n_S,
        "epochs": epochs, "batch_size": batch_size,
        "patience": patience, "min_delta": min_delta, "out_bias": out_bias,
        "cv_folds": folds,
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        "cv_auc_pr_mean": float(np.mean(prs)) if prs else float("nan"),
        "test_model_fold": best_fold,
        "test": test_m,
        "seconds": round(time.time() - t0, 1),
    }

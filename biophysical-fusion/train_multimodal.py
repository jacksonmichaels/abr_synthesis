"""
Generic mini-batched training/eval engine for any set of modalities.

Given a ``datasets.DrugData`` bundle (any combination of DNA / protein /
biophysical / regulatory blocks), builds a ``models.MultiModalNet`` with one
branch per block and runs BIG-TB's SD-CNN protocol (decision #6): fixed
train/test split (0.2, seed 42), plain 5-fold KFold on the training portion,
masked weighted BCE with tb.alpha_mat class weighting, tb.get_threshold_val
operating point — mini-batched (real data doesn't fit one GPU) and additionally
reporting a held-out TEST metric.

This generalizes eval_dna_cnn.py (DNA-only) to arbitrary modality sets; the two
share the same protocol and metric conventions. Optional TensorBoard logging
mirrors eval_dna_cnn's layout.
"""
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split

from bigtb_ref import tb
from models import MultiModalNet
from train import masked_weighted_bce

LR = float(np.exp(-9.0))  # matches BIG-TB SD-CNN / train.run_cv / eval_dna_cnn


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batch(arrays, idx, device):
    return [torch.from_numpy(a[idx]).to(device, non_blocking=True) for a in arrays]


def _train(model, arrays, alpha, idx, epochs, batch_size, device, seed, writer=None):
    _set_seed(seed)
    opt = optim.Adam(model.parameters(), lr=LR)
    idx = np.asarray(idx)
    model.train()
    for ep in range(epochs):
        perm = idx[np.random.permutation(len(idx))]
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
            writer.flush()
    return model


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
                 tb_dir=None, seed=0, branch_models=None, default_encoder="cnn"):
    """Train/eval a MultiModalNet on a DrugData bundle. Returns a result dict.

    branch_models    : {modality: encoder_type} — e.g. {'protein': 'transformer'}.
                       Each block inherits its modality's encoder; modalities not
                       named fall back to default_encoder.
    default_encoder  : encoder for any modality not in branch_models (default cnn).
    """
    t0 = time.time()
    branch_models = branch_models or {}
    drug, tag = data.drug, data.modality_tag()
    arrays = data.arrays()
    specs = data.branch_specs()
    encoder_types = [branch_models.get(b.modality, default_encoder) for b in data.blocks]
    # per-modality view for logging/results (all blocks of a modality share a type)
    enc_by_modality = {b.modality: e for b, e in zip(data.blocks, encoder_types)}
    y = data.y
    n = data.n
    counts = data.class_counts()
    print(f"[{drug}/{tag}] blocks={[b.name for b in data.blocks]} "
          f"specs={specs} encoders={enc_by_modality} n={n} R={counts['R']} "
          f"S={counts['S']} missing={counts['missing']}", flush=True)

    y2 = y.reshape(-1, 1)
    alpha = tb.alpha_mat(y2, None, weight=1.0).astype(np.float32)
    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=42)

    tb_root = (tb_dir / f"{drug}__{tag}") if tb_dir is not None else None

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    folds = []
    for fold, (tr, va) in enumerate(kf.split(train_idx)):
        writer = _new_writer(tb_root, f"cv_fold{fold}")
        model = MultiModalNet(specs, encoder_types, n_drugs=1).to(device)
        _train(model, arrays, alpha, train_idx[tr], epochs, batch_size, device, seed + fold, writer)
        m = _metrics(y[train_idx[va]], _predict(model, arrays, train_idx[va], batch_size, device))
        m["fold"] = fold
        folds.append(m)
        if writer is not None:
            _log_final(writer, "val", m, epochs)
            writer.close()
        print(f"[{drug}/{tag}] fold {fold}: AUC={m['auc']:.4f} AUC_PR={m['auc_pr']:.4f} "
              f"sens={m['sens']:.3f} spec={m['spec']:.3f} (n_val={m['n']})", flush=True)

    # held-out test: retrain on the full training split, score once on test
    writer = _new_writer(tb_root, "test")
    model = MultiModalNet(specs, encoder_types, n_drugs=1).to(device)
    _train(model, arrays, alpha, train_idx, epochs, batch_size, device, seed, writer)
    test_m = _metrics(y[test_idx], _predict(model, arrays, test_idx, batch_size, device))
    if writer is not None:
        _log_final(writer, "test", test_m, epochs)
        writer.close()
    print(f"[{drug}/{tag}] TEST: AUC={test_m['auc']:.4f} AUC_PR={test_m['auc_pr']:.4f} "
          f"sens={test_m['sens']:.3f} spec={test_m['spec']:.3f} (n_test={test_m['n']})", flush=True)

    aucs = [f["auc"] for f in folds if f["auc"] == f["auc"]]
    prs = [f["auc_pr"] for f in folds if f["auc_pr"] == f["auc_pr"]]
    return {
        "drug": drug, "modalities": data.modalities, "tag": tag,
        "genes": data.gene_order, "blocks": [b.name for b in data.blocks],
        "branch_specs": specs, "encoders": enc_by_modality,
        "encoder_types": encoder_types, "dropped": data.dropped,
        "n_isolates": n, "n_valid": counts["R"] + counts["S"],
        "n_resistant": counts["R"], "n_susceptible": counts["S"],
        "epochs": epochs, "batch_size": batch_size,
        "cv_folds": folds,
        "cv_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "cv_auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "cv_auc_pr_mean": float(np.mean(prs)) if prs else float("nan"),
        "test": test_m,
        "seconds": round(time.time() - t0, 1),
    }

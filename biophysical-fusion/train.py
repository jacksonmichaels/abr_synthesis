"""
Phase 4 — cross-validation training/eval harness. Protocol mirrors BIG-TB's
existing SD-CNN script exactly (decision #6): fixed random train/test
split, then plain (non-stratified) KFold on the training portion, masked
weighted BCE with alpha-matrix class weighting, sens/spec threshold search.
Reuses BIG-TB's alpha_mat / get_threshold_val directly rather than
reimplementing them; masked_multi_weighted_bce is ported from Keras to
torch since autograd needs it in-graph (kept single-drug/multi-drug shaped
alike per decision #6 — a (N, n_drugs) label matrix either way).

No mini-batching / early stopping yet: full-batch gradient descent per
epoch, sized for the small synthetic fixtures this is exercised against.
Revisit once real data size (post Unity access) is known.
"""
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split

from bigtb_ref import tb
from data import concat_upsampled


def masked_weighted_bce(logits, alpha, eps=1e-7):
    """Torch port of tb_cnn_codebase.masked_multi_weighted_bce. `alpha`
    encodes both label and class weight: positive = susceptible (weighted),
    negative = resistant (weighted), 0 = missing (masked out)."""
    y_pred = torch.sigmoid(logits).clamp(eps, 1 - eps)
    y_true = (alpha > 0).float()
    mask = (alpha != 0).float()
    a = alpha.abs()
    bce = -a * y_true * torch.log(y_pred) - (1 - a) * (1 - y_true) * torch.log(1 - y_pred)
    denom = mask.sum(dim=-1).clamp_min(eps)
    return ((bce * mask).sum(dim=-1) / denom).mean()


def _select_bio_input(model_cls, bio_Xs, dna_len):
    """bio_Xs is always the list[(N,3,K_g)] per-gene arrays from
    data.build_dataset. Models declare which shape they want via their
    `bio_input` class attribute (models.py): "per_gene" keeps the list as
    is, "upsampled_concat" (EarlyFusionCNN) collapses it into one array
    matching dna_len."""
    kind = getattr(model_cls, "bio_input", "per_gene")
    if kind == "per_gene":
        return list(bio_Xs), [a.shape[2] for a in bio_Xs]
    return [concat_upsampled(bio_Xs, dna_len)], None


def run_cv(model_cls, dna_X, bio_Xs, y, drug, n_splits=5, epochs=60, lr=float(np.exp(-9.0)),
           test_size=0.2, random_seed=42, device="cpu", **model_kwargs):
    n = dna_X.shape[0]
    train_idx, test_idx = train_test_split(np.arange(n), test_size=test_size, random_state=random_seed)

    y2 = y.reshape(-1, 1)  # (N, n_drugs); single-drug is just n_drugs=1
    # drug_name intentionally omitted: tb.alpha_mat's logging print uses a
    # unicode alpha symbol that crashes on Windows' default cp1252 console.
    alpha_all = tb.alpha_mat(y2, None, weight=1.0)

    bio_arrs, bio_lens = _select_bio_input(model_cls, bio_Xs, dna_X.shape[2])
    per_gene = getattr(model_cls, "bio_input", "per_gene") == "per_gene"

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    results = []
    for fold, (tr, va) in enumerate(kf.split(train_idx)):
        tr_idx, va_idx = train_idx[tr], train_idx[va]

        model = model_cls(dna_len=dna_X.shape[2], bio_lens=bio_lens, n_drugs=y2.shape[1], **model_kwargs).to(device)
        opt = optim.Adam(model.parameters(), lr=lr)

        dna_tr = torch.tensor(dna_X[tr_idx], device=device)
        bio_tr = [torch.tensor(a[tr_idx], device=device) for a in bio_arrs]
        alpha_tr = torch.tensor(alpha_all[tr_idx], dtype=torch.float32, device=device)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            bio_in = bio_tr if per_gene else bio_tr[0]
            loss = masked_weighted_bce(model(dna_tr, bio_in), alpha_tr)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            dna_va = torch.tensor(dna_X[va_idx], device=device)
            bio_va = [torch.tensor(a[va_idx], device=device) for a in bio_arrs]
            bio_va_in = bio_va if per_gene else bio_va[0]
            pred = torch.sigmoid(model(dna_va, bio_va_in)).cpu().numpy().reshape(-1)

        y_va = y[va_idx]
        valid = y_va != -1
        auc = roc_auc_score(y_va[valid], pred[valid])
        auc_pr = average_precision_score(1 - y_va[valid], 1 - pred[valid])
        thresh = tb.get_threshold_val(y_va[valid], pred[valid])
        row = {"fold": fold, "auc": auc, "auc_pr": auc_pr, **thresh}
        results.append(row)
        print(f"  fold {fold}: AUC={auc:.4f} AUC_PR={auc_pr:.4f} "
              f"sens={thresh['sens']:.3f} spec={thresh['spec']:.3f}")

    return results

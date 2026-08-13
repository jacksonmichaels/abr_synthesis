"""
Shared training primitives: the masked weighted BCE loss and the early stopper
both CV engines (``training.multimodal`` / ``training.multidrug``) build on.

Both are ports of BIG-TB's SD-CNN machinery — ``masked_weighted_bce`` from
``tb_cnn_codebase.masked_multi_weighted_bce`` (Keras -> torch, since autograd
needs it in-graph), ``EarlyStopper`` from Keras ``EarlyStopping(monitor=...,
restore_best_weights=True)``. The label matrix is (N, n_drugs) either way, so
single-drug and multi-drug share one loss (decision #6).
"""
import torch


def masked_weighted_bce(logits, alpha, eps=1e-7):
    """Torch port of tb_cnn_codebase.masked_multi_weighted_bce. `alpha`
    encodes both label and class weight: positive = susceptible (weighted),
    negative = resistant (weighted), 0 = missing (masked out).

    Reduction (TODO #3): the per-row masked BCE is averaged over the rows that
    carry at least one valid label, NOT over the raw batch size. This makes the
    loss invariant to masked (all-missing) padding rows — a batch of k valid
    rows and the same k rows padded with all-missing rows give an identical
    value. When no row is masked (the regime after the missing-phenotype filter
    in run_modal_cv), this is exactly the batch mean the baseline Keras loss
    reduces to, so it does not diverge from BIG-TB on real training data."""
    y_pred = torch.sigmoid(logits).clamp(eps, 1 - eps)
    y_true = (alpha > 0).float()
    mask = (alpha != 0).float()
    a = alpha.abs()
    bce = -a * y_true * torch.log(y_pred) - (1 - a) * (1 - y_true) * torch.log(1 - y_pred)
    valid_per_row = mask.sum(dim=-1)                       # valid labels in each row
    per_row = (bce * mask).sum(dim=-1) / valid_per_row.clamp_min(eps)
    n_valid_rows = (valid_per_row > 0).float().sum().clamp_min(eps)
    return per_row.sum() / n_valid_rows


class EarlyStopper:
    """Validation-loss early stopping with best-weight restore (TODO #4),
    mirroring Keras ``EarlyStopping(monitor='val_loss', patience=5,
    min_delta=1e-4, restore_best_weights=True)`` used by BIG-TB's
    run_SDCNN_ccp_crossval.

    An epoch counts as an improvement only if its loss beats the best-so-far by
    more than ``min_delta``. After ``patience`` consecutive non-improving epochs
    ``step`` returns True (stop). ``restore`` loads the state_dict snapshotted at
    the best epoch — call it whether training stopped early or ran to the epoch
    ceiling, exactly as restore_best_weights=True does.

    ``min_epochs`` is a warmup: for the first ``min_epochs`` epochs the patience
    counter is held at zero, so training cannot stop no matter how flat the
    monitored metric looks. Needed for architectures that start from a degenerate
    initialisation and spend a while escaping it before they learn anything — the
    full_run SetFusionNet cells are the case that motivated it. Their train loss
    sat at 0.2405 for ~12 epochs (near-collinear fused tokens), the val AUC best
    landed inside that plateau, and patience=15 then fired at epoch ~25 and
    restored weights from BEFORE the network broke out of the plateau at epoch
    13. Best-weight tracking still runs during warmup, so warming up can never
    return a worse model than not warming up — it only buys more epochs.
    Default 0 = the previous behaviour exactly."""

    def __init__(self, patience=5, min_delta=1e-4, mode="min", min_epochs=0):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.min_epochs = min_epochs
        self.best = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = None
        self.num_bad = 0
        self.best_state = None

    def step(self, epoch, val_metric, model):
        """Record ``val_metric`` for ``epoch``; snapshot weights if it improved.
        Returns True when training should stop now. ``mode='min'`` treats lower
        as better (val_loss); ``mode='max'`` treats higher as better (val AUC) —
        the latter avoids stopping at the low-loss majority-class collapse on
        heavily imbalanced drugs, where val_loss plateaus but ranking has not.

        ``epoch`` is 1-based (callers pass ``ep + 1``). While it is below
        ``min_epochs`` the bad-epoch counter is pinned to zero, so patience only
        starts accumulating once the warmup is over."""
        improved = (val_metric < self.best - self.min_delta) if self.mode == "min" \
            else (val_metric > self.best + self.min_delta)
        if improved:
            self.best = val_metric
            self.best_epoch = epoch
            self.num_bad = 0
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
            return False
        if epoch < self.min_epochs:      # warmup: never stop, never bank a bad epoch
            self.num_bad = 0
            return False
        self.num_bad += 1
        return self.num_bad >= self.patience

    def restore(self, model):
        """Restore the weights from the best epoch (no-op if never improved)."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

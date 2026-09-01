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


# ---------------------------------------------------------------------------
# Learn-to-Branch support (models.BranchedHead).
#
# The branched head needs two things the plain DenseHead does not, and both
# have to happen in the training loop rather than inside forward():
#
#   * the Gumbel-softmax temperature must be annealed ONCE PER EPOCH, not per
#     batch -- forward() has no idea which epoch it is in;
#   * the generic head is trained by an AUXILIARY term added to the objective
#     (Eq. 4 of Luo et al.: alpha*RE + beta*CE_personal + lambda*CE_generic; we
#     drop the reconstruction term, see below).
#
# We do not port the LSTM autoencoder or its reconstruction loss. It denoises
# irregularly-sampled sensor data; our inputs are aligned one-hot genotypes,
# where `token_signal` measured that ~99.86% of an encoded block is constant
# across isolates -- a reconstruction objective would spend itself rebuilding
# that constant. The reference-difference encoding (`--delta`) is the
# domain-appropriate version of the same idea and already exists.
# ---------------------------------------------------------------------------

def _branched_heads(model):
    """Every BranchedHead inside `model` (usually one, or none)."""
    from models import BranchedHead
    return [m for m in model.modules() if isinstance(m, BranchedHead)]


def anneal_branch_temperature(model, epoch, total_epochs):
    """Step the Gumbel-softmax temperature. No-op without a branched head."""
    for h in _branched_heads(model):
        h.anneal(epoch, total_epochs)


def branch_aux_loss(model, alpha):
    """`generic_weight * masked_weighted_bce(generic_logits, alpha)`, or 0.

    Reads the logits cached by ``BranchedHead.forward`` on the *training* path,
    so this must be called after the forward pass of the same step and only
    while the model is in train mode. Returns a plain 0.0 float when there is
    no branched head, so callers can add it unconditionally."""
    total = 0.0
    for h in _branched_heads(model):
        if h.aux_logits is not None and h.generic_weight:
            total = total + h.generic_weight * masked_weighted_bce(h.aux_logits, alpha)
    return total


def branch_assignments(model):
    """Discovered drug clusters, {group: [drug, ...]}, or None."""
    heads = _branched_heads(model)
    return heads[0].assignments() if heads else None

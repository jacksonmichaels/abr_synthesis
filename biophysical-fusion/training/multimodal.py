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
    early stopping (monitor=val AUC by default) + best-weight restore (TODO #4),
  * masked weighted BCE loss, tb.get_threshold_val operating point,
  * TEST metric = the best-val-AUC CV fold model scored once on the held-out
    split (baseline's sd-cnn_model_best.h5).

Early-stopping defaults deviate from the strict baseline (monitor='val_loss',
out_bias=log-odds), which collapsed to the all-susceptible majority on the
imbalanced drugs (MOXI/ETO): val_loss bottoms out at the degenerate solution,
so it stopped in ~1-9 epochs at CV AUC ~0.54. A knob sweep over {monitor,
patience, out_bias} on MOXIFLOXACIN DNA (2026-07, script since retired) found
monitor='auc' + patience=15 + out_bias=None recovers CV AUC 0.59->0.85 and a
usable operating point; those are now the defaults. Pass monitor='loss' /
out_bias='auto' to reproduce the strict baseline.

Everything is mini-batched (real data doesn't fit one GPU). ``arch`` selects the
network: 'late_fusion' (one encoder per block, concatenated) or 'mdcnn' (BIG-TB's
own topology — loci stacked as channels, see models/net.py). Optional TensorBoard
logging per fold.

NB: two knobs deviate from the baseline *crossval* script in favour of the
paper's stated protocol — CV uses StratifiedKFold(seed=42) where the baseline
used plain KFold(seed=1), and the output bias is initialised to the log-odds
where Keras leaves it at 0. Neither changes AUC ranking materially; see the
inline notes at each step.
"""
import math
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from bigtb_ref import tb
from models import (SETFUSION_DEFAULTS, CisFusionNet, MDCNNNet, MultiModalNet,
                    SetFusionNet)
from .checkpoint import RunCheckpointer, model_config
from .core import EarlyStopper, masked_weighted_bce

LR = float(np.exp(-9.0))  # matches BIG-TB SD-CNN (run_SDCNN_ccp_crossval)


def build_optimizer(model, lr=None, weight_decay=0.0):
    """Adam at the BIG-TB learning rate by default.

    ``weight_decay > 0`` switches to AdamW, because Adam's ``weight_decay``
    couples the penalty into the adaptive step and does not actually decay
    weights the way the name implies. At wd=0 we deliberately stay on Adam so a
    rerun of an existing config is bit-identical to the run that produced
    results/experiments/full_run.

    LR is exposed because exp(-9) ~ 1.2e-4 is roughly 10x below a normal Adam
    setting, and the full_run joint folds were still climbing at the 150-epoch
    cap (best_epoch 120-148 on 40% of late_fusion folds). Whether that is an
    epoch-budget problem or a learning-rate problem is what the joint_convergence
    pilot exists to answer."""
    lr = LR if lr is None else float(lr)
    if weight_decay:
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    return optim.Adam(model.parameters(), lr=lr)


LR_SCHEDULES = ("none", "cosine")


def build_scheduler(opt, epochs, schedule="none", warmup_epochs=0):
    """Per-EPOCH LR schedule, or None for the flat LR every run so far has used.

    'cosine' is linear warmup over ``warmup_epochs`` epochs, then cosine decay
    towards zero across the remaining budget. It exists for the wide/deep
    setfusion arms: a transformer at d_model 512 is where a flat LR from step
    one is most likely to be the thing that fails. Note the multiplier never
    exceeds 1.0, so this can only LOWER the LR relative to the run it is
    compared against — unlike raising the base LR, which
    ``results/experiments/joint_convergence`` already measured as catastrophic
    on this data (macro CV 0.911 -> 0.735 at lr 1e-3, -> 0.583 with dropout and
    weight decay on top).

    Interaction with early stopping, on purpose: the cosine is laid out over the
    full ``epochs`` cap, so a fold that stops at 100 of 300 only walked the first
    third of the curve. Decaying to zero by the typical stopping epoch instead
    would require knowing that epoch in advance.

    Returns None for schedule='none', so the caller's loop is untouched and
    every existing config stays bit-identical."""
    if schedule in (None, "none"):
        return None
    if schedule != "cosine":
        raise ValueError(f"unknown lr schedule {schedule!r}; choose from {LR_SCHEDULES}")
    warmup = max(int(warmup_epochs), 0)

    def factor(ep):                      # ep is 0-based, stepped once per epoch
        if warmup and ep < warmup:
            return (ep + 1) / warmup
        span = max(epochs - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min((ep - warmup) / span, 1.0)))

    return optim.lr_scheduler.LambdaLR(opt, factor)


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batch(arrays, idx, device):
    # .float() casts uint8 one-hot storage (the memory-lean layout the multi-drug
    # loader uses for its many locus blocks) up to float for the conv branches;
    # it is a no-op for the float32 blocks the single-drug path already uses.
    return [torch.from_numpy(a[idx]).to(device, non_blocking=True).float() for a in arrays]


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


def _val_auc(model, arrays, y, idx, batch_size, device):
    """Val-fold ROC-AUC on the resistant/susceptible ranking (NaN if the fold is
    single-class). Used when early stopping monitors AUC instead of loss."""
    idx = np.asarray(idx)
    yt = y[idx]
    valid = yt != -1
    if valid.sum() == 0 or len(np.unique(yt[valid])) < 2:
        return float("nan")
    pred = _predict(model, arrays, idx, batch_size, device)
    return float(roc_auc_score(yt[valid], pred[valid]))


def _train(model, arrays, alpha, tr_idx, va_idx, epochs, batch_size, device, seed,
           writer=None, patience=5, min_delta=1e-4, monitor="loss", y=None,
           min_epochs=0, lr=None, weight_decay=0.0, lr_schedule="none",
           warmup_epochs=0):
    """Mini-batched training with early stopping + best-weight restore (TODO #4).
    `va_idx=None` disables early stopping (fixed epoch loop). `monitor` selects
    the early-stopping metric: 'loss' (val masked-BCE, minimised) or 'auc' (val
    ROC-AUC, maximised — needs `y`). `min_epochs` holds patience at zero for that
    many epochs (warmup — see EarlyStopper). Returns (model, best_epoch, history);
    best_epoch is None when no validation fold is given. `history` holds the
    per-epoch train loss and monitored val metric (curves.save_curves plots
    them) so a run shows whether the epoch cap was reached or plateaued."""
    _set_seed(seed)
    opt = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    sched = build_scheduler(opt, epochs, lr_schedule, warmup_epochs)
    tr_idx = np.asarray(tr_idx)
    mode = "max" if monitor == "auc" else "min"
    stopper = EarlyStopper(patience, min_delta, mode=mode,
                           min_epochs=min_epochs) if va_idx is not None else None
    history = {"train_loss": [], f"val_{monitor}": []}
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
        if sched is not None:            # per-epoch, after the epoch's steps
            sched.step()
        if writer is not None:
            writer.add_scalar("loss/train", run_loss / max(seen, 1), ep + 1)
        if stopper is not None:
            if monitor == "auc":
                val_metric = _val_auc(model, arrays, y, va_idx, batch_size, device)
                history[f"val_{monitor}"].append(val_metric)  # NaN keeps epochs aligned
                if val_metric != val_metric:  # NaN fold: skip, don't count as bad
                    continue
            else:
                val_metric = _eval_loss(model, arrays, alpha, va_idx, batch_size, device)
                history[f"val_{monitor}"].append(val_metric)
            if writer is not None:
                writer.add_scalar(f"{monitor}/val", val_metric, ep + 1)
            if stopper.step(ep + 1, val_metric, model):
                break
    if stopper is not None:
        stopper.restore(model)   # restore_best_weights=True: best epoch, always
    if writer is not None:
        writer.flush()
    return model, (stopper.best_epoch if stopper is not None else None), history


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


def _build_model(arch, blocks, specs, encoder_types, n_drugs, out_bias,
                 branch_models=None, default_encoder="cnn", head=None,
                 mdcnn_trunk_per_modality=False, setfusion=None):
    """One model per fold.

    late_fusion -> our per-block encoder net; mdcnn -> BIG-TB's
    locus-as-channel topology; setfusion -> weight-shared encoders with
    locus-keyed set fusion, which is built from the BLOCKS rather than the
    specs because it keys off each block's (modality, locus) name.

    ``head`` carries the dense-head knobs ({hidden, dropout, per_drug_hidden});
    ``setfusion`` carries the setfusion-only capacity knobs (SETFUSION_DEFAULTS:
    d_model, nhead, layers, dim_ff, dropout, enc_*, bins). Both default to the
    values that produced full_run, so omitting them reproduces that run
    exactly."""
    head = dict(head or {})
    if arch == "mdcnn":
        return MDCNNNet.from_blocks(blocks, n_drugs=n_drugs, out_bias=out_bias,
                                    trunk_per_modality=mdcnn_trunk_per_modality,
                                    **head)
    if arch == "setfusion":
        # `hidden`/`per_drug_hidden` mean the same thing here as in DenseHead,
        # so they come off the same head dict. (`hidden` was previously dropped
        # on the floor for this arch — SetFusionNet's own default is the same
        # 256, so nothing that has already run is affected.)
        return SetFusionNet.from_blocks(blocks, n_drugs=n_drugs, out_bias=out_bias,
                                        hidden=head.get("hidden", 256),
                                        head_dropout=head.get("dropout", 0.0),
                                        per_drug_hidden=head.get("per_drug_hidden", 0),
                                        **{**SETFUSION_DEFAULTS, **(setfusion or {})})
    if arch == "cisfusion":
        return CisFusionNet.from_blocks(blocks, n_drugs=n_drugs, out_bias=out_bias,
                                        branch_models=branch_models,
                                        default_encoder=default_encoder, **head)
    return MultiModalNet(specs, encoder_types, n_drugs=n_drugs, out_bias=out_bias,
                         **head)


def run_modal_cv(data, epochs=60, n_splits=5, batch_size=128, device="cpu",
                 tb_dir=None, seed=0, branch_models=None, default_encoder="cnn",
                 patience=15, min_delta=1e-4, monitor="auc", out_bias=None,
                 arch="late_fusion", min_epochs=0, lr=None, weight_decay=0.0,
                 hidden=256, dropout=0.0, per_drug_hidden=0,
                 mdcnn_trunk_per_modality=False, run_name=None,
                 save_weights="best", weights_dir=None, data_config=None,
                 setfusion=None, lr_schedule="none", warmup_epochs=0):
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
    arch             : 'late_fusion' (our per-block encoder net), 'mdcnn'
                       (BIG-TB's own topology: loci stacked as channels on one
                       zero-padded axis, 12-bp conv over all of them from layer
                       1), or 'setfusion' (shared per-modality encoders, tokens
                       keyed by (modality, locus), transformer fusion + one
                       query per drug). The latter two ignore
                       branch_models/default_encoder and expect PER-LOCUS blocks.
    setfusion        : setfusion-only capacity overrides (see SETFUSION_DEFAULTS).
                       Ignored by every other arch.
    lr_schedule      : 'none' (flat LR, what every recorded run used) or
                       'cosine' with `warmup_epochs` (see build_scheduler).
    """
    t0 = time.time()
    branch_models = branch_models or {}
    setfusion = {k: v for k, v in (setfusion or {}).items()
                 if v is not None and SETFUSION_DEFAULTS.get(k) != v}
    head = {"hidden": hidden, "dropout": dropout, "per_drug_hidden": per_drug_hidden}
    drug, tag = data.drug, data.modality_tag()
    ckpt = RunCheckpointer(run_name, f"{drug}__{tag}", mode=save_weights,
                           weights_dir=weights_dir)
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
    enc_desc = "n/a (mdcnn)" if arch == "mdcnn" else enc_by_modality
    print(f"[{drug}/{tag}] arch={arch} blocks={[b.name for b in data.blocks]} "
          f"specs={specs} encoders={enc_desc} n_valid={n} R={n_R} S={n_S} "
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
    # out_bias='auto' -> log-odds init (baseline-aligned); a float -> that exact
    # bias; None -> leave PyTorch's default (no majority-class prior baked in).
    tr_pos = int((y[train_idx] == 1).sum())
    tr_neg = int((y[train_idx] == 0).sum())
    if out_bias == "auto":
        out_bias = float(np.log(tr_pos / tr_neg)) if tr_pos and tr_neg else 0.0

    tb_root = (tb_dir / f"{drug}__{tag}") if tb_dir is not None else None

    # --- #5: stratified 5-fold CV on the training split -----------------------
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []
    best_auc, best_model, best_fold = -np.inf, None, None
    for fold, (tr, va) in enumerate(skf.split(train_idx, y[train_idx])):
        writer = _new_writer(tb_root, f"cv_fold{fold}")
        model = _build_model(arch, data.blocks, specs, encoder_types, 1, out_bias,
                             branch_models, default_encoder, head=head,
                             mdcnn_trunk_per_modality=mdcnn_trunk_per_modality,
                             setfusion=setfusion).to(device)
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[{drug}/{tag}] {arch}: {n_params:,} parameters", flush=True)
        _, best_epoch, history = _train(model, arrays, alpha, train_idx[tr], train_idx[va],
                                        epochs, batch_size, device, seed + fold, writer,
                                        patience, min_delta, monitor=monitor, y=y,
                                        min_epochs=min_epochs, lr=lr,
                                        weight_decay=weight_decay,
                                        lr_schedule=lr_schedule,
                                        warmup_epochs=warmup_epochs)
        m = _metrics(y[train_idx[va]], _predict(model, arrays, train_idx[va], batch_size, device))
        m["fold"] = fold
        m["best_epoch"] = best_epoch
        m["history"] = history       # per-epoch train loss + val metric (training.curves)
        folds.append(m)
        # `model` already carries the best-epoch weights (stopper.restore)
        ckpt.add_fold(fold, model, best_epoch, m["auc"])
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

    # --- persist the weights + everything needed to rebuild them --------------
    kept_isolates = [data.isolate_ids[i] for i in keep]
    weights_path = ckpt.write({
        "run_name": run_name, "scope": "single", "tag": tag, "drug": drug,
        "model": model_config(
            arch=arch, blocks=data.blocks, encoder_types=encoder_types,
            drug_names=[drug], out_bias=out_bias, head=head,
            mdcnn_trunk_per_modality=mdcnn_trunk_per_modality,
            branch_models=branch_models, default_encoder=default_encoder,
            n_params=n_params, setfusion=setfusion),
        "data": {**(data_config or {}),
                 "modalities_used": data.modalities,
                 "modalities_requested": data.requested,
                 "dropped": data.dropped,
                 "loci": data.loci, "gene_order": data.gene_order,
                 "n_isolates": data.n, "n_valid": n,
                 "n_resistant": n_R, "n_susceptible": n_S},
        "split": {"test_size": 0.2, "split_seed": 42, "stratified": True,
                  "n_splits": n_splits, "kfold": "StratifiedKFold",
                  "kfold_seed": 42, "kfold_shuffle": True,
                  "isolate_order": "isolates.txt (post missing-phenotype filter)",
                  "test_isolate_ids": [kept_isolates[i] for i in test_idx]},
        "training": {"epochs": epochs, "batch_size": batch_size, "seed": seed,
                     "lr": LR if lr is None else float(lr),
                     "weight_decay": float(weight_decay),
                     "patience": patience, "min_delta": min_delta,
                     "monitor": monitor, "min_epochs": min_epochs,
                     "lr_schedule": lr_schedule,
                     "warmup_epochs": int(warmup_epochs)},
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
        "arch": arch, "n_params": n_params,
        "n_isolates": data.n, "n_valid": n,
        "n_resistant": n_R, "n_susceptible": n_S,
        "epochs": epochs, "batch_size": batch_size,
        "patience": patience, "min_delta": min_delta, "out_bias": out_bias,
        "monitor": monitor, "min_epochs": min_epochs,
        "lr": LR if lr is None else float(lr), "weight_decay": float(weight_decay),
        "lr_schedule": lr_schedule, "warmup_epochs": int(warmup_epochs),
        "setfusion": {**SETFUSION_DEFAULTS, **setfusion} if arch == "setfusion" else None,
        "hidden": hidden, "dropout": dropout, "per_drug_hidden": per_drug_hidden,
        "mdcnn_trunk_per_modality": bool(mdcnn_trunk_per_modality),
        "weights_dir": str(weights_path) if weights_path else None,
        "save_weights": save_weights,
        "cv_folds": folds,
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        "cv_auc_pr_mean": float(np.mean(prs)) if prs else float("nan"),
        "test_model_fold": best_fold,
        "test": test_m,
        "seconds": round(time.time() - t0, 1),
    }

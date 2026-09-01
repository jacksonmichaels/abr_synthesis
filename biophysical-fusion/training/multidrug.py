"""
Multi-drug ("MD-CNN") training/eval engine.

Trains a single ``models.MultiDrugNet`` to predict ALL drugs at once from the
union of their loci (one branch per locus), following the same BIG-TB protocol
as the single-drug ``training.multimodal.run_modal_cv`` but with a label MATRIX:

  * drop isolates with no phenotype for ANY drug,
  * random held-out split (seed 42) — multi-label, so not stratified,
  * per-DRUG alpha class-weights (tb.alpha_mat per column) fit on the train split,
  * KFold(5) CV on the train split; each fold early-stops on the macro-mean
    per-drug validation AUC (monitor='auc') with best-weight restore,
  * masked multi-drug weighted BCE (train.masked_weighted_bce already reduces
    over the drug axis and masks each isolate's missing drugs),
  * per-drug AND macro-mean AUC / AUC-PR reported for CV and the held-out TEST
    (the best-macro-val-AUC fold model), mirroring the MD-CNN reference.

Shares the low-level batching / seeding / loss / early-stopper with the
single-drug engine so the two stay consistent.
"""
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split

from bigtb_ref import tb
from models import (BRANCHED_DEFAULTS, EXPERIMENTAL_DEFAULTS,
                    EXPERIMENTAL_MODELS, LOCUSFUSION_DEFAULTS, SETFUSION_DEFAULTS,
                    TRANSFORMER_DEFAULTS, CisFusionNet, LocusFusionNet,
                    MDCNNNet, MultiDrugNet,
                    SetFusionNet, make_experimental, parse_block_key)
from .checkpoint import RunCheckpointer, model_config
from .core import (EarlyStopper, anneal_branch_temperature, branch_assignments,
                   branch_aux_loss, masked_weighted_bce)
from .multimodal import (LR, _batch, _new_writer, _set_seed, build_optimizer,
                         build_scheduler)


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
    # empty index: every arch exposes its output width as n_drugs / drug_names
    # (head.fc_out is None once per-drug heads are on, and SetFusionNet has no
    # .head at all, so neither can be read here).
    n_drugs = getattr(model, "n_drugs", None) or len(model.drug_names)
    return np.zeros((0, n_drugs))


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


def monitor_columns(Y, train_idx, min_n):
    """Which drug columns the early-stopping metric should average over.

    The monitored quantity is an UNWEIGHTED mean over 11 drugs, so a drug with
    almost no labels contributes a sixth of the signal's variance and none of
    its information. LEVOFLOXACIN has 269 phenotyped isolates — roughly 43 in a
    validation fold, ~15 of them resistant — and its per-epoch val AUC swings
    by more than the entire margin any of these runs is trying to measure, so
    early stopping ends up restoring whichever epoch LEVO got lucky on rather
    than the epoch that was best for the other ten drugs.

    ``min_n`` drops columns with fewer than that many labelled TRAIN rows from
    the monitor only. Every drug is still trained on (the loss is untouched)
    and still reported in CV/TEST — this changes when we stop, nothing else.
    min_n=0 monitors everything, which is the full_run behaviour."""
    counts = [int((Y[train_idx, j] != -1).sum()) for j in range(Y.shape[1])]
    if not min_n:
        return list(range(Y.shape[1])), counts
    cols = [j for j, n in enumerate(counts) if n >= min_n]
    if not cols:
        # never leave the stopper with nothing to watch — but say so, rather
        # than silently behaving like min_n=0 while the log claims otherwise
        print(f"  [monitor] --monitor-min-n={min_n} excludes EVERY drug "
              f"(max labelled train rows = {max(counts)}); ignoring it and "
              f"monitoring all {Y.shape[1]} drugs.", flush=True)
        return list(range(Y.shape[1])), counts
    return cols, counts


def _macro_val_auc(model, arrays, Y, idx, batch_size, device, cols=None):
    P = _predict(model, arrays, idx, batch_size, device)
    if cols is not None:
        Y, P = Y[:, cols], P[:, cols]
    _, macro_auc, _ = _per_drug_metrics(Y[idx], P, [str(i) for i in range(Y.shape[1])])
    return macro_auc


def _train(model, arrays, alpha, Y, tr_idx, va_idx, epochs, batch_size, device,
           seed, writer=None, patience=15, min_delta=1e-4, monitor="auc",
           min_epochs=0, lr=None, weight_decay=0.0, monitor_cols=None,
           lr_schedule="none", warmup_epochs=0):
    """Mini-batched training with early stopping on the macro val metric
    (monitor='auc' -> mean per-drug val AUC, maximised; 'loss' -> val BCE,
    minimised). Returns (model, best_epoch, history); `history` is the per-epoch
    train loss + monitored val metric, plotted by curves.save_curves so a run
    shows whether the epoch cap was reached or the curves plateaued."""
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
        # Gumbel-softmax temperature for models.BranchedHead: annealed once per
        # epoch over the WHOLE run, so theta sharpens toward a one-hot branch
        # assignment by the end. No-op for every other head.
        anneal_branch_temperature(model, ep, epochs)
        perm = tr_idx[np.random.permutation(len(tr_idx))]
        run_loss, seen = 0.0, 0
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            ab = torch.from_numpy(alpha[b]).to(device, non_blocking=True)
            opt.zero_grad()
            loss = masked_weighted_bce(model(_batch(arrays, b, device)), ab)
            # + lambda * CE_generic, the cold-start path (0.0 without a branched
            # head). Must follow the forward pass that cached the logits.
            loss = loss + branch_aux_loss(model, ab)
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
                metric = _macro_val_auc(model, arrays, Y, va_idx, batch_size, device,
                                        cols=monitor_cols)
                history[f"val_{monitor}"].append(metric)  # NaN keeps epochs aligned
                if metric != metric:      # NaN (no scorable drug): skip epoch
                    continue
            else:
                metric = _val_loss(model, arrays, alpha, va_idx, batch_size, device)
                history[f"val_{monitor}"].append(metric)
            if writer is not None:
                writer.add_scalar(f"{monitor}/val", metric, ep + 1)
            if stopper.step(ep + 1, metric, model):
                break
    if stopper is not None:
        stopper.restore(model)
    if writer is not None:
        writer.flush()
    return model, (stopper.best_epoch if stopper is not None else None), history


def _val_loss(model, arrays, alpha, idx, batch_size, device):
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
                     tb_dir=None, seed=0, branch_models=None, default_encoder="cnn",
                     patience=15, min_delta=1e-4, monitor="auc", out_bias=None,
                     arch="late_fusion", min_epochs=0, lr=None, weight_decay=0.0,
                     hidden=256, dropout=0.0, per_drug_hidden=0,
                     mdcnn_trunk_per_modality=False, monitor_min_n=0,
                     run_name=None, save_weights="best", weights_dir=None,
                     data_config=None, setfusion=None, lr_schedule="none",
                     warmup_epochs=0, transformer=None, locusfusion=None,
                     branched=None, experimental=None):
    """Train/eval a multi-drug net on a MultiDrugData bundle. Returns a result
    dict with per-drug and macro CV/TEST metrics.

    arch : 'late_fusion' -> MultiDrugNet (one encoder per locus, concatenated);
           'mdcnn' -> MDCNNNet, BIG-TB's own topology (every locus a channel on
           one zero-padded position axis, 12-bp conv across all of them from
           layer 1); 'setfusion' -> SetFusionNet, shared per-modality encoders
           with locus-keyed tokens and one attention query per drug. The latter
           two ignore branch_models/default_encoder and expect PER-LOCUS blocks.

    setfusion : setfusion-only capacity overrides (see models.SETFUSION_DEFAULTS);
                ignored by every other arch. lr_schedule/warmup_epochs select the
                per-epoch LR schedule (see training.multimodal.build_scheduler);
                'none' is the flat LR every recorded run used.

    transformer : capacity overrides for the transformer encoder / trunk (see
                models.TRANSFORMER_DEFAULTS), applied wherever a transformer is
                selected — per-locus branch under late_fusion / cisfusion, per
                trunk under mdcnn. Ignored under an all-CNN run.

    locusfusion : locusfusion-only capacity overrides (see
                models.LOCUSFUSION_DEFAULTS); ignored by every other arch. That
                arch also needs reference-difference input — load with
                delta=True or its tokenizer has nothing sparse to tokenize."""
    t0 = time.time()
    branch_models = branch_models or {}
    setfusion = {k: v for k, v in (setfusion or {}).items()
                 if v is not None and SETFUSION_DEFAULTS.get(k) != v}
    transformer = {k: v for k, v in (transformer or {}).items()
                   if v is not None and TRANSFORMER_DEFAULTS.get(k) != v}
    locusfusion = {k: v for k, v in (locusfusion or {}).items()
                   if v is not None and LOCUSFUSION_DEFAULTS.get(k) != v}
    experimental = {k: v for k, v in (experimental or {}).items()
                    if v is not None and EXPERIMENTAL_DEFAULTS.get(k) != v}
    drugs = data.drugs
    n_drugs = len(drugs)
    tag = data.modality_tag()
    specs = data.branch_specs()
    encoder_types = [branch_models.get(b.modality, default_encoder) for b in data.blocks]

    # --- drop isolates with NO phenotype for any drug -------------------------
    keep = np.nonzero((data.Y != -1).any(axis=1))[0]
    Y = data.Y[keep]
    arrays = [a[keep] for a in data.arrays()]
    n = len(keep)
    print(f"[multidrug/{tag}] arch={arch} drugs={n_drugs} blocks={len(data.blocks)} "
          f"specs~{specs[:3]}{'...' if len(specs) > 3 else ''} "
          f"n_valid={n} (dropped {data.n - n} all-missing)", flush=True)

    # --- random held-out split (multi-label -> not stratified), seed 42 -------
    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=42)

    alpha = _alpha_matrix(Y, train_idx, n_drugs)
    tb_root = (tb_dir / f"multidrug__{tag}") if tb_dir is not None else None

    # which drugs the early stopper listens to (see monitor_columns)
    monitor_cols, drug_counts = monitor_columns(Y, train_idx, monitor_min_n)
    monitor_drugs = [drugs[j] for j in monitor_cols]
    if len(monitor_cols) < n_drugs:
        skipped = [f"{drugs[j]}(n={drug_counts[j]})"
                   for j in range(n_drugs) if j not in set(monitor_cols)]
        print(f"[multidrug/{tag}] early-stopping monitor uses "
              f"{len(monitor_cols)}/{n_drugs} drugs (--monitor-min-n="
              f"{monitor_min_n}); excluded from the STOP SIGNAL ONLY, still "
              f"trained and reported: {skipped}", flush=True)
    head = {"hidden": hidden, "dropout": dropout, "per_drug_hidden": per_drug_hidden,
            "branched": branched}
    if branched:
        # late_fusion / mdcnn / cisfusion all end in make_head and take it via
        # **head; setfusion and locusfusion build their own read-outs and would
        # silently ignore it, which would look like "branching did nothing".
        if arch not in ("late_fusion", "mdcnn", "cisfusion"):
            raise ValueError(
                f"--head branched is not implemented for --arch {arch} (it builds "
                "its own read-out). Supported: late_fusion, mdcnn, cisfusion.")
        print(f"[multidrug/{tag}] branched head: "
              f"{ {**BRANCHED_DEFAULTS, **branched} }", flush=True)
    ckpt = RunCheckpointer(run_name, f"multidrug__{tag}", mode=save_weights,
                           weights_dir=weights_dir)
    kept_isolates = [data.isolate_ids[i] for i in keep]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds, best_macro, best_model, best_fold = [], -np.inf, None, None
    for fold, (tr, va) in enumerate(kf.split(train_idx)):
        writer = _new_writer(tb_root, f"cv_fold{fold}")
        if arch == "mdcnn":
            # one uniform trunk encoder; see training.multimodal._build_model for
            # why a per-modality mix is refused rather than half-applied
            kinds = sorted(set(encoder_types))
            if len(kinds) > 1:
                raise ValueError(
                    f"--arch mdcnn takes ONE encoder for every trunk, got {kinds}. "
                    "Use --default-encoder.")
            model = MDCNNNet.from_blocks(data.blocks, drug_names=drugs,
                                         out_bias=out_bias,
                                         trunk_per_modality=mdcnn_trunk_per_modality,
                                         encoder=(kinds[0] if kinds else "cnn"),
                                         transformer=transformer, **head)
        elif arch in EXPERIMENTAL_MODELS:
            model = make_experimental(
                arch, [parse_block_key(b.name) for b in data.blocks],
                [b.spec() for b in data.blocks], drug_names=drugs,
                out_bias=out_bias, hidden=hidden,
                **{**EXPERIMENTAL_DEFAULTS, **experimental})
        elif arch == "locusfusion":
            # one token per variant, fused within a locus then across loci; the
            # (modality, locus) keys come off the block names as for setfusion.
            model = LocusFusionNet.from_blocks(
                data.blocks, drug_names=drugs, out_bias=out_bias,
                hidden=hidden, head_dropout=dropout,
                per_drug_hidden=per_drug_hidden,
                **{**LOCUSFUSION_DEFAULTS, **locusfusion})
        elif arch == "cisfusion":
            model = CisFusionNet.from_blocks(data.blocks, drug_names=drugs,
                                             out_bias=out_bias,
                                             branch_models=branch_models,
                                             default_encoder=default_encoder,
                                             transformer=transformer, **head)
        elif arch == "setfusion":
            # built from the blocks: the (modality, locus) keys live in their
            # names. `hidden`/`per_drug_hidden` mean what they mean everywhere
            # else; `setfusion` carries the capacity knobs this arch alone has.
            model = SetFusionNet.from_blocks(data.blocks, drug_names=drugs,
                                             out_bias=out_bias,
                                             hidden=hidden,
                                             head_dropout=dropout,
                                             per_drug_hidden=per_drug_hidden,
                                             **{**SETFUSION_DEFAULTS, **setfusion})
        else:
            model = MultiDrugNet(specs, drugs, encoder_types, out_bias=out_bias,
                                 transformer=transformer, **head)
        model = model.to(device)
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[multidrug/{tag}] {arch}: {n_params:,} parameters", flush=True)
        _, best_epoch, history = _train(model, arrays, alpha, Y, train_idx[tr], train_idx[va],
                                        epochs, batch_size, device, seed + fold, writer,
                                        patience, min_delta, monitor, min_epochs,
                                        lr=lr, weight_decay=weight_decay,
                                        monitor_cols=monitor_cols,
                                        lr_schedule=lr_schedule,
                                        warmup_epochs=warmup_epochs)
        P = _predict(model, arrays, train_idx[va], batch_size, device)
        per, macro_auc, macro_pr = _per_drug_metrics(Y[train_idx[va]], P, drugs)
        folds.append({"fold": fold, "best_epoch": best_epoch, "macro_auc": macro_auc,
                      "macro_auc_pr": macro_pr, "per_drug": per,
                      "history": history})   # per-epoch curves (training.curves)
        # `model` here already carries the best-epoch weights (stopper.restore)
        ckpt.add_fold(fold, model, best_epoch, macro_auc)
        if writer is not None:
            writer.close()
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
        "model": model_config(
            arch=arch, blocks=data.blocks, encoder_types=encoder_types,
            drug_names=drugs, out_bias=out_bias, head=head,
            mdcnn_trunk_per_modality=mdcnn_trunk_per_modality,
            branch_models=branch_models, default_encoder=default_encoder,
            n_params=n_params, setfusion=setfusion, transformer=transformer,
            locusfusion=locusfusion, experimental=experimental),
        "data": {**(data_config or {}),
                 "modalities_used": data.modalities,
                 "modalities_requested": data.requested,
                 "dropped": data.dropped,
                 "loci": data.loci, "gene_order": data.gene_order,
                 "n_isolates": data.n, "n_valid": n,
                 "dropped_all_missing": data.n - n},
        "split": {"test_size": 0.2, "split_seed": 42, "stratified": False,
                  "n_splits": n_splits, "kfold": "KFold", "kfold_seed": 42,
                  "kfold_shuffle": True,
                  "isolate_order": "isolates.txt (post all-missing filter)",
                  "test_isolate_ids": [kept_isolates[i] for i in test_idx]},
        "training": {"epochs": epochs, "batch_size": batch_size, "seed": seed,
                     "lr": LR if lr is None else float(lr),
                     "weight_decay": float(weight_decay),
                     "patience": patience, "min_delta": min_delta,
                     "monitor": monitor, "min_epochs": min_epochs,
                     "monitor_min_n": monitor_min_n,
                     "monitor_drugs": monitor_drugs,
                     "lr_schedule": lr_schedule,
                     "warmup_epochs": int(warmup_epochs)},
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
        "arch": arch, "n_params": n_params,
        "dropped": data.dropped, "n_isolates": data.n, "n_valid": n,
        "epochs": epochs, "batch_size": batch_size, "n_splits": n_splits,
        "patience": patience, "min_delta": min_delta, "out_bias": out_bias,
        "monitor": monitor, "min_epochs": min_epochs,
        "lr": LR if lr is None else float(lr), "weight_decay": float(weight_decay),
        "lr_schedule": lr_schedule, "warmup_epochs": int(warmup_epochs),
        "setfusion": {**SETFUSION_DEFAULTS, **setfusion} if arch == "setfusion" else None,
        "locusfusion": ({**LOCUSFUSION_DEFAULTS, **locusfusion}
                        if arch == "locusfusion" else None),
        "experimental": ({**EXPERIMENTAL_DEFAULTS, **experimental}
                         if arch in EXPERIMENTAL_MODELS else None),
        "transformer": ({**TRANSFORMER_DEFAULTS, **transformer}
                        if "transformer" in set(encoder_types) else None),
        "branched": {**BRANCHED_DEFAULTS, **branched} if branched else None,
        "branch_assignments": branch_assignments(best_model) if branched else None,
        "hidden": hidden, "dropout": dropout, "per_drug_hidden": per_drug_hidden,
        "mdcnn_trunk_per_modality": bool(mdcnn_trunk_per_modality),
        "monitor_min_n": monitor_min_n, "monitor_drugs": monitor_drugs,
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

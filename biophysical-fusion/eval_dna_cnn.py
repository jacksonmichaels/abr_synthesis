"""
Full evaluation of the DNA-only 1D CNN baseline (models.DNAOnlyCNN) on the
real BIG-TB genotype/phenotype data (all isolates), per drug.

This is the DNA-only control for the biophysical-fusion experiment (TODO.md
Phase 5). Protocol mirrors BIG-TB's SD-CNN (decision #6) and reuses its
machinery directly:
  - fixed random train/test split (test_size=0.2, seed 42),
  - plain (non-stratified) 5-fold KFold on the training portion,
  - masked weighted BCE loss with tb.alpha_mat class weighting,
  - tb.get_threshold_val for the sensitivity/specificity operating point.

Differences from train.run_cv (which is full-batch and fusion-shaped):
  - mini-batched (real data is ~18k isolates x several-thousand bp; the whole
    training tensor does not fit on one GPU), and
  - reports a held-out TEST metric (final model trained on the full training
    split, scored once on the untouched test split) in addition to the CV
    folds. run_cv only ever scored validation folds.

Metrics follow train.py's conventions exactly: y in {0=resistant,
1=susceptible}; AUC = roc_auc_score(y, sigmoid); AUC_PR = average precision
with resistant as the positive class.

Live monitoring is via TensorBoard (standard tooling, not a custom UI). Every
invocation is isolated in its own folder results/dna_cnn/runs/{run_name}/
(run_name defaults to a timestamp, override with --run-name), so separate
runs never overwrite each other and each shows as a distinct group in the
dashboard. Within a run, each CV fold and the final test model is its own
TensorBoard run under {run_name}/tb/{DRUG}/{cv_fold_k|test}, logging per-epoch
training loss; final validation/test AUC/AUC_PR/sens/spec are logged as
scalars, and a per-drug HParams row makes the drugs sortable/comparable in the
HPARAMS tab. Point TensorBoard at the runs/ root to see all runs:
    tensorboard --logdir biophysical-fusion/results/dna_cnn/runs
Final per-drug results are also written to {run_name}/{DRUG}.json plus a
combined {run_name}/summary.csv.

Usage:
    python eval_dna_cnn.py                      # all DRUG_TO_LOCI drugs
    python eval_dna_cnn.py --drugs RIFAMPICIN ISONIAZID
    python eval_dna_cnn.py --epochs 60 --batch-size 128 --device cuda
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from torch.utils.tensorboard import SummaryWriter

from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, tb
from data import build_dataset
from models import DNAOnlyCNN
from train import masked_weighted_bce

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "dna_cnn"
RUNS_DIR = RESULTS_DIR / "runs"  # each invocation gets its own subfolder here
LR = float(np.exp(-9.0))  # matches train.run_cv / BIG-TB SD-CNN


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _train(model, dna_X, alpha, idx, epochs, batch_size, device, seed, writer=None):
    """Mini-batched training. Logs per-epoch mean loss to `writer` (a
    TensorBoard SummaryWriter) so the loss curve is watchable live."""
    _set_seed(seed)
    opt = optim.Adam(model.parameters(), lr=LR)
    idx = np.asarray(idx)
    model.train()
    for ep in range(epochs):
        perm = idx[np.random.permutation(len(idx))]
        run_loss, seen = 0.0, 0
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            xb = torch.from_numpy(dna_X[b]).to(device, non_blocking=True)
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


def _predict(model, dna_X, idx, batch_size, device):
    idx = np.asarray(idx)
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            xb = torch.from_numpy(dna_X[b]).to(device, non_blocking=True)
            out.append(torch.sigmoid(model(xb)).cpu().numpy().reshape(-1))
    return np.concatenate(out)


def _metrics(y_true, pred):
    """AUC / AUC_PR / operating point on the valid (non-missing) subset,
    mirroring train.run_cv. NaN-guarded for single-class subsets."""
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
    """Log a fold/test's final scalar metrics at the last training step, so
    they line up on the x-axis with the end of the loss curve."""
    for k in ("auc", "auc_pr", "sens", "spec"):
        if m[k] == m[k]:  # skip NaN
            writer.add_scalar(f"{prefix}/{k}", m[k], step)
    writer.flush()


def _summary_row(result):
    return {
        "drug": result["drug"],
        "genes": "+".join(result["genes"]),
        "n_valid": result["n_valid"],
        "n_R": result["n_resistant"],
        "n_S": result["n_susceptible"],
        "cv_auc_mean": result["cv_auc_mean"],
        "cv_auc_std": result["cv_auc_std"],
        "cv_auc_pr_mean": result["cv_auc_pr_mean"],
        "test_auc": result["test"]["auc"],
        "test_auc_pr": result["test"]["auc_pr"],
        "test_sens": result["test"]["sens"],
        "test_spec": result["test"]["spec"],
        "seconds": result["seconds"],
    }


def evaluate_drug(drug, epochs, n_splits, batch_size, device, tb_dir, seed=0):
    drug = drug.upper()
    t0 = time.time()
    dna_X, _, gene_order, y, isolate_ids = build_dataset(
        drug, REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, compute_bio=False
    )
    dna_len = dna_X.shape[2]
    n = len(y)
    valid_total = int((y != -1).sum())
    n_R, n_S = int((y == 0).sum()), int((y == 1).sum())
    print(f"[{drug}] loaded: dna_X {dna_X.shape} genes={gene_order} "
          f"n={n} valid={valid_total} R={n_R} S={n_S} ({time.time() - t0:.0f}s)", flush=True)

    y2 = y.reshape(-1, 1)
    alpha = tb.alpha_mat(y2, None, weight=1.0).astype(np.float32)
    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=42)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    folds = []
    for fold, (tr, va) in enumerate(kf.split(train_idx)):
        writer = SummaryWriter(tb_dir / drug / f"cv_fold{fold}", flush_secs=10)
        model = DNAOnlyCNN(dna_len=dna_len, n_drugs=1).to(device)
        _train(model, dna_X, alpha, train_idx[tr], epochs, batch_size, device, seed + fold, writer)
        m = _metrics(y[train_idx[va]], _predict(model, dna_X, train_idx[va], batch_size, device))
        m["fold"] = fold
        folds.append(m)
        _log_final(writer, "val", m, epochs)
        writer.close()
        print(f"[{drug}] fold {fold}: AUC={m['auc']:.4f} AUC_PR={m['auc_pr']:.4f} "
              f"sens={m['sens']:.3f} spec={m['spec']:.3f} (n_val={m['n']})", flush=True)

    # Held-out test: retrain on the full training split, score once on test
    writer = SummaryWriter(tb_dir / drug / "test", flush_secs=10)
    model = DNAOnlyCNN(dna_len=dna_len, n_drugs=1).to(device)
    _train(model, dna_X, alpha, train_idx, epochs, batch_size, device, seed, writer)
    test_m = _metrics(y[test_idx], _predict(model, dna_X, test_idx, batch_size, device))
    _log_final(writer, "test", test_m, epochs)
    writer.close()
    print(f"[{drug}] TEST: AUC={test_m['auc']:.4f} AUC_PR={test_m['auc_pr']:.4f} "
          f"sens={test_m['sens']:.3f} spec={test_m['spec']:.3f} (n_test={test_m['n']})", flush=True)

    aucs = [f["auc"] for f in folds if f["auc"] == f["auc"]]  # drop NaN
    prs = [f["auc_pr"] for f in folds if f["auc_pr"] == f["auc_pr"]]
    result = {
        "drug": drug, "genes": gene_order, "dna_len": dna_len,
        "n_isolates": n, "n_valid": valid_total,
        "n_resistant": n_R, "n_susceptible": n_S,
        "epochs": epochs, "batch_size": batch_size,
        "cv_folds": folds,
        "cv_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "cv_auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "cv_auc_pr_mean": float(np.mean(prs)) if prs else float("nan"),
        "test": test_m,
        "seconds": round(time.time() - t0, 1),
    }

    # Per-drug HParams row -> sortable/comparable across drugs in TB's HPARAMS tab
    hp_writer = SummaryWriter(tb_dir / "_hparams" / drug)
    metric_dict = {
        "hparam/test_auc": test_m["auc"], "hparam/test_auc_pr": test_m["auc_pr"],
        "hparam/test_sens": test_m["sens"], "hparam/test_spec": test_m["spec"],
        "hparam/cv_auc_mean": result["cv_auc_mean"],
    }
    metric_dict = {k: v for k, v in metric_dict.items() if v == v}  # drop NaN
    hp_writer.add_hparams(
        {"drug": drug, "genes": "+".join(gene_order), "n_valid": valid_total,
         "epochs": epochs, "batch_size": batch_size},
        metric_dict, run_name=".",
    )
    hp_writer.close()
    return result


def _write_summary(run_dir):
    rows = []
    for jf in sorted(run_dir.glob("*.json")):
        r = json.loads(jf.read_text())
        if "cv_auc_mean" not in r:
            continue
        rows.append(_summary_row(r))
    if rows:
        pd.DataFrame(rows).round(4).to_csv(run_dir / "summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drugs", nargs="+", default=None,
                    help="drugs to evaluate (default: all in DRUG_TO_LOCI)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default=None,
                    help="name for this run's folder under results/dna_cnn/runs/ "
                         "(default: timestamp). Each run is isolated so it shows as "
                         "a distinct group in TensorBoard.")
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    tb_dir = run_dir / "tb"
    run_dir.mkdir(parents=True, exist_ok=True)

    drugs = args.drugs or list(tb.DRUG_TO_LOCI)
    print(f"Run '{run_name}' -> {run_dir}")
    print(f"Evaluating DNA-only CNN on {len(drugs)} drug(s): {drugs}")
    print(f"device={args.device} epochs={args.epochs} batch_size={args.batch_size} "
          f"n_splits={args.n_splits}")
    # point TensorBoard at RUNS_DIR so every run_name shows as its own top-level group
    print(f"TensorBoard: tensorboard --logdir {RUNS_DIR}\n")

    for drug in drugs:
        try:
            result = evaluate_drug(drug, args.epochs, args.n_splits,
                                   args.batch_size, args.device, tb_dir, args.seed)
            (run_dir / f"{drug.upper()}.json").write_text(json.dumps(result, indent=2))
            _write_summary(run_dir)
        except Exception as e:  # keep going across drugs; one bad drug shouldn't sink the run
            print(f"[{drug}] FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    _write_summary(run_dir)
    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()

"""
One-off knob sweep on MOXIFLOXACIN DNA to diagnose the majority-class collapse
seen under the baseline-aligned protocol (CV AUC 0.54, sens 0, 60-epoch run
early-stopped in ~30s). Loads the DrugData once and re-runs run_modal_cv across
a small grid of {early-stop monitor, patience, output-bias init}. Prints a table
ranked by CV AUC. Read-only w.r.t. the repo; writes nothing.

    python scripts/sweep_moxi.py --device cuda
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR
from datasets import load_dataset
from training.multimodal import run_modal_cv

# (name, kwargs-override for run_modal_cv). Baseline-aligned control first.
CONFIGS = [
    ("C0 control  (loss/p5/bias=auto)",   dict(monitor="loss", patience=5,  out_bias="auto")),
    ("C1 auc      (auc /p5/bias=auto)",   dict(monitor="auc",  patience=5,  out_bias="auto")),
    ("C2 p15      (loss/p15/bias=auto)",  dict(monitor="loss", patience=15, out_bias="auto")),
    ("C3 fullep   (loss/p999/bias=auto)", dict(monitor="loss", patience=999,out_bias="auto")),
    ("C4 nobias   (loss/p5/bias=None)",   dict(monitor="loss", patience=5,  out_bias=None)),
    ("C5 auc+p15  (auc /p15/bias=auto)",  dict(monitor="auc",  patience=15, out_bias="auto")),
    ("C6 auc+nob  (auc /p15/bias=None)",  dict(monitor="auc",  patience=15, out_bias=None)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug", default="MOXIFLOXACIN")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    print(f"Loading {args.drug} (dna)...", flush=True)
    data = load_dataset(args.drug, ["dna"], REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,
                        regulatory_dir=REAL_REGULATORY_DIR)

    rows = []
    for name, kw in CONFIGS:
        print(f"\n===== {name} =====", flush=True)
        t0 = time.time()
        r = run_modal_cv(data, epochs=args.epochs, device=args.device, seed=0, **kw)
        eps = [f["best_epoch"] for f in r["cv_folds"]]
        rows.append((name, r["cv_auc_mean"], r["cv_auc_std"], r["test"]["auc"],
                     r["test"]["sens"], r["test"]["spec"], eps, round(time.time() - t0, 1)))

    rows_sorted = sorted(rows, key=lambda x: (x[1] if x[1] == x[1] else -1), reverse=True)
    print("\n\n================ SWEEP SUMMARY (ranked by CV AUC) ================")
    print(f"{'config':34s} {'CV_AUC':>8s} {'±std':>6s} {'TEST':>7s} "
          f"{'sens':>5s} {'spec':>5s} {'best_epochs':>16s} {'sec':>6s}")
    for name, cv, sd, te, se, sp, eps, sec in rows_sorted:
        print(f"{name:34s} {cv:8.4f} {sd:6.3f} {te:7.4f} "
              f"{se:5.2f} {sp:5.2f} {str(eps):>16s} {sec:6.1f}")
    print("\nBaseline (BIG-TB SD-CNN MOXI) TEST AUC = 0.886")


if __name__ == "__main__":
    main()

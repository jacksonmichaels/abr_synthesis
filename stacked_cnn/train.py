"""
Stacked CNN, single-drug: one model per (drug, modality set).

Loads the requested modalities (DNA / protein / biophysical / regulatory, one
block per locus) via datasets.loader.load_dataset, builds the stacked CNN
(models.stacked_cnn.MDCNNNet) and runs BIG-TB's SD-CNN CV + held-out-test
protocol (training.multimodal.run_modal_cv). Results are written per drug under
a run folder, named by drug and modality set:

    results/{run_name}/{DRUG}__{modality-tag}.json   (+ _curves.png)
    results/{run_name}/summary.csv

Pass `all` to --modalities and/or --drugs to run every modality / every drug.

Examples (run from the repository root):
    python stacked_cnn/train.py --modalities dna --drugs ISONIAZID
    python stacked_cnn/train.py --modalities all --drugs all --device cuda \
        --split-dir results/splits/heldout_2 --epochs 300 --patience 30 --min-epochs 50
    python stacked_cnn/train.py --modalities dna --drugs all --loci eis embA embB \
        embC ethA ethR fabG1 gid gyrA gyrB inhA katG pncA rpoB rpoC rpsL rrl rrs tlyA
"""
import argparse
import json
import sys
import time
from pathlib import Path

# put the repository root on the path so the package imports resolve no matter
# which directory the job was launched from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402  (import after the sys.path bootstrap, by design)
import torch  # noqa: E402

import bigtb as tb  # noqa: E402
from datasets.loader import LOCUS_SETS, MODALITIES, load_dataset  # noqa: E402
from paths import (MODEL_WEIGHTS_DIR, REAL_GENOTYPE_DIR,  # noqa: E402
                   REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR, RESULTS_DIR)
from training.checkpoint import SAVE_CHOICES, write_pointer  # noqa: E402
from training.curves import save_curves  # noqa: E402
from training.multimodal import run_modal_cv  # noqa: E402

ALL_MODALITIES = list(MODALITIES)
ALL_DRUGS = list(tb.DRUG_TO_LOCI)


def _resolve_choices(requested, universe, kind, ap):
    """Expand the sentinel 'all' to the full universe, else validate."""
    picked = [x.upper() if kind == "drug" else x.lower() for x in requested]
    if "all" in (x.lower() for x in requested):
        return list(universe)
    unknown = [x for x in picked if x not in universe]
    if unknown:
        ap.error(f"unknown {kind}(s) {unknown}; choose from {list(universe)} or 'all'")
    return picked


def _summary_row(r):
    return {
        "drug": r["drug"], "modalities": r["tag"],
        "genes": "+".join(r["genes"]), "n_valid": r["n_valid"],
        "n_R": r["n_resistant"], "n_S": r["n_susceptible"],
        "cv_auc_mean": r["cv_auc_mean"], "cv_auc_std": r["cv_auc_std"],
        "cv_auc_pr_mean": r["cv_auc_pr_mean"],
        "test_auc": r["test"]["auc"], "test_auc_pr": r["test"]["auc_pr"],
        "test_sens": r["test"]["sens"], "test_spec": r["test"]["spec"],
        "seconds": r["seconds"],
    }


def _write_summary(run_dir):
    rows = []
    for jf in sorted(run_dir.glob("*.json")):
        r = json.loads(jf.read_text())
        if "cv_auc_mean" in r:
            rows.append(_summary_row(r))
    if rows:
        pd.DataFrame(rows).round(4).to_csv(run_dir / "summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modalities", nargs="+", default=["dna"], metavar="M",
                    help=f"any subset of {ALL_MODALITIES}, or 'all' (default: dna)")
    ap.add_argument("--drugs", nargs="+", default=None, metavar="DRUG",
                    help="drug(s) to run, or 'all' (default: ISONIAZID RIFAMPICIN)")
    ap.add_argument("--loci", nargs="+", default=None, metavar="GENE",
                    help="gene loci to load for dna/protein/biophysical, for every "
                         "selected drug (default: the drug's --locus-set loci)")
    ap.add_argument("--locus-set", default="bigtb", choices=list(LOCUS_SETS),
                    help="default gene loci per drug (datasets.loader.LOCUS_SETS): "
                         "'bigtb' = BIG-TB's per-drug DRUG_TO_LOCI. Ignored when "
                         "--loci is given.")
    ap.add_argument("--genotype-dir", default=None,
                    help="aligned per-gene FASTA folder to read instead of "
                         "paths.REAL_GENOTYPE_DIR")
    ap.add_argument("--split-dir", default=None,
                    help="directory of per-drug split files {DRUG}.csv (New_ID, role, "
                         "fold), e.g. lineage-grouped splits; fixes the test split "
                         "and CV folds. Default: stratified 80/20 + StratifiedKFold, "
                         "seed 42.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=15,
                    help="early-stopping patience on val AUC (default: 15)")
    ap.add_argument("--min-epochs", type=int, default=0,
                    help="hold early stopping off for this many epochs (default: 0)")
    ap.add_argument("--save-weights", default="best", choices=list(SAVE_CHOICES),
                    help="which CV fold weights to persist: 'best' (default — the "
                         "fold scored on TEST), 'all', or 'none'. Written with a "
                         f"rebuild config to {MODEL_WEIGHTS_DIR}/{{run-name}}/.")
    ap.add_argument("--weights-dir", default=None,
                    help=f"override the weights root (default: {MODEL_WEIGHTS_DIR})")
    ap.add_argument("--run-name", default=None,
                    help=f"run folder name under {RESULTS_DIR} (default: timestamp)")
    # accepted so existing job commands still parse; the stacked CNN on real
    # data is all this script runs
    ap.add_argument("--arch", default="mdcnn", choices=["mdcnn"], help=argparse.SUPPRESS)
    ap.add_argument("--real", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    modalities = _resolve_choices(args.modalities, ALL_MODALITIES, "modality", ap)
    drugs = _resolve_choices(args.drugs or ["ISONIAZID", "RIFAMPICIN"],
                             ALL_DRUGS, "drug", ap)
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    genotype_dir = args.genotype_dir or REAL_GENOTYPE_DIR
    phenotype_csv, regulatory_dir = REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR

    print(f"Run '{run_name}' -> {run_dir}")
    print(f"modalities={modalities} drugs={drugs} locus_set={args.locus_set} "
          f"loci={args.loci or 'per drug'}")
    print(f"device={args.device} epochs={args.epochs} batch_size={args.batch_size} "
          f"n_splits={args.n_splits} patience={args.patience} "
          f"min_epochs={args.min_epochs}")

    failed = []
    for drug in drugs:
        if args.split_dir and not (Path(args.split_dir) / f"{drug}.csv").is_file():
            print(f"[{drug}] skipped: no split file in {args.split_dir} "
                  "(cell not feasible for this held-out lineage)", flush=True)
            continue
        try:
            data = load_dataset(drug, modalities, genotype_dir, phenotype_csv,
                                regulatory_dir, loci=args.loci,
                                locus_set=args.locus_set)
            result = run_modal_cv(data, epochs=args.epochs, n_splits=args.n_splits,
                                  batch_size=args.batch_size, device=args.device,
                                  seed=args.seed, patience=args.patience,
                                  min_epochs=args.min_epochs,
                                  split_file=(None if args.split_dir is None else
                                              str(Path(args.split_dir) / f"{drug}.csv")),
                                  run_name=run_name,
                                  save_weights=args.save_weights,
                                  weights_dir=args.weights_dir,
                                  data_config={
                                      "genotype_dir": str(genotype_dir),
                                      "phenotype_csv": str(phenotype_csv),
                                      "regulatory_dir": str(regulatory_dir),
                                      "locus_set": args.locus_set,
                                      "loci_override": args.loci,
                                  })
            out = run_dir / f"{drug}__{result['tag']}.json"
            out.write_text(json.dumps(result, indent=2))
            write_pointer(run_dir, f"{drug}__{result['tag']}", result.get("weights_dir"))
            # per-fold loss / val-metric curves — shows whether `epochs` was enough
            save_curves(result["cv_folds"], out.with_name(f"{out.stem}_curves.png"),
                        title=f"{drug} / {result['tag']}  (epoch cap {args.epochs}, "
                              f"monitor=auc, patience={args.patience}"
                              + (f", warmup={args.min_epochs}" if args.min_epochs
                                 else "") + ")")
            _write_summary(run_dir)
        except Exception as e:  # one bad drug shouldn't sink the run...
            failed.append(drug)
            print(f"[{drug}] FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    _write_summary(run_dir)
    if failed:  # ...but the job must not report success
        sys.exit(f"FAILED drugs: {failed}")
    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()

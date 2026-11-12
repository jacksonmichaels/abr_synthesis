"""
Stacked CNN, multi-drug: one model predicts every drug at once (MD-CNN).

    results/{run_name}/multidrug__{modality-tag}.json   (+ _curves.png)
    results/{run_name}/multidrug_summary.csv   (one row per drug + MACRO)

Loci: --loci if given; else the ordered union of the drugs' --locus-set loci
for a named set other than bigtb; else every locus FASTA in the genotype dir
(MD-CNN's own drug-independent rule).

Examples (run from the repository root):
    python stacked_cnn/train_multidrug.py --modalities dna --device cuda
    python stacked_cnn/train_multidrug.py --modalities all --locus-set lab64 \
        --split-file results/splits/heldout_2/MULTIDRUG.csv --epochs 300 --patience 30 --min-epochs 50
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

from datasets.loader import LOCI_19, LOCUS_SETS, MODALITIES  # noqa: E402
from datasets.multidrug import (ALL_DRUGS, load_multidrug_dataset,  # noqa: E402
                                union_loci)
from paths import (MODEL_WEIGHTS_DIR, REAL_GENOTYPE_DIR,  # noqa: E402
                   REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR, RESULTS_DIR)
from training.checkpoint import SAVE_CHOICES, write_pointer  # noqa: E402
from training.curves import save_curves  # noqa: E402
from training.multidrug import run_multidrug_cv  # noqa: E402

ALL_MODALITIES = list(MODALITIES)


def _resolve_drugs(requested, ap):
    if not requested or any(x.lower() == "all" for x in requested):
        return list(ALL_DRUGS)
    picked = [x.upper() for x in requested]
    unknown = [x for x in picked if x not in ALL_DRUGS]
    if unknown:
        ap.error(f"unknown drug(s) {unknown}; choose from {ALL_DRUGS} or 'all'")
    return picked


def _resolve_modalities(requested, ap):
    picked = [x.lower() for x in requested]
    if "all" in picked:
        return list(ALL_MODALITIES)
    unknown = [x for x in picked if x not in ALL_MODALITIES]
    if unknown:
        ap.error(f"unknown modalit(ies) {unknown}; choose from {ALL_MODALITIES} or 'all'")
    return picked


def _write_summary(run_dir, result):
    rows = []
    for d in result["drugs"]:
        tp = result["test_per_drug"][d]
        rows.append({
            "drug": d, "cv_auc": result["cv_per_drug_auc"][d],
            "test_auc": tp["auc"], "test_auc_pr": tp["auc_pr"],
            "test_sens": tp["sens"], "test_spec": tp["spec"],
            "n_R": tp["n_R"], "n_S": tp["n_S"],
        })
    rows.append({
        "drug": "MACRO", "cv_auc": result["cv_macro_auc_mean"],
        "test_auc": result["test_macro_auc"], "test_auc_pr": result["test_macro_auc_pr"],
        "test_sens": float("nan"), "test_spec": float("nan"),
        "n_R": sum(r["n_R"] for r in rows), "n_S": sum(r["n_S"] for r in rows),
    })
    pd.DataFrame(rows).round(4).to_csv(run_dir / "multidrug_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modalities", nargs="+", default=["dna"], metavar="M",
                    help=f"any subset of {ALL_MODALITIES}, or 'all' (default: dna)")
    ap.add_argument("--drugs", nargs="+", default=None, metavar="DRUG",
                    help="drugs to predict jointly, or 'all' (default: all 11)")
    ap.add_argument("--loci", nargs="+", default=None, metavar="GENE",
                    help="gene loci to load (default: see the module docstring)")
    ap.add_argument("--locus-set", default="bigtb", choices=list(LOCUS_SETS),
                    help="a named set other than 'bigtb' loads the ordered union of "
                         "the drugs' loci in that set (datasets.loader.LOCUS_SETS); "
                         "'bigtb' (default) loads every locus on disk. Ignored when "
                         "--loci is given.")
    ap.add_argument("--genotype-dir", default=None,
                    help="aligned per-gene FASTA folder to read instead of "
                         "paths.REAL_GENOTYPE_DIR")
    ap.add_argument("--split-file", default=None,
                    help="CSV (New_ID, role train/test, fold) fixing the test split "
                         "and CV folds, e.g. the lineage-grouped MULTIDRUG.csv. "
                         "Default: random 80/20 + KFold(5), seed 42.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=15,
                    help="early-stopping patience on macro val AUC (default: 15)")
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
    # accepted so existing job commands still parse
    ap.add_argument("--arch", default="mdcnn", choices=["mdcnn"], help=argparse.SUPPRESS)
    ap.add_argument("--real", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    drugs = _resolve_drugs(args.drugs, ap)
    modalities = _resolve_modalities(args.modalities, ap)
    if args.split_file and not Path(args.split_file).is_file():
        ap.error(f"--split-file {args.split_file} does not exist")
    run_name = args.run_name or time.strftime("multidrug_%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    geno = args.genotype_dir or REAL_GENOTYPE_DIR
    pheno, reg = REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR

    print(f"Run '{run_name}' -> {run_dir}")
    print(f"modalities={modalities} drugs={len(drugs)} device={args.device} "
          f"epochs={args.epochs} patience={args.patience} min_epochs={args.min_epochs}")

    loci = args.loci
    if loci is None and args.locus_set != "bigtb":
        loci = union_loci(drugs, locus_set=args.locus_set)
        print(f"loci: --locus-set {args.locus_set} ({len(loci)}): {loci}")
    elif loci is None:
        loci = list(LOCI_19)
        print(f"loci: the 19 curated loci (MD-CNN rule): {loci}")
    data = load_multidrug_dataset(drugs, modalities, geno, pheno, reg, loci)
    result = run_multidrug_cv(data, epochs=args.epochs, n_splits=args.n_splits,
                              batch_size=args.batch_size, device=args.device,
                              seed=args.seed, patience=args.patience,
                              min_epochs=args.min_epochs,
                              run_name=run_name, save_weights=args.save_weights,
                              weights_dir=args.weights_dir,
                              split_file=args.split_file,
                              data_config={
                                  "genotype_dir": str(geno),
                                  "phenotype_csv": str(pheno),
                                  "regulatory_dir": str(reg),
                                  "drugs": drugs,
                                  "locus_set": args.locus_set,
                                  "loci_override": args.loci,
                                  "split_file": args.split_file,
                              })

    result.update(locus_set=args.locus_set, genotype_dir=str(geno),
                  regulatory_dir=str(reg))
    (run_dir / f"multidrug__{result['tag']}.json").write_text(json.dumps(result, indent=2))
    _write_summary(run_dir, result)
    # breadcrumb: the run folder records where its weights went
    write_pointer(run_dir, f"multidrug__{result['tag']}", result.get("weights_dir"))
    # per-fold loss / macro-val-AUC curves — shows whether `epochs` was enough
    save_curves(result["cv_folds"], run_dir / f"multidrug__{result['tag']}_curves.png",
                title=f"multidrug / {result['tag']}  (epoch cap {args.epochs}, "
                      f"monitor=auc, patience={args.patience}"
                      + (f", warmup={args.min_epochs}" if args.min_epochs else "") + ")")
    print(f"\nWrote {run_dir}/multidrug__{result['tag']}.json + multidrug_summary.csv")
    print("All done.", flush=True)


if __name__ == "__main__":
    main()

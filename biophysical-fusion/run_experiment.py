"""
Pipeline entry point — modality-selectable.

Loads any subset of modalities (DNA / protein / biophysical / regulatory) via
the single dataloader (datasets.load_dataset), builds a MultiModalNet with one
branch per feature block, and runs BIG-TB's SD-CNN CV + held-out-test protocol
(train_multimodal.run_modal_cv). Results are written per drug under a run
folder, named by drug *and* the modality set so runs are self-describing:

    results/experiments/{run_name}/{DRUG}__{modality-tag}.json
    results/experiments/{run_name}/summary.csv

Modes:
  --real       (default) real BIG-TB data on Unity (bigtb_ref.REAL_*).
  --synthetic  small synthetic fixtures — proves the wiring, numbers meaningless.

Pass `all` to --modalities and/or --drugs to run every modality / every drug.

Examples:
    python run_experiment.py --modalities dna
    python run_experiment.py --modalities dna biophysical --drugs ISONIAZID
    python run_experiment.py --modalities all --drugs all --device cuda
    python run_experiment.py --modalities dna protein biophysical regulatory \
        --drugs ISONIAZID --epochs 60 --device cuda
    python run_experiment.py --synthetic --modalities all --drugs all
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, tb
from datasets import DRUG_TO_REGULATORY, MODALITIES, load_dataset
from fixtures import build_fixture_dataset
from models import ENCODERS
from train_multimodal import run_modal_cv

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "experiments"
ALL_MODALITIES = list(MODALITIES)
ALL_DRUGS = list(tb.DRUG_TO_LOCI)


def _resolve_choices(requested, universe, kind, ap):
    """Expand the sentinel 'all' to the full universe, else validate the
    requested items against it. `universe` is the ordered list of valid names."""
    if any(x.lower() == "all" for x in requested):
        return list(universe)
    picked = [x.upper() if kind == "drug" else x.lower() for x in requested]
    unknown = [x for x in picked if x not in universe]
    if unknown:
        ap.error(f"unknown {kind}(s) {unknown}; choose from {universe} or 'all'")
    return picked


def _parse_encoders(tokens, ap):
    """Parse ['dna=cnn', 'protein=transformer'] into {modality: encoder}."""
    if not tokens:
        return {}
    mapping = {}
    for tok in tokens:
        if "=" not in tok:
            ap.error(f"--encoders expects MODALITY=TYPE tokens, got {tok!r}")
        mod, enc = (s.strip().lower() for s in tok.split("=", 1))
        if mod not in MODALITIES:
            ap.error(f"--encoders: unknown modality {mod!r}; choose from {list(MODALITIES)}")
        if enc not in ENCODERS:
            ap.error(f"--encoders: unknown encoder {enc!r}; choose from {list(ENCODERS)}")
        mapping[mod] = enc
    return mapping


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


def _make_synthetic(tmp, drugs, modalities, loci, regulatory_loci):
    """One fixture set covering the requested gene loci (+ regulatory regions if
    that modality is requested). Honors --loci / --regulatory-loci overrides."""
    genes = sorted(set(loci)) if loci else sorted(
        {g for d in drugs for g in tb.DRUG_TO_LOCI[d.upper()]})
    regions = []
    if "regulatory" in modalities:
        regions = sorted(set(regulatory_loci)) if regulatory_loci else sorted(
            {r for d in drugs for r in DRUG_TO_REGULATORY.get(d.upper(), [])})
        regions = [r for r in regions if r not in set(genes)]  # don't write a locus twice
    return build_fixture_dataset(
        tmp, genes=genes, drugs=[d.upper() for d in drugs],
        n_isolates=80, n_codons=40, seed=0, regulatory_regions=regions,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modalities", nargs="+", default=["dna"], metavar="M",
                    help=f"any subset of {ALL_MODALITIES}, or 'all' (default: dna)")
    ap.add_argument("--drugs", nargs="+", default=None, metavar="DRUG",
                    help="drug(s) to run, or 'all' (default: ISONIAZID RIFAMPICIN)")
    ap.add_argument("--loci", nargs="+", default=None, metavar="GENE",
                    help="which gene loci to load for dna/protein/biophysical "
                         "(default: the drug's DRUG_TO_LOCI). Applies to all "
                         "selected drugs, so usually pair with a single --drugs.")
    ap.add_argument("--regulatory-loci", nargs="+", default=None, metavar="REGION",
                    help="which regulatory regions to load (default: the "
                         "WHO-derived per-drug set).")
    ap.add_argument("--encoders", nargs="+", default=None, metavar="MODALITY=TYPE",
                    help="per-modality encoder, e.g. --encoders dna=cnn "
                         f"protein=transformer. Types: {list(ENCODERS)}. "
                         "Modalities not named use --default-encoder.")
    ap.add_argument("--default-encoder", default="cnn", choices=list(ENCODERS),
                    help="encoder for modalities not named in --encoders (default: cnn)")
    real = ap.add_mutually_exclusive_group()
    real.add_argument("--real", dest="real", action="store_true", default=True,
                      help="run on the real BIG-TB data (default)")
    real.add_argument("--synthetic", dest="real", action="store_false",
                      help="run on small synthetic fixtures instead")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tb", action="store_true",
                    help="log to TensorBoard under the run folder's tb/ dir")
    ap.add_argument("--run-name", default=None,
                    help="run folder name under results/experiments/ (default: timestamp)")
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    modalities = _resolve_choices(args.modalities, ALL_MODALITIES, "modality", ap)
    drugs = _resolve_choices(args.drugs or ["ISONIAZID", "RIFAMPICIN"],
                             ALL_DRUGS, "drug", ap)
    branch_models = _parse_encoders(args.encoders, ap)
    epochs = args.epochs if args.epochs is not None else (60 if args.real else 5)
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = (run_dir / "tb") if args.tb else None

    print(f"Run '{run_name}' -> {run_dir}")
    print(f"mode={'real' if args.real else 'synthetic'} modalities={modalities} "
          f"drugs={drugs}")
    print(f"encoders: {branch_models or '{}'} (default: {args.default_encoder})")
    print(f"device={args.device} epochs={epochs} batch_size={args.batch_size} "
          f"n_splits={args.n_splits}")
    if args.tb:
        print(f"TensorBoard: tensorboard --logdir {RESULTS_DIR}\n")

    # resolve data source (real paths, or a synthetic fixture set built once)
    import contextlib
    import tempfile
    with contextlib.ExitStack() as stack:
        if args.real:
            genotype_dir, phenotype_csv, regulatory_dir = (
                REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_GENOTYPE_DIR)
        else:
            tmp = stack.enter_context(tempfile.TemporaryDirectory())
            genotype_dir, phenotype_csv = _make_synthetic(
                tmp, drugs, modalities, args.loci, args.regulatory_loci)
            regulatory_dir = genotype_dir

        for drug in drugs:
            try:
                data = load_dataset(drug, modalities, genotype_dir,
                                    phenotype_csv, regulatory_dir=regulatory_dir,
                                    loci=args.loci, regulatory_loci=args.regulatory_loci)
                result = run_modal_cv(data, epochs=epochs, n_splits=args.n_splits,
                                      batch_size=args.batch_size, device=args.device,
                                      tb_dir=tb_dir, seed=args.seed,
                                      branch_models=branch_models,
                                      default_encoder=args.default_encoder)
                out = run_dir / f"{drug}__{result['tag']}.json"
                out.write_text(json.dumps(result, indent=2))
                _write_summary(run_dir)
            except Exception as e:  # one bad drug shouldn't sink the run
                print(f"[{drug}] FAILED: {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()

    _write_summary(run_dir)
    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()

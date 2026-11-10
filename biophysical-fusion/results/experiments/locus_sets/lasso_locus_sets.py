#!/usr/bin/env python3
"""LASSO (L1-logistic, whole aligned sequence) on the lineage-grouped split, for
the two extra locus universes ``who_t12`` and ``lab64`` (2026-10-30).

This is a thin driver over results/experiments/lineage_split/lasso_lineage.py:
the design matrix, encoding, constant-column rule, C grid, liblinear fit,
split handling (--split-dir), metrics, TEST-from-best-fold rule, JSON record
and summary.csv are all THAT module's code, imported, not copied. The only
differences are

  * the loci: ``datasets.loader.drug_loci(drug, locus_set=...)`` with
    locus_set in {who_t12, lab64} (``bigtb`` is accepted too; it is the same
    list as lasso_lineage's ``perdrug`` and exists for the encoding sanity
    check). They are handed to lasso_lineage.run_drug as an explicit locus list
    while ``locus_set`` keeps the canned name, so the record's ``tag`` is
    ``dna-onehot-{who_t12,lab64}`` and ``locus_set`` is the universe name;
  * the FASTAs come from --genotype-dir (default $W/locus_sets/aligned, the
    repaired 68-FASTA folder), not data_snapshot/genomic_data/aligned;
  * a requested locus with no FASTA in --genotype-dir is an ERROR (lasso_lineage
    only warns and drops it), unless --allow-missing; a silently shrunken
    universe would mislabel the cell;
  * the record gains one key, ``genotype_dir``; every other key is unchanged;
  * --max-loci N (debug only) keeps the first N loci and suffixes the tag
    ``_first{N}`` so a debug cell can never pass for a real one.

Usage::

    python results/experiments/locus_sets/lasso_locus_sets.py \
        --drugs CAPREOMYCIN --locus-set who_t12 \
        --split-dir results/experiments/lineage_split/splits \
        --out results/experiments/locus_sets/lasso_onehot_who_t12
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR))

_LL_PATH = PROJECT_DIR / "results/experiments/lineage_split/lasso_lineage.py"
_spec = importlib.util.spec_from_file_location("lasso_lineage", _LL_PATH)
ll = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ll)

from datasets.loader import drug_loci  # noqa: E402

W = Path("/scratch3/workspace/jacksonmicha_umass_edu-abr_synthesis")
DEFAULT_GENOTYPE_DIR = str(W / "locus_sets/aligned")
LOCUS_SET_CHOICES = ("who_t12", "lab64", "sai63", "who_t12_sai", "bigtb")


def _fasta_for(genotype_dir, locus):
    """Same lookup rule as datasets.sequences.load_sequence_df."""
    return sorted(Path(genotype_dir).glob(f"{locus}*.fasta"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drugs", nargs="+", default=["all"])
    ap.add_argument("--locus-set", choices=LOCUS_SET_CHOICES, required=True)
    ap.add_argument("--encoding", choices=("onehot", "delta"), default="onehot")
    ap.add_argument("--solver", choices=("liblinear", "saga"), default="liblinear")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--c-grid", type=float, nargs="+", default=list(ll.C_GRID))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--genotype-dir", default=DEFAULT_GENOTYPE_DIR)
    ap.add_argument("--phenotype-csv", default=ll.REAL_PHENOTYPE_CSV)
    ap.add_argument("--split-dir", default=None,
                    help="per-drug {DRUG}.csv split files (New_ID, role, fold); "
                         "overrides the stratified split")
    ap.add_argument("--allow-missing", action="store_true",
                    help="drop loci with no FASTA (with a warning) instead of "
                         "failing")
    ap.add_argument("--max-loci", type=int, default=None,
                    help="DEBUG: keep only the first N loci of the set")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    drugs = (ll.ALL_DRUGS if any(d.lower() == "all" for d in args.drugs)
             else [d.upper() for d in args.drugs])
    unknown = [d for d in drugs if d not in ll.ALL_DRUGS]
    if unknown:
        ap.error(f"unknown drug(s) {unknown}; choose from {ll.ALL_DRUGS} or 'all'")

    universe = args.locus_set
    tag_set = universe + (f"_first{args.max_loci}" if args.max_loci else "")

    plan = {}
    for drug in drugs:
        loci = drug_loci(drug, locus_set=universe)
        if args.max_loci:
            loci = loci[:args.max_loci]
        absent = [g for g in loci if not _fasta_for(args.genotype_dir, g)]
        if absent and not args.allow_missing:
            ap.error(f"{drug}/{universe}: no FASTA in {args.genotype_dir} for "
                     f"{absent} (pass --allow-missing to drop them)")
        plan[drug] = loci

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for drug in drugs:
        # run_drug takes the explicit list from args.loci; args.locus_set only
        # feeds the tag / locus_set fields, so they carry the universe name.
        args.loci = plan[drug]
        args.locus_set = tag_set
        print(f"[{drug}] locus_set={tag_set}: {len(args.loci)} loci "
              f"{'+'.join(args.loci)} from {args.genotype_dir}", flush=True)
        r = ll.run_drug(drug, args)
        r["genotype_dir"] = str(args.genotype_dir)
        (out / f"{drug}__{r['tag']}.json").write_text(json.dumps(r, indent=2))
        print(f"[{drug}] CV {r['cv_auc_mean']:.4f} +/- {r['cv_auc_std']:.4f} "
              f"@ C={r['best_C']} (support {r['cv_support_mean']:.0f} of "
              f"{r['n_features']}), TEST {r['test']['auc']:.4f} "
              f"[{r['seconds']:.0f}s]", flush=True)
        ll._write_summary(out)


if __name__ == "__main__":
    main()

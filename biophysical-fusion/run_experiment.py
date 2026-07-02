"""
End-to-end smoke test: synthetic fixtures -> dataset -> all three fusion
models -> one short CV run each. Data is synthetic (see TODO.md #3), so
this is not a real experiment — it only proves the Phase 0-4 pipeline is
wired correctly and ready the moment real data / the Kulkarni table land.
"""
import tempfile

from bigtb_ref import tb
from data import build_dataset
from fixtures import build_fixture_dataset
from models import CrossAttentionFusionCNN, EarlyFusionCNN, LateFusionCNN
from train import run_cv

DRUG = "RIFAMPICIN"

MODELS = [
    ("LateFusionCNN", LateFusionCNN),
    ("EarlyFusionCNN", EarlyFusionCNN),
    ("CrossAttentionFusionCNN", CrossAttentionFusionCNN),
]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        genotype_dir, phenotype_csv = build_fixture_dataset(
            tmp, genes=tb.DRUG_TO_LOCI[DRUG], drugs=[DRUG], n_isolates=60, n_codons=40, seed=0
        )
        dna_X, bio_Xs, gene_order, y, isolate_ids = build_dataset(DRUG, genotype_dir, phenotype_csv)
        bio_shapes = [a.shape for a in bio_Xs]
        print(f"dna_X {dna_X.shape}  bio_Xs {list(zip(gene_order, bio_shapes))}  "
              f"y {y.shape}  n_valid={(y != -1).sum()}")

        for name, cls in MODELS:
            print(f"\n=== {name} ===")
            run_cv(cls, dna_X, bio_Xs, y, drug=DRUG, n_splits=3, epochs=5)


if __name__ == "__main__":
    main()

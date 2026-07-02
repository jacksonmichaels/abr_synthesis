"""
Phase 0 — synthetic genotype/phenotype fixtures.

No real BIG-TB data exists in this checkout (see TODO.md #3). This
generates small, self-contained aligned-FASTA + phenotype CSV data, in the
exact layout `tb_cnn_codebase.make_genotype_df`/`rs_encoding_to_numeric`
expect, so every later phase is testable without the real dataset.
"""
import random
from pathlib import Path

import pandas as pd

BASES = "ACGT"


def _random_seq(n_codons, rng):
    return "".join(rng.choice(BASES) for _ in range(n_codons * 3))


def _mutate(seq, rng, n_snps, gap_run):
    seq = list(seq)
    for _ in range(n_snps):
        i = rng.randrange(len(seq))
        seq[i] = rng.choice(BASES)
    if gap_run:
        start = rng.randrange(len(seq) - gap_run)
        for i in range(start, start + gap_run):
            seq[i] = "-"
    return "".join(seq)


def make_gene_fixture(genotype_dir, gene, isolate_ids, n_codons=40, seed=0):
    """Write one {gene}_synthetic.fasta, one aligned sequence per isolate.
    Every isolate's sequence has the same length (n_codons*3): gaps stand
    in for indels/frameshifts without shortening the alignment, matching
    BIG-TB's aligned-FASTA convention."""
    rng = random.Random(seed)
    ref = _random_seq(n_codons, rng)
    path = Path(genotype_dir) / f"{gene}_synthetic.fasta"
    with open(path, "w") as fh:
        for iso in isolate_ids:
            # occasional in-frame indel (3/6 gap columns) or a lone gap
            # (stands in for a frameshift: one broken codon, rest in frame)
            gap_run = rng.choice([0, 0, 0, 1, 3, 6])
            seq = _mutate(ref, rng, n_snps=rng.randint(0, 4), gap_run=gap_run)
            fh.write(f">{iso}\n{seq}\n")
    return path


def make_phenotype_fixture(phenotype_csv, isolate_ids, drugs, seed=0, missing_frac=0.1):
    rng = random.Random(seed)
    rows = {
        iso: {d: (rng.choice(["R", "S"]) if rng.random() > missing_frac else "") for d in drugs}
        for iso in isolate_ids
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "New_ID"
    df.to_csv(phenotype_csv)
    return phenotype_csv


def build_fixture_dataset(root, genes, drugs, n_isolates=60, n_codons=40, seed=0):
    """Write a synthetic genotype dir + phenotype csv under `root`.
    Returns (genotype_dir, phenotype_csv)."""
    root = Path(root)
    genotype_dir = root / "genotype"
    genotype_dir.mkdir(parents=True, exist_ok=True)
    isolate_ids = [f"ISO{idx:04d}" for idx in range(n_isolates)]
    for i, gene in enumerate(genes):
        make_gene_fixture(genotype_dir, gene, isolate_ids, n_codons=n_codons, seed=seed + i)
    phenotype_csv = root / "phenotype.csv"
    make_phenotype_fixture(phenotype_csv, isolate_ids, drugs, seed=seed)
    return genotype_dir, phenotype_csv

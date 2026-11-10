"""The `who_t12` and `lab64` locus sets, and that adding them moved nothing else.

    cd biophysical-fusion && python -m genomefusion.tests.test_named_sets

Reads the cached real partition (fast, no VCFs, no GPU).
"""
import numpy as np

from datasets.loader import LAB64_LOCI, drug_loci
from genomefusion.partition import WORKSPACE, load_or_build
from genomefusion.train import ALL_DRUGS, base_19r, locus_subset, parse_args

CACHE_V2 = WORKSPACE / "genomefusion_cache_v3"
C15T = 1_673_424


def _p():
    return load_or_build(CACHE_V2)


def test_plain_sets_are_gene_only():
    p = _p()
    for drug in ALL_DRUGS:
        for which in ("who_t12", "lab64"):
            kinds = {p.loci[int(i)].kind for i in locus_subset(p, which, drug=drug)}
            assert "intergenic" not in kinds, (which, drug, kinds)
    print("ok  plain sets are gene loci only")


def test_r_is_a_superset():
    p = _p()
    for drug in ALL_DRUGS:
        for which in ("who_t12", "lab64"):
            a, b = locus_subset(p, which, drug=drug), locus_subset(p, which + "r", drug=drug)
            assert set(a) < set(b), (which, drug)
    print("ok  r variants strictly contain their plain sets")


def test_lab64_is_drug_independent_and_complete():
    p = _p()
    sets = {tuple(locus_subset(p, "lab64", drug=d)) for d in ALL_DRUGS}
    assert len(sets) == 1
    assert len(next(iter(sets))) == len(LAB64_LOCI), len(next(iter(sets)))
    print(f"ok  lab64 is {len(LAB64_LOCI)} loci for every drug")


def test_r_sees_c15t():
    """The dominant ETO marker must be reachable, or the arm is not a fair test."""
    p = _p()
    locus = int(p.locus_of[C15T])
    for which in ("who_t12r", "lab64r"):
        assert locus in set(locus_subset(p, which, drug="ETHIONAMIDE").tolist()), which
    print(f"ok  c-15t (locus {p.loci[locus].name}) is in who_t12r and lab64r for ETO")


def test_named_genes_are_the_right_genes():
    p = _p()
    for drug in ALL_DRUGS:
        names = set(drug_loci(drug, locus_set="who_t12"))
        for i in locus_subset(p, "who_t12", drug=drug):
            L = p.loci[int(i)]
            assert L.gene in names or L.name in names, (drug, L.name, L.gene)
    print("ok  who_t12 loci all carry a requested gene name")


def test_existing_sets_unchanged():
    """`19r` and the ladder define running jobs; this change must not move them."""
    p = _p()
    assert np.array_equal(locus_subset(p, "19r"), base_19r(p))
    assert len(locus_subset(p, "size:6776", ladder_seed=0)) == len(p.loci)
    print("ok  19r and size:N unchanged by the new `drug` argument")


def test_cli_accepts_the_new_sets():
    for which in ("who_t12", "who_t12r", "lab64", "lab64r"):
        a = parse_args(["--drugs", "ETHIONAMIDE", "--out", "x", "--split-dir", "y",
                        "--locus-set", which])
        assert a.locus_set == which
    print("ok  --locus-set accepts the new names")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()

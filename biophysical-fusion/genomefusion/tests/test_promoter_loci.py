"""The `19p` locus set: does it put back the bases `19` drops?

    cd biophysical-fusion && python -m genomefusion.tests.test_promoter_loci

Reads the cached real partition (fast, no VCFs). The load-bearing assertion is
`test_c15t`: the fabG1-inhA c-15t at 1,673,424 is the dominant ethionamide
marker in this cohort and v3 carries it inside `dna:fabG1`, so a parity arm
that cannot see it is not at parity.
"""
import numpy as np

from genomefusion.partition import CACHE, load_or_build
from genomefusion.train import locus_subset

#: the fabG1-inhA promoter c-15t; ethionamide carrier rate 0.44 R vs 0.09 S
C15T = 1_673_424
#: eis c-14t, the kanamycin/amikacin promoter marker
EIS_PROMOTER = 2_715_333


def _p():
    return load_or_build(CACHE.parent / "genomefusion_cache_v3")


def test_19_has_no_intergenic():
    """`is_19` is set only while gene loci are built, so `19` is gene-only."""
    p = _p()
    s = locus_subset(p, "19")
    kinds = {p.loci[int(i)].kind for i in s}
    assert "intergenic" not in kinds, kinds
    assert len(s) == 19, len(s)
    print(f"ok  19      {len(s)} loci, kinds {sorted(kinds)}")


def test_19p_is_a_superset():
    p = _p()
    a, b = locus_subset(p, "19"), locus_subset(p, "19p")
    assert set(a) <= set(b), "19p must contain every locus of 19"
    added = sorted(set(b) - set(a))
    assert added, "19p added nothing"
    assert all(p.loci[i].kind == "intergenic" for i in added)
    bp = sum(p.loci[i].end - p.loci[i].start for i in added)
    print(f"ok  19p     {len(b)} loci = 19 + {len(added)} intergenic ({bp:,} bp)")


def test_c15t():
    """The bases that matter are in 19p and absent from 19."""
    p = _p()
    a, b = set(locus_subset(p, "19")), set(locus_subset(p, "19p"))
    for name, pos in (("fabG1-inhA c-15t", C15T), ("eis c-14t", EIS_PROMOTER)):
        li = int(p.locus_of[pos])
        assert li >= 0, f"{name} at {pos} is masked"
        assert li not in a, f"{name} unexpectedly already in the 19"
        assert li in b, f"{name} at {pos} (locus {li}) missing from 19p"
        print(f"ok  {name}: locus {li} not in 19, in 19p")


def test_flank_monotone():
    """A wider flank can only add loci."""
    p = _p()
    prev = set()
    for f in (0, 50, 100, 200):
        cur = set(p.promoter_loci(f))
        assert prev <= cur, f"flank {f} dropped loci"
        prev = cur
    assert not set(p.promoter_loci(0)), "a 0 bp flank should add nothing"
    print("ok  flank    monotone in width, empty at 0 bp")


def test_minus19p_is_clean():
    """`minus19p` must exclude the promoters; `minus19` demonstrably does not."""
    p = _p()
    a = set(locus_subset(p, "19p"))
    clean = set(locus_subset(p, "minus19p"))
    dirty = set(locus_subset(p, "minus19"))
    assert not (clean & a), "minus19p overlaps 19p"
    assert clean | a == set(range(len(p.loci))), "minus19p + 19p is not the genome"
    prom = set(p.promoter_loci(100))
    assert not (clean & prom), "minus19p still holds promoter loci"
    assert prom <= dirty, "expected minus19 to hold every promoter locus"
    for name, pos in (("fabG1-inhA c-15t", C15T), ("eis c-14t", EIS_PROMOTER)):
        li = int(p.locus_of[pos])
        assert li in dirty and li not in clean, name
    print(f"ok  minus19p {len(clean)} loci, 0 promoter loci "
          f"(minus19 has {len(prom & dirty)}/{len(prom)})")


def test_strand_direction():
    """A minus-strand gene's promoter is past its `end`, not before `start`."""
    p = _p()
    minus = [i for i, L in enumerate(p.loci) if L.is_19 and L.strand < 0]
    assert minus, "no minus-strand gene among the 19"
    for i in minus:
        L = p.loci[i]
        got = p.promoter_loci(100, genes=[i])
        for x in got:
            assert p.loci[x].start >= L.end, (
                f"{L.gene} is on the minus strand; locus {x} at "
                f"{p.loci[x].start} is 3' of it, not 5'")
    print(f"ok  strand   {len(minus)} minus-strand genes take their 5' window past `end`")


def test_regulatory_annotation():
    """WHO regions tag loci without moving a base."""
    p = _p()
    if not p.meta.get("regulatory"):
        print("SKIP regulatory: cache predates the annotation (rebuild for a v2 cache)")
        return
    reg = [L for L in p.loci if L.is_regulatory]
    assert reg, "no locus tagged regulatory"
    # the partition must still be one-locus-per-base
    assert len({L.index for L in p.loci}) == len(p.loci)
    # c-15t's locus must know it acts on fabG1 or inhA
    li = p.loci[int(p.locus_of[C15T])]
    assert li.is_regulatory, "the c-15t locus is not tagged regulatory"
    assert {"fabG1", "inhA"} & set(li.regulates), li.regulates
    coding = [L for L in reg if L.kind != "intergenic"]
    print(f"ok  regul    {len(reg)} loci tagged ({len(coding)} coding -- these are why "
          f"a region cannot BE a locus); c-15t locus regulates {li.regulates}")


def test_regulated_edge():
    """Every tagged regulatory locus should resolve to the locus it acts on."""
    p = _p()
    if not p.meta.get("regulatory"):
        print("SKIP edge: cache predates the annotation")
        return
    idx = p.regulated_index()
    assert idx.shape == (len(p.loci),)
    li = int(p.locus_of[C15T])
    tgt = int(idx[li])
    assert tgt >= 0, "the c-15t locus resolves to no gene"
    assert p.loci[tgt].gene in ("fabG1", "inhA"), p.loci[tgt].gene
    assert not (idx == np.arange(len(p.loci))).any(), "a locus regulates itself"
    print(f"ok  edge     {int((idx >= 0).sum()):,}/{len(p.loci):,} loci carry an edge; "
          f"c-15t locus -> {p.loci[tgt].gene}")


def test_19r_is_distance_free():
    p = _p()
    if not p.meta.get("regulatory"):
        print("SKIP 19r: cache predates the annotation")
        return
    from genomefusion.train import base_19r
    s = base_19r(p)
    assert set(np.flatnonzero([L.is_19 for L in p.loci])) <= set(s.tolist())
    assert int(p.locus_of[C15T]) in s.tolist(), "19r lost the c-15t locus"
    print(f"ok  19r      {len(s)} loci, chosen by WHO catalogue not by distance")


def test_ladder_is_nested():
    """Each rung must be a subset of the next, or the ladder is not monotone."""
    p = _p()
    if not p.meta.get("regulatory"):
        print("SKIP ladder: cache predates the annotation")
        return
    from genomefusion.train import size_subset
    prev, n_all = None, len(p.loci)
    sizes = [64, 128, 256, 512, 1024, 2048, 4096, 6776]
    for n in sizes:
        cur = set(size_subset(p, n, seed=0).tolist())
        assert len(cur) == min(n, n_all), f"size:{n} gave {len(cur)} loci"
        if prev is not None:
            assert prev <= cur, f"size:{n} is not a superset of the previous rung"
        prev = cur
    assert prev == set(range(n_all)), "the top rung is not the whole genome"
    # and a different seed must give a different (still nested) ladder
    assert set(size_subset(p, 128, seed=1).tolist()) != set(size_subset(p, 128, 0).tolist())
    print(f"ok  ladder   {sizes} nested, top rung = all {n_all:,} loci")


if __name__ == "__main__":
    for fn in (test_19_has_no_intergenic, test_19p_is_a_superset, test_c15t,
               test_flank_monotone, test_minus19p_is_clean, test_strand_direction,
               test_regulatory_annotation, test_regulated_edge, test_19r_is_distance_free,
               test_ladder_is_nested):
        fn()
    print("\nall promoter-locus checks passed")

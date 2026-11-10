"""
Tokenizer checks against real VCFs and the H37Rv annotation (~1 min).

    cd biophysical-fusion && python -m genomefusion.tests.test_tokenizer

These are the facts a wrong token would quietly break: the codon grid, the
famous resistance codons, frameshift arithmetic, deletions that cross a locus
boundary, and the promoter geometry that replaced v5's upstream windows.
"""
import numpy as np

from datasets.biochem import CODON_TABLE
from genomefusion.layout import CONSEQUENCES, FRAMESHIFT, INFRAME_INDEL, RESIDUE_SYMBOLS
from genomefusion.partition import CACHE as _CACHE, LOCI_19, load_or_build

CACHE = _CACHE.parent / "genomefusion_cache_v3"
from genomefusion.reference import load_reference, reference_codons
from genomefusion.tokenizer import tokenize_isolate
from genomefusion.build_tokens import VCF_DIR, VCF_SUFFIX, phenotype_ids

P = load_or_build(CACHE)
REF = load_reference(CACHE)
REF_ARR = np.frombuffer(REF, np.uint8)


def locus(gene):
    return next(L for L in P.loci if L.gene == gene)


def test_partition_covers_each_base_once():
    assigned = P.locus_of >= 0
    assert assigned.sum() + P.meta["masked_bases"] == P.genome_len
    assert len(P.loci) == P.meta["n_loci"]
    print(f"test_partition_covers_each_base_once: ok "
          f"({P.meta['n_loci']:,} loci, {P.meta['masked_frac']:.1%} masked)")


def test_all_19_loci_present():
    got = {L.gene for L in P.loci if L.is_19}
    assert got == set(LOCI_19), set(LOCI_19) - got
    print("test_all_19_loci_present: ok")


def test_reference_codons():
    """The codon grid must reproduce the residues the literature names."""
    cases = [("katG", 314, "S"), ("rpoB", 449, "S"), ("rpoB", 444, "H"),
             ("pncA", 0, "M"), ("gyrA", 93, "D"), ("embB", 305, "M")]
    for gene, codon, want in cases:
        _, cods = reference_codons(locus(gene), P, REF)
        got = CODON_TABLE.get(bytes(cods[codon]).decode(), "X")
        assert got == want, f"{gene} codon {codon + 1}: got {got}, want {want}"
    print(f"test_reference_codons: ok ({len(cases)} codons, 1-based S315/S450/H445/M1/D94/M306)")


def test_promoter_geometry():
    """The inhA-promoter base is upstream of fabG1, in an intergenic locus."""
    i = 1673424                      # 1,673,425 1-based: the inhA-promoter c-15t
    L = P.loci[P.locus_of[i]]
    assert L.kind == "intergenic", L.kind
    assert P.up_plus[i] == 15, P.up_plus[i]
    assert L.up_genes[1] == locus("fabG1").name, L.up_genes
    print(f"test_promoter_geometry: ok ({L.name}: 15 bp upstream of fabG1, "
          f"{P.up_minus[i]} bp upstream of {L.up_genes[0]})")


def test_one_locus_per_base():
    """No genome position may belong to two loci."""
    rng = np.random.default_rng(0)
    pos = rng.integers(0, P.genome_len, 200_000)
    li = P.locus_of[pos]
    ok = li >= 0
    starts = np.asarray([L.start for L in P.loci])
    ends = np.asarray([L.end for L in P.loci])
    inside = (pos[ok] >= starts[li[ok]]) & (pos[ok] < ends[li[ok]])
    assert inside.all(), f"{(~inside).sum()} positions outside their locus span"
    print(f"test_one_locus_per_base: ok ({ok.sum():,} sampled positions)")


def test_tokenize_real_isolate():
    sid = phenotype_ids()[0]
    tok = tokenize_isolate(VCF_DIR / f"{sid}{VCF_SUFFIX}", P, REF, ref_arr=REF_ARR)
    s = tok.stats
    assert s["n_tokens"] > 100, s
    assert len(tok.locus) == s["n_tokens"]
    assert (np.diff(tok.locus) >= 0).all(), "tokens must be sorted by locus"
    assert tok.sum_n.sum() == s["n_tokens"]
    assert len(np.unique(tok.locus)) == len(tok.sum_locus)
    assert np.array_equal(np.unique(tok.locus), tok.sum_locus)
    assert (tok.span >= 1).all()
    print(f"test_tokenize_real_isolate: ok ({sid}: {s['n_tokens']} tokens over "
          f"{s['n_active_loci']} loci, {s['n_masked']} events masked)")


def test_frameshift_arithmetic():
    """Every indel token's consequence must follow span % 3."""
    n_fs = n_if = 0
    for sid in phenotype_ids()[:12]:
        tok = tokenize_isolate(VCF_DIR / f"{sid}{VCF_SUFFIX}", P, REF, ref_arr=REF_ARR)
        ind = np.isin(tok.consequence, [FRAMESHIFT, INFRAME_INDEL])
        for sp, c in zip(tok.span[ind], tok.consequence[ind]):
            want = FRAMESHIFT if sp % 3 else INFRAME_INDEL
            assert c == want, f"span {sp} called {CONSEQUENCES[c]}"
        n_fs += int((tok.consequence == FRAMESHIFT).sum())
        n_if += int((tok.consequence == INFRAME_INDEL).sum())
    assert n_fs and n_if, (n_fs, n_if)
    print(f"test_frameshift_arithmetic: ok ({n_fs} frameshift, {n_if} in-frame over 12 isolates)")


def test_truncation_range():
    for sid in phenotype_ids()[:8]:
        tok = tokenize_isolate(VCF_DIR / f"{sid}{VCF_SUFFIX}", P, REF, ref_arr=REF_ARR)
        assert (tok.truncation >= 0).all() and (tok.truncation <= 1).all()
        zero = ~np.isin(tok.consequence, [FRAMESHIFT, 2, 5])   # nonsense, start_loss
        assert (tok.truncation[zero] == 0).all(), "truncation set off a truncating consequence"
    print("test_truncation_range: ok")


def test_deletion_split_across_loci():
    """A deletion crossing a locus boundary leaves one token in each locus.

    Only DELETION spans are bounded by the locus. An insertion's `span` is the
    length of the inserted sequence, which occupies no reference bases at all,
    so it may be far longer than the locus it sits in.
    """
    from genomefusion.tokenizer import NT_INDEX
    sid = phenotype_ids()[0]
    tok = tokenize_isolate(VCF_DIR / f"{sid}{VCF_SUFFIX}", P, REF, ref_arr=REF_ARR)
    dels = (tok.span > 1) & (tok.base == NT_INDEX["-"])
    assert dels.sum() > 1, "this isolate should carry multi-base deletions"
    for li, sp, pos in zip(tok.locus[dels], tok.span[dels], tok.pos[dels]):
        L = P.loci[li]
        assert L.start <= pos < L.end, f"deletion token at {pos} outside {L.name}"
        assert pos + sp <= L.end, f"deletion span {sp} at {pos} runs past {L.name}"
        # `span` counts the bases this locus lost. The run may be broken by a
        # masked hole (the RLC mask carves holes inside genes), so the bases it
        # covers must be this locus's or masked -- never another locus's.
        seen = P.locus_of[pos:pos + sp]
        other = int(((seen != li) & (seen >= 0)).sum())
        assert other == 0, f"deletion span {sp} at {pos} covers {other} bases of another locus"
    ins = (tok.span > 1) & (tok.base != NT_INDEX["-"])
    print(f"test_deletion_split_across_loci: ok ({int(dels.sum())} deletion spans, max "
          f"{int(tok.span[dels].max())} bp; {int(ins.sum())} insertion spans, not bounded)")


def test_residues_are_real_symbols():
    sid = phenotype_ids()[3]
    tok = tokenize_isolate(VCF_DIR / f"{sid}{VCF_SUFFIX}", P, REF, ref_arr=REF_ARR)
    r = tok.residue[tok.residue >= 0]
    assert (r < len(RESIDUE_SYMBOLS)).all()
    cds = tok.region == 0
    assert (tok.consequence[cds] >= 0).all(), "a coding token with no consequence"
    assert (tok.consequence[~cds] == -1).all(), "a non-coding token with a consequence"
    print("test_residues_are_real_symbols: ok")


if __name__ == "__main__":
    test_partition_covers_each_base_once()
    test_all_19_loci_present()
    test_reference_codons()
    test_promoter_geometry()
    test_one_locus_per_base()
    test_tokenize_real_isolate()
    test_frameshift_arithmetic()
    test_truncation_range()
    test_deletion_split_across_loci()
    test_residues_are_real_symbols()
    print("all tokenizer tests passed")

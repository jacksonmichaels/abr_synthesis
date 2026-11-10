"""
v4 tokens == v3 tokens on every channel the two share, on real isolates.

    cd biophysical-fusion
    python -m locusfusion_v4.tests.test_parity      # ~2 min, ~8 GB

Builds v3 (LocusFusionV2Net, --lf-unknown-symbols, i.e. LocusFusion v3) from the
loader's blocks over five loci with promoters, and compares its `_locus_tokens`
against tokenizer.build_tokens slot by slot: the base (dna / regulatory), bio,
d_bio, phase, coordinate, which slots are filled, and the [NULL] count and
fraction. Also checks the consequence channel on the two isolates the pncA
substitution scan presented (A25A synonymous, S66* nonsense).

Needs the real data (ABR_GENOTYPE_DIR / ABR_PHENOTYPE_CSV / ABR_GENBANK, or the
/project defaults); skipped without it.
"""
import functools
from pathlib import Path

import numpy as np
import torch

from bigtb_ref import REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR
from locusfusion_v4.layout import NONSENSE, SLOTS, SYN
from locusfusion_v4.tokenizer import build_tokens, regulatory_for

LOCI = ["pncA", "katG", "rrs", "fabG1", "inhA"]
DRUGS = ["PYRAZINAMIDE", "ISONIAZID"]
N_CHECK = 3000

HAVE_DATA = Path(REAL_GENOTYPE_DIR).exists()


@functools.cache
def both():
    from datasets import load_dataset
    from models.locusfusion_v2 import LocusFusionV2Net
    reg = regulatory_for(DRUGS, LOCI)
    data = load_dataset("ISONIAZID", ["dna", "protein", "biophysical", "regulatory"],
                        REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV,
                        regulatory_dir=REAL_REGULATORY_DIR, loci=LOCI, regulatory_loci=reg,
                        per_modality_branch=False, delta=True, protein_frame="reference",
                        unknown_symbols=True, verbose=False)
    net = LocusFusionV2Net.from_blocks(data.blocks, unknown_symbols=True)
    ts = build_tokens(DRUGS, REAL_GENOTYPE_DIR, REAL_PHENOTYPE_CSV, REAL_REGULATORY_DIR,
                      loci=LOCI, regulatory=reg, verbose=False)
    assert ts.isolate_ids == data.isolate_ids
    # isolates with the most tokens first, then a random sample of the rest
    rng = np.random.default_rng(0)
    busy = np.argsort(-ts.n_occ.sum(1))[:N_CHECK // 2]
    rest = rng.choice(ts.n, N_CHECK // 2, replace=False)
    rows = np.unique(np.concatenate([busy, rest]))
    xs = [torch.from_numpy(a[rows]).float() for a in data.arrays()]
    with torch.no_grad():
        feat3, coord3, valid3, stats3 = net._locus_tokens(xs)
    loci3 = net.loci
    order = [loci3.index(l) for l in ts.loci]
    return ts, rows, net.layout, (feat3[:, order].numpy(), coord3[:, order].numpy(),
                                  valid3[:, order].numpy(), stats3[:, order].numpy())


def test_same_slots_and_null_stats():
    ts, rows, _, (_, _, valid3, stats3) = both()
    assert (valid3[:, :, 1:] == ts.valid[rows]).all()
    assert np.allclose(stats3[..., 0], np.log1p(ts.n_occ[rows]))
    assert np.allclose(stats3[..., 1], ts.n_occ[rows] / np.maximum(ts.kept, 1))


def test_shared_channels():
    ts, rows, lay, (feat3, coord3, valid3, _) = both()
    v = ts.valid[rows]
    f4, f3 = ts.feat[rows][v], feat3[:, :, 1:][v]
    pairs = {"dna": lay["slots"]["dna"], "regulatory": lay["slots"]["regulatory"],
             "bio": lay["slots"]["biophysical"], "d_bio": lay["d_bio"],
             "phase": (lay["f_phase"], lay["f_phase"] + 3)}
    for name, (a, b) in pairs.items():
        lo, hi = SLOTS[name]
        bad = ~np.isclose(f4[:, lo:hi], f3[:, a:b], atol=1e-5).all(1)
        assert not bad.any(), f"{name}: {int(bad.sum())} of {len(bad)} tokens differ"
    assert np.allclose(ts.coord[rows][v], coord3[:, :, 1:][v], atol=1e-4)


def test_consequence_on_presented_isolates():
    ts = both()[0]
    li = ts.loci.index("pncA")
    for iso, want in (("SAMEA1118274", SYN), ("SAMN05915262", NONSENSE)):
        i = ts.isolate_ids.index(iso)
        cons = ts.feat[i, li][ts.valid[i, li], SLOTS["consequence"][0]:SLOTS["consequence"][1]]
        assert cons[:, want].any(), (iso, cons)


def test_site_pos_codon_agree():
    ts = both()[0]
    for li, kind in enumerate(ts.locus_kind):
        if kind != "protein":
            continue
        v = ts.valid[:, li] & (ts.codon[:, li] >= 0)
        assert (ts.site_pos[:, li][v] == ts.codon[:, li][v]).all(), ts.loci[li]


def test_candidates_are_synonymous_and_excluded():
    """Re-derive every candidate from the CDS window: its column is base `phase`
    of reference codon `codon`, its base differs from H37Rv's, and the swapped
    codon codes for the same residue."""
    from datasets import cds
    from datasets.biochem import CODON_TABLE
    from datasets.sequences import reference_row
    from locusfusion_v4.layout import NT_SYMBOLS
    ts = both()[0]
    c = ts.cand
    base = c["feat"][:, SLOTS["dna"][0]:SLOTS["dna"][1]].argmax(1)
    phase = c["feat"][:, SLOTS["phase"][0]:SLOTS["phase"][1]].argmax(1)
    checked = 0
    for li, locus in enumerate(ts.loci):
        m = np.nonzero(c["locus"] == li)[0]
        if not len(m):
            continue
        ref = reference_row(REAL_GENOTYPE_DIR, locus)
        s0, e0 = cds.cds_columns(REAL_GENOTYPE_DIR, locus)
        cols = [i for i in range(s0, e0) if ref[i] != "-"]
        for i in m:
            k, ph = int(c["codon"][i]), int(phase[i])
            cc = cols[3 * k:3 * k + 3]
            assert cc[ph] == c["col"][i], (locus, k)
            cod = [ref[x] for x in cc]
            alt = cod.copy()
            alt[ph] = NT_SYMBOLS[base[i]]
            assert alt != cod and CODON_TABLE["".join(alt)] == CODON_TABLE["".join(cod)] != "*"
            checked += 1
    assert checked == len(c["locus"])
    fab = ts.loci.index("fabG1")
    assert not ((c["locus"] == fab) & (c["codon"] == 202)).any()      # L203L


if __name__ == "__main__":
    from locusfusion_v4.tests._run import run
    run(globals(), None if HAVE_DATA else "real genotype data not reachable")

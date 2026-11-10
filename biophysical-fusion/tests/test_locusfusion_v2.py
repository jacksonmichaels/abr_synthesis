"""
Checks for ``models.LocusFusionV2Net`` — the one-token-per-changed-base tokenizer.

Unit tests on synthetic tensors with hand-built frames (no data, no training),
plus one guarded check on the real reference rows that is skipped when the BIG-TB
data is not mounted. Seconds on CPU:

    python tests/test_locusfusion_v2.py

The claims being asserted — each one a thing this tokenizer promises that v1
did not, or a failure it was built to avoid:

  * **A locus that matches H37Rv presents only its [NULL] token.** Nothing else
    is emitted, and isolates identical to the reference get identical logits.
  * **Only nucleotide blocks make tokens.** A protein change with no DNA change
    behind it creates nothing; protein and biophysical ride on DNA tokens.
  * **The residue lands on its own codon, gap-aware.** The column -> codon map
    counts reference BASES, so a gap column upstream in the reference row moves
    the codon boundary, and a synonymous change carries no residue.
  * **The promoter is placed by genome position, on either strand**, and the kept
    axis runs 5' -> 3'.
  * **One base, one token, within a locus.** Promoter columns the gene record
    also covers are dropped, so a shared variant is one token — and, by decision,
    a base shared ACROSS loci stays a token in each.
  * **The cap is per locus**, shared by promoter and record, promoter first.
  * **rRNA genes carry no codons**: a base and a position only.
  * **Residues are on the H37Rv codon grid.** An indel changes only the codons
    it touches, a stop codon is a residue (the 21st protein channel), and
    isolate-frame protein input is refused at construction.
  * **Nothing is called wild type.** The per-locus token is [NULL].
  * Fallback without frames warns; dense input warns; a checkpoint rebuilds the
    same maps and the same logits; every parameter gets a gradient.
  * **Structure is a row of the distance matrix.** Each located token carries its
    C-alpha distance to every other variant token of its locus, with a mask that
    is off for promoter, rRNA, stop and [NULL]; the base 48 channels -- and so
    every existing checkpoint -- are untouched, and the fold survives config.json.
"""
import math
import sys
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from models import (LOCUSFUSION_V2_DEFAULTS, LocusFusionV2Net,  # noqa: E402
                    locusfusion_defaults)
from models.locusfusion_v2 import (C_TOK, D_BIO, F_IS_NULL,  # noqa: E402
                                   F_PHASE, RESIDUE_PROPERTIES, SLOTS,
                                   STRUCT_LENGTH, _RES_ORDER)
from training.checkpoint import build_model_from_config, model_config  # noqa: E402

_RESULTS = []

# --- a synthetic cohort with REAL-SHAPED frames ------------------------------
# gA, + strand. Record: 120 columns, H37Rv gap at column 45, genome 1000..1119.
#     CDS columns [30, 91) hold 60 bases = 20 codons. Promoter: 40 columns at
#     genome 980..1019, so its last 20 columns are the record's first 20 bases.
# gB, - strand. Record: 90 columns, genome 5000..5089 stored reverse-complement,
#     CDS columns [15, 75) = 20 codons. Promoter: 30 columns at genome 5100..5129,
#     upstream of a minus-strand gene means HIGHER genome coordinates.
# rrX, an rRNA gene: 60 columns, no CDS, and a stub protein block like the one
#     datasets/protein.py's whole-record fallback produces.
KEYS = [("dna", "gA"), ("dna", "gB"), ("dna", "rrX"),
        ("protein", "gA"), ("protein", "gB"), ("protein", "rrX"),
        ("biophysical", "gA"),
        ("regulatory", "gA"), ("regulatory", "gB")]
SPECS = [(5, 120), (5, 90), (5, 60),
         (21, 20), (21, 20), (21, 5),
         (3, 20),
         (5, 40), (5, 30)]
# H37Rv's residue at each of the 20 codons, as the aa blocks carry it on
# FeatureBlock.meta. Distinct residues so a filled slot names a real one:
# alanine at every codon but codon 5, which is serine.
REF_AA = "A" * 5 + "S" + "A" * 14
FRAMES = [
    {"ref_gap_cols": [45], "genome_start": 1000, "genome_end": 1119, "strand": "+",
     "cds_start": 30, "cds_end": 91, "has_cds": True},
    {"ref_gap_cols": [], "genome_start": 5000, "genome_end": 5090, "strand": "-",
     "cds_start": 15, "cds_end": 75, "has_cds": True},
    {"ref_gap_cols": [], "genome_start": 9000, "genome_end": 9060, "strand": "+",
     "cds_start": 10, "cds_end": 60, "has_cds": False},
    {"ref_residues": REF_AA}, {"ref_residues": REF_AA}, {"ref_residues": "AAAAA"},
    {"ref_residues": REF_AA},
    {"ref_gap_cols": [], "genome_start": 980, "genome_end": 1020, "strand": "+"},
    {"ref_gap_cols": [], "genome_start": 5100, "genome_end": 5130, "strand": "-"},
]
B = {key: i for i, key in enumerate(KEYS)}           # block index by (modality, locus)


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
        _RESULTS.append(True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        _RESULTS.append(False)


def reference_batch(b=4):
    """Delta encoding of isolates identical to H37Rv at every locus."""
    return [torch.zeros(b, c, length) for c, length in SPECS]


def put(xs, key, column, channel=0, value=1.0, row=slice(None)):
    xs = [x.clone() for x in xs]
    xs[B[key]][row, channel, column] = value
    return xs


def net(**kw):
    torch.manual_seed(0)
    kw.setdefault("block_frames", FRAMES)
    m = LocusFusionV2Net(KEYS, SPECS, **kw)
    m.eval()
    return m


def maps(m, locus):
    li = m.loci.index(locus)
    return {k: getattr(m, f"{k}_map_{li}").numpy()
            for k in ("coord", "codon", "phase", "keep")}, m._plan[li]


def token(m, xs, locus, slot, row=0):
    """(feature vector, coord, valid) of one variant slot (1-based after [NULL])."""
    feat, coord, valid, _ = m._locus_tokens(xs)
    li = m.loci.index(locus)
    return feat[row, li, slot], float(coord[row, li, slot]), bool(valid[row, li, slot])


# --- the per-locus token ---------------------------------------------------

def _test_reference_matching_locus_is_null_only():
    m = net()
    xs = reference_batch()
    feat, _, valid, stats = m._locus_tokens(xs)
    assert valid[:, :, 0].all(), "a [NULL] slot is not always valid"
    assert not valid[:, :, 1:].any(), "an isolate identical to H37Rv produced variant tokens"
    assert (feat[:, :, 0, F_IS_NULL] == 1).all()
    assert (stats == 0).all(), "stats of a reference-matching locus are not zero"
    logits = m(xs)
    assert torch.allclose(logits, logits[:1].expand_as(logits), atol=1e-5), \
        "identical isolates got different logits"


def _test_nothing_is_called_wild_type():
    m = net()
    names = [n for n, _ in m.named_parameters()] + [n for n, _ in m.named_buffers()]
    bad = [n for n in names
           if any("wt" in part.split("_") for part in n.lower().split("."))]
    assert not bad, f"parameters still named for wild type: {bad}"
    import models.locusfusion_v2 as mod
    assert not hasattr(mod, "F_IS_WT"), "F_IS_WT still defined"
    assert any(n.startswith("null_emb") for n in names) and \
        any(n.startswith("null_proj") for n in names)


# --- what makes a token ----------------------------------------------------

def _test_only_nucleotides_make_tokens():
    m = net()
    xs = put(reference_batch(1), ("protein", "gA"), 5, channel=3)   # residue, no base
    rep = {r["locus"]: r for r in m.variant_report(xs)}
    assert int(rep["gA"]["n_variants"]) == 0, "a protein change created a token"
    xs = put(xs, ("dna", "gA"), 48, channel=1)
    rep = {r["locus"]: r for r in m.variant_report(xs)}
    assert int(rep["gA"]["n_variants"]) == 1


def _test_residue_lands_on_its_codon_gap_aware():
    """Column 48 of gA: 47 reference bases before it (column 45 is a gap), 17 of
    them after the start codon's first base -> codon 5, phase 2. Raw columns would
    say (48 - 30) // 3 = codon 6."""
    m = net()
    mp, _ = maps(m, "gA")
    off = [b for b in m._plan[m.loci.index("gA")]["nt"] if b["modality"] == "dna"][0]["offset"]
    assert mp["codon"][off + 48] == 5 and mp["phase"][off + 48] == 2, \
        (mp["codon"][off + 48], mp["phase"][off + 48])
    assert mp["codon"][off + 45] == -1, "a reference gap column got a codon"

    xs = put(reference_batch(1), ("dna", "gA"), 48, channel=2)       # the base
    xs = put(xs, ("protein", "gA"), 5, channel=7)                    # its residue
    xs = put(xs, ("protein", "gA"), 6, channel=9)                    # the wrong codon
    xs = put(xs, ("biophysical", "gA"), 5, channel=0, value=0.5)
    f, coord, valid = token(m, xs, "gA", 1)
    assert valid
    assert int(f[SLOTS["dna"][0]:SLOTS["dna"][1]].argmax()) == 2
    aa = f[SLOTS["protein"][0]:SLOTS["protein"][1]]
    assert int(aa.argmax()) == 7 and float(aa.sum()) == 1.0, \
        "the token did not carry exactly its own codon's residue"
    assert abs(float(f[SLOTS["biophysical"][0]]) - 0.5) < 1e-6
    assert int(f[F_PHASE:F_PHASE + 3].argmax()) == 2
    assert abs(coord - 17 / 3) < 1e-5, coord


def _test_synonymous_change_restates_the_reference():
    """A delta-encoded aa block is zero where the residue did not change, which
    is indistinguishable from "no residue here". The reference fills it instead:
    the token says which residue the codon still codes for, and the property
    delta -- the only channel that separates it from a substitution -- is zero."""
    m = net()
    xs = put(reference_batch(1), ("dna", "gA"), 48, channel=2)       # protein stays zero
    f, _, valid = token(m, xs, "gA", 1)
    assert valid
    ser = _RES_ORDER.index("S")                       # REF_AA's codon 5
    aa = f[SLOTS["protein"][0]:SLOTS["protein"][1]]
    assert int(aa.argmax()) == ser and float(aa.sum()) == 1.0, \
        "a synonymous token did not restate the reference residue"
    props = torch.as_tensor(RESIDUE_PROPERTIES[ser])
    assert torch.allclose(f[SLOTS["biophysical"][0]:SLOTS["biophysical"][1]], props,
                          atol=1e-6), "the isolate's properties were left empty"
    assert float(f[D_BIO[0]:D_BIO[1]].abs().sum()) == 0, \
        "a synonymous change reported a property delta"
    assert float(f[F_PHASE:F_PHASE + 3].sum()) == 1, "a coding base lost its phase"


def _test_substitution_carries_reference_and_delta():
    """A real substitution: both residues are on the token and the delta is what
    the substitution did to each property."""
    m = net()
    ser, asn = _RES_ORDER.index("S"), _RES_ORDER.index("N")
    xs = put(reference_batch(1), ("dna", "gA"), 48, channel=2)       # the base
    xs = put(xs, ("protein", "gA"), 5, channel=asn)                  # S -> N
    alt = torch.as_tensor(RESIDUE_PROPERTIES[asn])
    ref = torch.as_tensor(RESIDUE_PROPERTIES[ser])
    for c in range(3):                                               # its properties
        xs = put(xs, ("biophysical", "gA"), 5, channel=c, value=float(alt[c]))
    f, _, valid = token(m, xs, "gA", 1)
    assert valid
    assert int(f[SLOTS["protein"][0]:SLOTS["protein"][1]].argmax()) == asn
    assert torch.allclose(f[D_BIO[0]:D_BIO[1]], alt - ref, atol=1e-5), \
        "the property delta is not alt - ref"
    assert D_BIO[1] == F_IS_NULL, "a reference-value slot is still in the layout"


def _test_delta_without_a_biophysical_block():
    """The properties are a table lookup from the residue, so a dna+protein cell
    still gets the reference and the delta."""
    keys = [k for k in KEYS if k[0] != "biophysical"]
    specs = [sp for k, sp in zip(KEYS, SPECS) if k[0] != "biophysical"]
    frames = [fr for k, fr in zip(KEYS, FRAMES) if k[0] != "biophysical"]
    torch.manual_seed(0)
    m = LocusFusionV2Net(keys, specs, block_frames=frames); m.eval()
    idx = {k: i for i, k in enumerate(keys)}
    xs = [torch.zeros(1, c, length) for c, length in specs]
    ser, asn = _RES_ORDER.index("S"), _RES_ORDER.index("N")
    xs[idx[("dna", "gA")]][0, 2, 48] = 1.0
    xs[idx[("protein", "gA")]][0, asn, 5] = 1.0
    feat, _, valid, _ = m._locus_tokens(xs)
    li = m.loci.index("gA")
    f = feat[0, li, 1]
    assert bool(valid[0, li, 1])
    ref = torch.as_tensor(RESIDUE_PROPERTIES[ser])
    alt = torch.as_tensor(RESIDUE_PROPERTIES[asn])
    assert torch.allclose(f[D_BIO[0]:D_BIO[1]], alt - ref, atol=1e-5)


def _test_a_deleted_base_does_not_restate_the_reference():
    """A gap token is a base the isolate LOST. The aa block is delta-zero at that
    codon (nothing SUBSTITUTED it), so the reference fill must not claim the codon
    still reads as H37Rv -- the deletion would vanish from the residue channels."""
    m = net()
    gap = SLOTS["dna"][0] + 4                       # A C T G -
    xs = put(reference_batch(1), ("dna", "gA"), 48, channel=4)
    f, _, valid = token(m, xs, "gA", 1)
    ser = _RES_ORDER.index("S")                     # REF_AA's codon 5
    assert valid and float(f[gap]) == 1.0, "a deleted base made no token"
    assert float(f[SLOTS["protein"][0]:SLOTS["protein"][1]].abs().sum()) == 0, \
        "a deletion was reported as an unchanged residue"
    assert float(f[SLOTS["biophysical"][0]:SLOTS["biophysical"][1]].abs().sum()) == 0
    # ...and the delta says the reference residue is GONE, as it does for a stop
    assert torch.allclose(f[D_BIO[0]:D_BIO[1]],
                          -torch.as_tensor(RESIDUE_PROPERTIES[ser]), atol=1e-6), \
        "a deletion did not report the loss of the reference residue"


def _test_flank_and_promoter_carry_no_codon():
    m = net()
    xs = put(reference_batch(1), ("dna", "gA"), 10)                  # 5' flank
    xs = put(xs, ("regulatory", "gA"), 3)                            # promoter
    xs = put(xs, ("protein", "gA"), 0, channel=4)
    feat, _, valid, _ = m._locus_tokens(xs)
    li = m.loci.index("gA")
    assert int(valid[0, li, 1:].sum()) == 2
    no_codon = torch.cat([feat[0, li, 1:3, SLOTS["protein"][0]:SLOTS["biophysical"][1]],
                          feat[0, li, 1:3, F_PHASE:F_PHASE + 3]], dim=-1)
    assert float(no_codon.abs().sum()) == 0, "a non-coding base carried a residue or phase"
    # the promoter base is in the regulatory slot, the flank base in the dna slot
    assert float(feat[0, li, 1, SLOTS["regulatory"][0]:SLOTS["regulatory"][1]].sum()) == 1
    assert float(feat[0, li, 2, SLOTS["dna"][0]:SLOTS["dna"][1]].sum()) == 1


# --- placement -------------------------------------------------------------

def _test_promoter_placed_by_genome_position_plus_strand():
    m = net()
    mp, p = maps(m, "gA")
    # start codon: genome 1000 + 30 bases = 1030; promoter column 0 is genome 980
    assert abs(mp["coord"][0] - (980 - 1030) / 3) < 1e-5, mp["coord"][0]
    assert abs(mp["coord"][19] - (999 - 1030) / 3) < 1e-5


def _test_promoter_placed_by_genome_position_minus_strand():
    m = net()
    mp, p = maps(m, "gB")
    # start codon at record column 15 -> genome 5089 - 15 = 5074. Upstream of a
    # minus-strand gene is higher genome positions: promoter column 0 = 5129.
    assert abs(mp["coord"][0] - (5074 - 5129) / 3) < 1e-5, mp["coord"][0]
    assert abs(mp["coord"][29] - (5074 - 5100) / 3) < 1e-5
    rec = [b for b in p["nt"] if b["modality"] == "dna"][0]
    assert mp["codon"][rec["offset"] + 15] == 0 and mp["codon"][rec["offset"] + 18] == 1


def _test_kept_axis_is_monotonic():
    m = net()
    for locus in m.loci:
        mp, _ = maps(m, locus)
        d = np.diff(mp["coord"][mp["keep"]])
        assert (d >= -1e-6).all(), f"{locus}: coordinate steps backwards along the kept axis"


def _test_shared_base_is_one_token():
    m = net()
    mp, p = maps(m, "gA")
    assert int((~mp["keep"]).sum()) == 20, "expected the 20 shared promoter columns dropped"
    # the same base, present in both blocks (promoter col 25 == record col 5)
    xs = put(reference_batch(1), ("regulatory", "gA"), 25)
    xs = put(xs, ("dna", "gA"), 5)
    rep = {r["locus"]: r for r in m.variant_report(xs)}
    assert int(rep["gA"]["n_variants"]) == 1, "a shared base produced two tokens"
    assert int(rep["gA"]["region"][0, 0]) == 1, "the kept copy is not the record's"


def _test_base_shared_across_loci_is_a_token_in_each():
    """A deliberate choice, pinned so it cannot change by accident: the dedupe
    stops at the locus boundary. Locus gC's promoter window covers genome
    1105..1114, which is inside gA's record — the fabG1 x inhA-window shape."""
    keys = [("dna", "gA"), ("dna", "gC"), ("regulatory", "gC")]
    specs = [(5, 120), (5, 60), (5, 10)]
    frames = [FRAMES[0],
              {"ref_gap_cols": [], "genome_start": 1200, "genome_end": 1260, "strand": "+",
               "cds_start": 20, "cds_end": 50, "has_cds": True},
              {"ref_gap_cols": [], "genome_start": 1105, "genome_end": 1115, "strand": "+"}]
    torch.manual_seed(0)
    m = LocusFusionV2Net(keys, specs, block_frames=frames)
    assert m.frames["gC"]["duplicate_columns"] == 0, "cross-locus columns were dropped"
    xs = [torch.zeros(1, c, length) for c, length in specs]
    xs[0][0, 0, 106] = 1.0          # gA record column 106 = genome 1105 (one gap before it)
    xs[2][0, 0, 0] = 1.0            # gC promoter column 0 = genome 1105, the same base
    rep = {r["locus"]: r for r in m.variant_report(xs)}
    assert int(rep["gA"]["n_variants"]) == 1 and int(rep["gC"]["n_variants"]) == 1, \
        {k: int(v["n_variants"]) for k, v in rep.items()}


# --- the budget ------------------------------------------------------------

def _test_cap_is_per_locus_promoter_first():
    m = net()
    xs = reference_batch(1)
    for c in range(0, 10):                       # 10 promoter variants (kept columns)
        xs = put(xs, ("regulatory", "gA"), c)
    for c in range(31, 91, 6):                   # 10 record variants
        xs = put(xs, ("dna", "gA"), c)
    rep = {r["locus"]: r for r in m.variant_report(xs)}
    r = rep["gA"]
    assert m.tokens_per_locus == 1 + LOCUSFUSION_V2_DEFAULTS["max_variants"]
    assert int(r["n_variants"]) == 20 and int(r["valid"].sum()) == 16
    assert int((r["region"][0][r["valid"][0]] == 0).sum()) == 10, \
        "promoter variants were truncated before record variants"


def _test_all_differing_locus_reads_full_on_null():
    m = net()
    xs = reference_batch(1)
    xs[B[("dna", "gB")]][:, 4, :] = 1.0          # the whole record is a gap
    _, _, valid, stats = m._locus_tokens(xs)
    li = m.loci.index("gB")
    assert int(valid[0, li, 1:].sum()) == 16
    assert float(stats[0, li, 1]) > 0.5, "an unassembled record reads as nearly unchanged"


# --- rRNA ------------------------------------------------------------------

def _test_rrna_carries_no_codons():
    m = net()
    mp, _ = maps(m, "rrX")
    assert (mp["codon"] == -1).all() and (mp["phase"] == -1).all()
    assert m.frames["rrX"]["has_cds"] is False
    xs = put(reference_batch(1), ("dna", "rrX"), 12)
    xs = put(xs, ("protein", "rrX"), 0, channel=5)     # the stub translation changed
    f, coord, valid = token(m, xs, "rrX", 1)
    assert valid and abs(coord - 2 / 3) < 1e-5         # origin at the rRNA gene start
    assert float(f[SLOTS["protein"][0]:].abs().sum()) == 0, \
        "an rRNA token carried a residue, biophysics or phase"
    assert float(f[D_BIO[0]:D_BIO[1]].abs().sum()) == 0, \
        "an rRNA token carried a property delta"


# --- the reference codon grid ---------------------------------------------

def _test_reference_frame_translation():
    from datasets import biochem as bc
    ref = "ATG" + "TCG" + "AAA" + "GGC" + "TGA"                    # M S K G *
    t = lambda iso, r=ref: bc.translate_reference_frame(iso, r)
    assert t(ref) == "MSKG*"
    assert t("ATGACGAAAGGCTGA") == "MTKG*", "substitution"
    assert t("ATGTCGTAAGGCTGA") == "MS*G*", "a stop gain is a residue"
    # a 1-bp deletion: the degapped read is a frameshift ("MSKA"); on the grid
    # only the codon that lost the base changes
    assert bc.translate_seq("ATGTC-AAAGGCTGA") == "MSKA"
    assert t("ATGTC-AAAGGCTGA") == "M-KG*", "a frameshift changed codons it does not touch"
    assert t("ATG---AAAGGCTGA") == "M-KG*", "an in-frame deletion shifted the numbering"
    # bases at a column where the REFERENCE has a gap belong to no reference codon
    refg = "ATGTCG--AAAGGCTGA"
    assert t("ATGTCGCCAAAGGCTGA", refg) == "MSKG*", "an insertion shifted the numbering"
    assert t("ATGTCNAAAGGCTGA") == "MXKG*", "an ambiguous codon is not X"
    assert bc.translate_reference_frame("CCATGTCGTGACC", "CCATGTCGTGACC", (2, 11)) == "MS*"
    hot = bc.one_hot_aa("MS*", pad_to=4, with_stop=True)
    assert hot.shape == (21, 4) and hot[20, 2] == 1 and hot[:, 3].sum() == 0
    assert bc.one_hot_aa("MS*", pad_to=4).shape == (20, 4), "the isolate-frame default changed"


def _test_stop_gain_carries_a_residue():
    """A nonsense mutation and a synonymous one used to look the same on the
    token: base filled, residue slot empty. STOP is now channel 20."""
    m = net()
    xs = put(reference_batch(2), ("dna", "gA"), 48, channel=2)
    xs = put(xs, ("protein", "gA"), 5, channel=20, row=0)            # isolate 0: stop gained
    feat, _, valid, _ = m._locus_tokens(xs)
    li = m.loci.index("gA")
    lo, hi = SLOTS["protein"]
    assert valid[:, li, 1].all()
    assert float(feat[0, li, 1, lo + 20]) == 1.0, "the stop codon is not on its token"
    assert int(feat[1, li, 1, lo:hi].argmax()) == _RES_ORDER.index("S"), \
        "the synonymous twin did not restate the reference residue"
    assert float(feat[0, li, 1, D_BIO[0]:D_BIO[1]].abs().sum()) > 0, \
        "a stop gain reported no property change"
    assert float(feat[1, li, 1, D_BIO[0]:D_BIO[1]].abs().sum()) == 0
    assert not torch.allclose(m(xs)[0], m(xs)[1]), "a stop gain and a synonymous change scored the same"


def _test_isolate_frame_protein_rejected():
    specs = list(SPECS); specs[B[("protein", "gA")]] = (20, 20)        # 20 channels: isolate frame
    try:
        LocusFusionV2Net(KEYS, specs, block_frames=FRAMES)
    except ValueError as e:
        assert "protein_frame='reference'" in str(e), e
    else:
        raise AssertionError("a 20-channel protein block was accepted")
    specs = list(SPECS); specs[B[("protein", "gA")]] = (21, 19)        # wrong width for 20 codons
    try:
        LocusFusionV2Net(KEYS, specs, block_frames=FRAMES)
    except ValueError as e:
        assert "codon grid" in str(e), e
    else:
        raise AssertionError("a protein block narrower than the reference CDS was accepted")
    # rRNA has no codons, so its (fallback) protein width is not checked
    specs = list(SPECS); specs[B[("protein", "rrX")]] = (21, 7)
    LocusFusionV2Net(KEYS, specs, block_frames=FRAMES)


# --- plumbing --------------------------------------------------------------

def _test_missing_frames_warn_and_run():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = LocusFusionV2Net(KEYS, SPECS)
    assert any("no genome span" in str(x.message) for x in w), "no fallback warning"
    assert m(put(reference_batch(2), ("dna", "gA"), 48)).shape == (2, 1)


def _test_dense_input_warns():
    m = net()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m([torch.ones(2, c, length) for c, length in SPECS])
    assert any("--delta" in str(x.message) for x in w), "dense input did not warn"


def _test_bad_construction_rejected():
    for kwargs, what in [
        ({"block_frames": FRAMES[:-1]}, "a frame list of the wrong length"),
    ]:
        try:
            LocusFusionV2Net(KEYS, SPECS, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{what} was accepted")
    try:
        LocusFusionV2Net([("protein", "gA")], [(21, 20)])
    except ValueError as e:
        assert "NUCLEOTIDE" in str(e)
    else:
        raise AssertionError("a protein-only locus was accepted")
    assert "uncovered_frac" not in locusfusion_defaults("locusfusion_v2")
    try:
        LocusFusionV2Net(KEYS, SPECS, block_frames=FRAMES, uncovered_frac=0.5)
    except TypeError:
        pass
    else:
        raise AssertionError("v1's uncovered_frac knob was accepted")


def _test_backward_reaches_every_parameter():
    torch.manual_seed(0)
    m = LocusFusionV2Net(KEYS, SPECS, block_frames=FRAMES, drug_names=["A", "B"],
                         carry_variants=2)
    xs = put(put(reference_batch(4), ("dna", "gA"), 48, row=0), ("regulatory", "gB"), 3, row=1)
    m(xs).sum().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None]
    assert not dead, f"parameters with no gradient: {dead}"


def _test_config_round_trips():
    class _B:
        def __init__(self, key, spec, meta):
            self.name = f"{key[0]}:{key[1]}"
            self._spec, self.meta, self.channel_names = spec, meta, []

        def spec(self):
            return self._spec

    blocks = [_B(k, s, f) for k, s, f in zip(KEYS, SPECS, FRAMES)]
    a = LocusFusionV2Net.from_blocks(blocks, drug_names=["A", "B"])
    cfg = model_config(arch="locusfusion_v2", blocks=blocks, encoder_types=["cnn"] * len(KEYS),
                       drug_names=["A", "B"], out_bias=None,
                       head={"hidden": 256, "dropout": 0.0, "per_drug_hidden": 0})
    b = build_model_from_config({"model": cfg})
    b.load_state_dict(a.state_dict(), strict=True)
    for li in range(len(a.loci)):
        for k in ("coord", "codon", "phase", "keep"):
            assert torch.equal(getattr(a, f"{k}_map_{li}"), getattr(b, f"{k}_map_{li}")), (li, k)
    a.eval(); b.eval()
    xs = put(reference_batch(3), ("dna", "gB"), 20, row=2)
    assert torch.allclose(a(xs), b(xs), atol=1e-6), "rebuilt model gives different logits"


def _test_feature_layout():
    spans = sorted(SLOTS.values())
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 <= b0, f"feature slots overlap: {spans}"
    assert spans[-1][1] <= D_BIO[0]
    assert D_BIO[1] <= F_IS_NULL < F_PHASE and F_PHASE + 3 == C_TOK == 41
    assert SLOTS["protein"][1] - SLOTS["protein"][0] == 21, \
        "protein slot is not 20 residues + STOP"
    assert SLOTS["dna"][1] - SLOTS["dna"][0] == 5, "dna slot is not A C T G -"
    assert D_BIO[1] - D_BIO[0] == SLOTS["biophysical"][1] - SLOTS["biophysical"][0]
    assert RESIDUE_PROPERTIES.shape == (21, 3)
    assert not RESIDUE_PROPERTIES[_RES_ORDER.index("*")].any(), "STOP has properties"


# --- the structure channels --------------------------------------------------
# gA's 20 codons laid out along x, 3.8 A apart, except codon 12, which folds back
# to sit 5 A above codon 5. Codon 19 is the stop: no coordinate. gB gets a fold
# too; rrX (rRNA) gets none.
def _fold(n, fold_back=True):
    ca = [[3.8 * k, 0.0, 0.0] for k in range(n)]
    if fold_back and n > 12:
        ca[12] = [3.8 * 5, 5.0, 0.0]
    ca[n - 1] = None
    return {"ca": ca, "model_id": "AF-TEST-F1", "afdb_version": 0}


S_KEYS = KEYS + [("structure", "gA"), ("structure", "gB")]
S_SPECS = SPECS + [(0, 20), (0, 20)]
S_FRAMES = FRAMES + [_fold(20), _fold(20, fold_back=False)]
S = {key: i for i, key in enumerate(S_KEYS)}


def s_net(**kw):
    torch.manual_seed(0)
    m = LocusFusionV2Net(S_KEYS, S_SPECS, block_frames=S_FRAMES, **kw)
    m.eval()
    return m


def s_batch(b=1):
    return [torch.zeros(b, c, length) for c, length in S_SPECS]


def s_put(xs, key, column, channel=0):
    xs = [x.clone() for x in xs]
    xs[S[key]][:, channel, column] = 1.0
    return xs


def _struct(m, xs, locus, slot):
    """(proximity row, "is a real entry" row) of one variant slot.

    The channels hold exp(-d / STRUCT_LENGTH), so a real entry is > 0 and an
    absent one is exactly 0; the distance in A is -ln(proximity) * STRUCT_LENGTH.
    """
    feat, _, _, _ = m._locus_tokens(xs)
    f = feat[0, m.loci.index(locus), slot]
    V = m.max_variants
    prox = f[m.f_struct:m.f_struct + V]
    return prox, (prox > 0).float()


def _angstrom(prox):
    """The proximity channels read back as distances in A (inf where absent)."""
    return -torch.log(prox.clamp(min=1e-12)) * STRUCT_LENGTH


def _test_structure_width_and_checkpoint_compat():
    plain, s = net(), s_net()
    assert plain.c_tok == C_TOK == 41 and plain.tok_proj.in_features == 41, \
        "a model without structure changed width"
    assert not plain.use_structure and s.use_structure
    assert s.c_tok == 41 + s.max_variants == s.tok_proj.in_features, \
        "structure is not exactly max_variants channels"


def _test_structure_leaves_the_other_channels_alone():
    xs = s_put(s_put(s_batch(), ("dna", "gA"), 48, 2), ("regulatory", "gB"), 3)
    a = net()._locus_tokens([x for x, k in zip(xs, S_KEYS) if k[0] != "structure"])[0]
    b = s_net()._locus_tokens(xs)[0]
    assert torch.equal(a, b[..., :C_TOK]), "adding structure changed a base channel"


def _test_structure_row_is_the_distance_matrix():
    """Promoter column 0 (slot 1), codon 5 (column 48, slot 2), codon 12 (column
    67, slot 3). Row index j is variant slot j+1."""
    m = s_net()
    xs = s_put(s_put(s_put(s_batch(), ("regulatory", "gA"), 0), ("dna", "gA"), 48),
               ("dna", "gA"), 67)
    rep = {r["locus"]: r for r in m.variant_report(xs)}["gA"]
    assert rep["codon"][0, :3].tolist() == [-1, 5, 12], rep["codon"][0, :3]
    prox, mk = _struct(m, xs, "gA", 2)                       # codon 5
    d = _angstrom(prox)
    assert mk[:3].tolist() == [0, 1, 1] and not mk[3:].any(), mk
    assert abs(float(prox[1]) - 1.0) < 1e-5, "a token is not maximally close to itself"
    assert abs(float(d[2]) - 5.0) < 1e-3, f"codon 5 -> 12 should be 5 A, got {float(d[2])}"
    prox12, _ = _struct(m, xs, "gA", 3)
    assert abs(float(_angstrom(prox12)[1]) - 5.0) < 1e-3, "the matrix is not symmetric"
    dp, mp = _struct(m, xs, "gA", 1)                         # the promoter token
    assert not dp.any() and not mp.any(), "a promoter token was placed"
    assert torch.allclose(rep["struct_dist"][0, 1, 2], torch.tensor(5.0), atol=1e-4)
    # slot-aligned: stacking the tokens' distance channels IS the matrix
    V = m.max_variants
    feat = m._locus_tokens(xs)[0][0, m.loci.index("gA"), 1:1 + V]
    stacked = feat[:, m.f_struct:m.f_struct + V]
    k = rep["struct_dist"].shape[-1]
    real = stacked[:k, :k] > 0
    assert torch.allclose(_angstrom(stacked[:k, :k])[real],
                          rep["struct_dist"][0][real], atol=1e-3)
    assert not stacked[k:].any() and not stacked[:, k:].any()


def _test_structure_same_codon_is_zero_not_masked():
    m = s_net()
    xs = s_put(s_put(s_batch(), ("dna", "gA"), 47), ("dna", "gA"), 48)   # codon 5, phases 1, 2
    prox, mk = _struct(m, xs, "gA", 1)
    assert mk[:2].tolist() == [1, 1], "two tokens in one codon are not both placed"
    assert abs(float(prox[1]) - 1.0) < 1e-5, "0 A did not read as maximum proximity"


def _test_structure_masks_stop_rrna_and_null():
    m = s_net()
    xs = s_put(s_put(s_batch(), ("dna", "gA"), 88), ("dna", "gA"), 48)   # stop, codon 5
    xs = s_put(xs, ("dna", "rrX"), 20)
    rep = {r["locus"]: r for r in m.variant_report(xs)}
    assert rep["gA"]["codon"][0, :2].tolist() == [5, 19]
    d, mk = _struct(m, xs, "gA", 2)
    assert not mk.any() and not d.any(), "the stop codon was placed"
    _, mk5 = _struct(m, xs, "gA", 1)
    assert mk5.tolist()[:2] == [1, 0], "a located token claims a distance to the stop"
    feat, _, _, _ = m._locus_tokens(xs)
    assert not feat[:, :, :, C_TOK:][:, m.loci.index("rrX")].any(), "an rRNA token was placed"
    assert not feat[:, :, 0, C_TOK:].any(), "[NULL] was placed"


def _test_structure_bad_construction_rejected():
    cases = [
        (S_KEYS + [("structure", "rrX")], S_SPECS + [(0, 5)], S_FRAMES + [_fold(5)],
         "a structure block on an rRNA locus"),
        (S_KEYS[:-1] + [("structure", "gB")], S_SPECS[:-1] + [(0, 19)],
         S_FRAMES[:-1] + [_fold(19)], "a structure block off the codon grid"),
        (S_KEYS, S_SPECS, S_FRAMES[:-1] + [{}], "a structure block with no coordinates"),
    ]
    for keys, specs, frames, what in cases:
        try:
            LocusFusionV2Net(keys, specs, block_frames=frames)
        except ValueError:
            continue
        raise AssertionError(f"{what} was accepted")


def _test_structure_backward_and_round_trip():
    import json

    class _B:
        def __init__(self, key, spec, meta):
            self.name = f"{key[0]}:{key[1]}"
            self._spec, self.meta, self.channel_names = spec, meta, []

        def spec(self):
            return self._spec

    blocks = [_B(k, s, f) for k, s, f in zip(S_KEYS, S_SPECS, S_FRAMES)]
    torch.manual_seed(0)
    a = LocusFusionV2Net.from_blocks(blocks, drug_names=["A", "B"], carry_variants=2)
    xs = s_put(s_put(s_batch(3), ("dna", "gA"), 48), ("dna", "gA"), 67)
    a(xs).sum().backward()
    dead = [n for n, p in a.named_parameters() if p.grad is None]
    assert not dead, f"parameters with no gradient: {dead}"
    # through JSON, as config.json is written: a missing coordinate is None there
    cfg = json.loads(json.dumps(model_config(
        arch="locusfusion_v2", blocks=blocks, encoder_types=["cnn"] * len(S_KEYS),
        drug_names=["A", "B"], out_bias=None,
        head={"hidden": 256, "dropout": 0.0, "per_drug_hidden": 0},
        locusfusion={"carry_variants": 2}), allow_nan=False))
    b = build_model_from_config({"model": cfg})
    b.load_state_dict(a.state_dict(), strict=True)
    assert b.use_structure and b.c_tok == a.c_tok
    for li in range(len(a.loci)):
        for k in ("ca", "has"):
            assert torch.equal(getattr(a, f"struct_{k}_{li}"), getattr(b, f"struct_{k}_{li}"))
    a.eval(); b.eval()
    assert torch.allclose(a(xs), b(xs), atol=1e-6), "rebuilt model gives different logits"


def _test_structure_length_scale():
    """struct_length sets the decay: codon 5 -> 12 is 5 A apart, so it reads
    exp(-5/10) by default and exp(-5/30) at 30 A. A config records it only when
    it differs from the default, so every older checkpoint rebuilds at 10 A."""
    import json

    xs = s_put(s_put(s_batch(), ("dna", "gA"), 48), ("dna", "gA"), 67)
    li = None
    for L in (STRUCT_LENGTH, 30.0):
        m = s_net(struct_length=L)
        li = m.loci.index("gA")
        f = m._locus_tokens(xs)[0][0, li, 1]
        got = float(f[m.f_struct + 1])
        assert abs(got - float(np.exp(-5.0 / L))) < 1e-5, (L, got)
        assert abs(float(f[m.f_struct]) - 1.0) < 1e-6, "own slot is not 1"
    assert LOCUSFUSION_V2_DEFAULTS["struct_length"] == STRUCT_LENGTH

    class _B:
        def __init__(self, key, spec, meta):
            self.name = f"{key[0]}:{key[1]}"
            self._spec, self.meta, self.channel_names = spec, meta, []

        def spec(self):
            return self._spec

    blocks = [_B(k, sp, fr) for k, sp, fr in zip(S_KEYS, S_SPECS, S_FRAMES)]
    head = {"hidden": 256, "dropout": 0.0, "per_drug_hidden": 0}
    for knobs, want in (({}, STRUCT_LENGTH), ({"struct_length": 30.0}, 30.0)):
        cfg = json.loads(json.dumps(model_config(
            arch="locusfusion_v2", blocks=blocks, encoder_types=["cnn"] * len(S_KEYS),
            drug_names=["A"], out_bias=None, head=head, locusfusion=knobs)))
        b = build_model_from_config({"model": cfg})
        assert b.struct_length == want, (knobs, b.struct_length)
    try:
        s_net(struct_length=0)
    except ValueError:
        pass
    else:
        raise AssertionError("struct_length=0 was accepted")


# --- the real reference rows, when mounted ---------------------------------

# --- structure knobs: heavy atoms, the contact gate, the invalid value ---------
# gA as above, plus one side-chain atom per residue 1 A up in z -- except codons
# 5 and 12 (C-alpha 5 A apart), whose side chains point at each other and end
# 2 A apart. The pair (5, 12) is listed as failing PAE.
def _heavy_frame(frame, bad=()):
    heavy = []
    for k, xyz in enumerate(frame["ca"]):
        if xyz is None:
            heavy.append(None)
        elif k == 5:
            heavy.append([xyz, [xyz[0], 1.5, 0.0]])
        elif k == 12:
            heavy.append([xyz, [xyz[0], 3.5, 0.0]])
        else:
            heavy.append([xyz, [xyz[0], xyz[1], 1.0]])
    return {**frame, "heavy": heavy, "pae_bad": [list(b) for b in bad]}


H_FRAMES = FRAMES + [_heavy_frame(_fold(20), bad=[(5, 12)]), _heavy_frame(_fold(20, False))]


def h_net(**kw):
    torch.manual_seed(0)
    m = LocusFusionV2Net(S_KEYS, S_SPECS, block_frames=H_FRAMES, **kw)
    m.eval()
    return m


def _rows(m, xs, locus="gA"):
    feat = m._locus_tokens(xs)[0][0, m.loci.index(locus), 1:1 + m.max_variants]
    return feat[:, m.f_struct:m.f_struct + m.max_variants]


def _test_structure_invalid_value():
    """-1 for every entry with no geometry -- the promoter token, the stop codon,
    padding -- while the same residue still reads 1 and a real pair exp(-d/L)."""
    xs = s_put(s_put(s_put(s_batch(), ("regulatory", "gA"), 0), ("dna", "gA"), 48),
               ("dna", "gA"), 67)
    base, neg = _rows(s_net(), xs), _rows(s_net(struct_invalid=-1.0), xs)
    real = base > 0
    assert torch.equal(base[real], neg[real]), "a real entry changed"
    assert (neg[:3][~real[:3]] == -1).all(), neg[:3]
    assert torch.allclose(neg[1, 1], torch.tensor(1.0)), "same residue is not 1"
    assert abs(float(neg[1, 2]) - math.exp(-0.5)) < 1e-5
    assert (neg[1, 3:] == -1).all(), "padding partners of a real token are not -1"
    assert not neg[3:].any(), "a padding token's own row is not zeroed"


def _test_structure_heavy_atoms():
    """Closest heavy atoms: codon 5 -> 12 is 2 A (side chains), not 5 A (C-alpha),
    and never more than the C-alpha distance."""
    xs = s_put(s_put(s_put(s_batch(), ("dna", "gA"), 48), ("dna", "gA"), 67),
               ("dna", "gA"), 51)
    ca, hv = h_net(), h_net(struct_dist="heavy")
    rc = {r["locus"]: r for r in ca.variant_report(xs)}["gA"]
    rh = {r["locus"]: r for r in hv.variant_report(xs)}["gA"]
    assert rc["codon"][0, :3].tolist() == [5, 6, 12], rc["codon"][0, :3]
    dc, dh = rc["struct_dist"][0], rh["struct_dist"][0]
    assert abs(float(dc[0, 2]) - 5.0) < 1e-4 and abs(float(dh[0, 2]) - 2.0) < 1e-4, (dc, dh)
    assert abs(float(dh[0, 1]) - 3.8) < 1e-4, "neighbours' heavy distance is not 3.8 A"
    assert (dh <= dc + 1e-5).all() and float(dh[0, 0]) == 0.0
    try:
        LocusFusionV2Net(S_KEYS, S_SPECS, block_frames=S_FRAMES, struct_dist="heavy")
    except ValueError:
        pass
    else:
        raise AssertionError("struct_dist='heavy' built without heavy atoms on the meta")


def _test_structure_contact_gate():
    """The gate drops a pair failing PAE (5, 12) and a pair beyond CONTACT_REACH
    (codon 0 -> 6, 22.8 A C-alpha) but keeps a near, confident one (5 -> 6)."""
    from datasets.structure import CONTACT_REACH
    assert abs(CONTACT_REACH - 19.82) < 1e-6
    xs = s_put(s_put(s_put(s_put(s_batch(), ("dna", "gA"), 30), ("dna", "gA"), 48),
                     ("dna", "gA"), 51), ("dna", "gA"), 67)
    m = h_net(struct_gate="contact", struct_invalid=-1.0)
    rep = {r["locus"]: r for r in m.variant_report(xs)}["gA"]
    assert rep["codon"][0, :4].tolist() == [0, 5, 6, 12], rep["codon"][0, :4]
    mk = rep["struct_mask"][0]
    assert not mk[1, 3] and not mk[3, 1], "the PAE-failing pair passed"
    assert not mk[0, 2], "a pair beyond CONTACT_REACH passed"
    assert mk[1, 2] and mk[0, 0] and mk[0, 1], "a near confident pair was gated"
    rows = _rows(m, xs)
    assert float(rows[1, 3]) == -1.0 and float(rows[1, 2]) > 0
    # ungated, every located pair is real
    assert {r["locus"]: r for r in h_net().variant_report(xs)}["gA"]["struct_mask"][0, :4, :4].all()


def _test_structure_knobs_round_trip():
    import json

    class _B:
        def __init__(self, key, spec, meta):
            self.name = f"{key[0]}:{key[1]}"
            self._spec, self.meta, self.channel_names = spec, meta, []

        def spec(self):
            return self._spec

    blocks = [_B(k, s, f) for k, s, f in zip(S_KEYS, S_SPECS, H_FRAMES)]
    knobs = {"struct_dist": "heavy", "struct_gate": "contact", "struct_invalid": -1.0,
             "struct_length": 30.0}
    torch.manual_seed(0)
    a = LocusFusionV2Net.from_blocks(blocks, drug_names=["A", "B"], **knobs)
    cfg = json.loads(json.dumps(model_config(
        arch="locusfusion_v2", blocks=blocks, encoder_types=["cnn"] * len(S_KEYS),
        drug_names=["A", "B"], out_bias=None,
        head={"hidden": 256, "dropout": 0.0, "per_drug_hidden": 0},
        locusfusion=knobs), allow_nan=False))
    b = build_model_from_config({"model": cfg})
    b.load_state_dict(a.state_dict(), strict=True)
    assert (b.struct_dist, b.struct_gate, b.struct_invalid) == ("heavy", "contact", -1.0)
    xs = s_put(s_put(s_batch(3), ("dna", "gA"), 48), ("dna", "gA"), 67)
    a.eval(); b.eval()
    assert torch.allclose(a(xs), b(xs), atol=1e-6), "rebuilt model gives different logits"


def _test_real_frames_all_loci():
    try:
        from bigtb_ref import REAL_GENOTYPE_DIR as GD, REAL_REGULATORY_DIR as RD, tb
        from datasets import cds
        from datasets.sequences import reference_row
        if not Path(GD).exists() or not Path(cds.GENBANK).exists():
            raise FileNotFoundError
    except Exception:  # noqa: BLE001
        print("      (real data not mounted — skipped)")
        return
    loci = sorted({g for gs in tb.DRUG_TO_LOCI.values() for g in gs})
    keys, specs, frames = [], [], []
    for g in loci:
        r = reference_row(GD, g)
        keys.append(("dna", g)); specs.append((5, len(r)))
        frames.append(cds.block_frame(GD, g, len(r), translated=True))
        p = reference_row(RD, g)
        if p:
            keys.append(("regulatory", g)); specs.append((5, len(p)))
            frames.append(cds.block_frame(RD, g, len(p)))
    m = LocusFusionV2Net(keys, specs, block_frames=frames)
    expect = {"katG": 741, "inhA": 270, "rpoB": 1173, "gyrA": 839, "rpsL": 125}
    for li, p in enumerate(m._plan):
        mp = {k: getattr(m, f"{k}_map_{li}").numpy() for k in ("coord", "codon", "keep")}
        locus = p["locus"]
        assert m.frames[locus]["genome_anchored"], f"{locus} not placed in the genome"
        assert (np.diff(mp["coord"][mp["keep"]]) >= -1e-6).all(), f"{locus} not monotonic"
        if locus in expect:
            assert int(mp["codon"].max()) + 1 == expect[locus], (locus, int(mp["codon"].max()) + 1)
        if locus in ("rrs", "rrl"):
            assert (mp["codon"] == -1).all(), f"{locus} has codons"
    assert m.frames["inhA"]["promoter_offset"] == -535.0
    assert m.frames["gid"]["duplicate_columns"] == 102




# --- opt-in tokenizer ablations: unknown symbols, reference properties -------
# (2026-10-01; models.locusfusion_v2.token_layout). Defaults must stay C_TOK 41.
from models.locusfusion_v2 import token_layout  # noqa: E402
from datasets.sequences import delta_one_hot_nt  # noqa: E402
from datasets.biochem import one_hot_aa  # noqa: E402

U_SPECS = [(6 if k[0] in ("dna", "regulatory") else 23 if k[0] == "protein" else c, n)
           for k, (c, n) in zip(KEYS, SPECS)]


def unet(**kw):
    torch.manual_seed(0)
    kw.setdefault("block_frames", FRAMES)
    m = LocusFusionV2Net(KEYS, U_SPECS, unknown_symbols=True, **kw)
    m.eval()
    return m


def _test_default_layout_is_unchanged():
    L = token_layout()
    assert L["slots"] == SLOTS and L["d_bio"] == D_BIO and L["c_tok"] == C_TOK == 41
    assert L["f_is_null"] == F_IS_NULL and L["f_phase"] == F_PHASE and L["ref_bio"] is None
    assert np.array_equal(L["residue_properties"], RESIDUE_PROPERTIES)
    assert net().tok_proj.in_features == 41
    assert LOCUSFUSION_V2_DEFAULTS["unknown_symbols"] is False
    assert LOCUSFUSION_V2_DEFAULTS["ref_bio"] is False


def _test_layout_widths():
    assert token_layout(True, False)["c_tok"] == 45      # +N dna, +N reg, +"-" +"X" protein
    assert token_layout(False, True)["c_tok"] == 44
    assert token_layout(True, True)["c_tok"] == 48, "both on must be the draft's 48"
    assert net(ref_bio=True).tok_proj.in_features == 44
    assert unet(ref_bio=True).tok_proj.in_features == 48


def _test_encoders_give_unknowns_a_channel():
    a = delta_one_hot_nt("ACNT", "ACGT", with_unknown=True)
    assert a.shape == (4, 6) and a[2, 5] == 1 and a[2, :5].sum() == 0
    assert a[[0, 1, 3]].sum() == 0, "matching bases must still be zeroed"
    b = delta_one_hot_nt("ACNT", "ACGT")
    assert b.shape == (4, 5) and b.sum() == 0, "default: an N still reads as reference"
    aa = one_hot_aa("AX-", with_stop=True, with_unknown=True)
    assert aa.shape == (23, 3) and aa[22, 1] == 1 and aa[21, 2] == 1
    assert one_hot_aa("AX-", with_stop=True).shape == (21, 3)


def _test_uncalled_base_makes_a_token():
    m = unet()
    L = m.layout
    xs = [torch.zeros(1, c, n) for c, n in U_SPECS]
    xs[B[("dna", "gA")]][0, 5, 48] = 1.0                       # N at a coding column
    f, _, valid = token(m, xs, "gA", 1)
    assert valid, "an uncalled base produced no token"
    assert float(f[L["slots"]["dna"][0] + 5]) == 1.0
    pr = f[L["slots"]["protein"][0]:L["slots"]["protein"][1]]
    assert float(pr.abs().sum()) == 0, "an uncalled base restated the reference residue"


def _test_ref_bio_carries_the_reference_properties():
    m = net(ref_bio=True)
    rb = m.layout["ref_bio"]
    xs = put(reference_batch(1), ("dna", "gA"), 48, channel=2)  # synonymous at codon 5
    f, _, valid = token(m, xs, "gA", 1)
    ser = _RES_ORDER.index("S")
    assert valid and torch.allclose(f[rb[0]:rb[1]], torch.as_tensor(RESIDUE_PROPERTIES[ser]),
                                    atol=1e-6)
    db = m.layout["d_bio"]
    assert float(f[db[0]:db[1]].abs().sum()) == 0


def _test_channel_mismatch_is_refused():
    for build in (lambda: LocusFusionV2Net(KEYS, U_SPECS, block_frames=FRAMES),
                  lambda: LocusFusionV2Net(KEYS, SPECS, block_frames=FRAMES,
                                           unknown_symbols=True)):
        try:
            build()
        except ValueError:
            continue
        raise AssertionError("a model / data channel mismatch was not refused")


def _test_ablation_knobs_round_trip():
    import json

    class _B:
        def __init__(self, key, spec, meta):
            self.name = f"{key[0]}:{key[1]}"
            self._spec, self.meta, self.channel_names = spec, meta, []

        def spec(self):
            return self._spec

    blocks = [_B(k, s, f) for k, s, f in zip(KEYS, U_SPECS, FRAMES)]
    knobs = {"unknown_symbols": True, "ref_bio": True}
    torch.manual_seed(0)
    a = LocusFusionV2Net.from_blocks(blocks, drug_names=["A"], **knobs)
    cfg = json.loads(json.dumps(model_config(
        arch="locusfusion_v2", blocks=blocks, encoder_types=["cnn"] * len(KEYS),
        drug_names=["A"], out_bias=None,
        head={"hidden": 256, "dropout": 0.0, "per_drug_hidden": 0},
        locusfusion=knobs), allow_nan=False))
    b = build_model_from_config({"model": cfg})
    b.load_state_dict(a.state_dict(), strict=True)
    assert b.unknown_symbols and b.ref_bio and b.tok_proj.in_features == 48


if __name__ == "__main__":
    check("ablation: default layout is unchanged (C_TOK 41)", _test_default_layout_is_unchanged)
    check("ablation: layout widths 45 / 44 / 48", _test_layout_widths)
    check("ablation: encoders give N / '-' / 'X' a channel", _test_encoders_give_unknowns_a_channel)
    check("ablation: an uncalled base makes a token", _test_uncalled_base_makes_a_token)
    check("ablation: ref_bio carries H37Rv's properties", _test_ref_bio_carries_the_reference_properties)
    check("ablation: model/data channel mismatch is refused", _test_channel_mismatch_is_refused)
    check("ablation: knobs survive a JSON config", _test_ablation_knobs_round_trip)
    check("a locus matching H37Rv presents only [NULL]", _test_reference_matching_locus_is_null_only)
    check("nothing is named for wild type", _test_nothing_is_called_wild_type)
    check("only nucleotide blocks make tokens", _test_only_nucleotides_make_tokens)
    check("the residue lands on its own codon, gap-aware", _test_residue_lands_on_its_codon_gap_aware)
    check("a synonymous change restates the reference residue",
          _test_synonymous_change_restates_the_reference)
    check("a substitution carries both residues and the property delta",
          _test_substitution_carries_reference_and_delta)
    check("the property delta survives without a biophysical block",
          _test_delta_without_a_biophysical_block)
    check("a deleted base does not restate the reference residue",
          _test_a_deleted_base_does_not_restate_the_reference)
    check("flank and promoter bases carry no codon", _test_flank_and_promoter_carry_no_codon)
    check("promoter placed by genome position, + strand", _test_promoter_placed_by_genome_position_plus_strand)
    check("promoter placed by genome position, - strand", _test_promoter_placed_by_genome_position_minus_strand)
    check("the kept axis is monotonic", _test_kept_axis_is_monotonic)
    check("a base shared by record and promoter is one token", _test_shared_base_is_one_token)
    check("a base shared ACROSS loci is a token in each", _test_base_shared_across_loci_is_a_token_in_each)
    check("the cap is per locus, promoter first", _test_cap_is_per_locus_promoter_first)
    check("an all-differing locus reads full on [NULL]", _test_all_differing_locus_reads_full_on_null)
    check("rRNA genes carry no codons", _test_rrna_carries_no_codons)
    check("reference-frame translation", _test_reference_frame_translation)
    check("a stop gain carries a residue", _test_stop_gain_carries_a_residue)
    check("isolate-frame protein input is rejected", _test_isolate_frame_protein_rejected)
    check("missing frames warn and still run", _test_missing_frames_warn_and_run)
    check("dense input warns", _test_dense_input_warns)
    check("bad construction is rejected", _test_bad_construction_rejected)
    check("backward reaches every parameter", _test_backward_reaches_every_parameter)
    check("a checkpoint config rebuilds the same model", _test_config_round_trips)
    check("feature layout", _test_feature_layout)
    check("structure: width, and old checkpoints unchanged",
          _test_structure_width_and_checkpoint_compat)
    check("structure: the other channels are untouched",
          _test_structure_leaves_the_other_channels_alone)
    check("structure: a token's row is its distance-matrix row",
          _test_structure_row_is_the_distance_matrix)
    check("structure: one codon is 0 A, not masked", _test_structure_same_codon_is_zero_not_masked)
    check("structure: stop, rRNA and [NULL] are masked", _test_structure_masks_stop_rrna_and_null)
    check("structure: bad construction is rejected", _test_structure_bad_construction_rejected)
    check("structure: the decay length is a knob, default 10 A", _test_structure_length_scale)
    check("structure: backward, and a JSON config rebuilds it",
          _test_structure_backward_and_round_trip)
    check("structure: invalid entries read struct_invalid", _test_structure_invalid_value)
    check("structure: heavy-atom distances", _test_structure_heavy_atoms)
    check("structure: the contact gate", _test_structure_contact_gate)
    check("structure: new knobs survive a JSON config", _test_structure_knobs_round_trip)
    check("real frames, all loci (skipped without data)", _test_real_frames_all_loci)
    n_fail = _RESULTS.count(False)
    print(f"\n{len(_RESULTS) - n_fail}/{len(_RESULTS)} passed")
    sys.exit(1 if n_fail else 0)

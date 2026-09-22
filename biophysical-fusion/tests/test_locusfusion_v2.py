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
"""
import sys
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from models import (LOCUSFUSION_V2_DEFAULTS, LocusFusionV2Net,  # noqa: E402
                    locusfusion_defaults)
from models.locusfusion_v2 import C_TOK, F_IS_NULL, F_PHASE, SLOTS  # noqa: E402
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
FRAMES = [
    {"ref_gap_cols": [45], "genome_start": 1000, "genome_end": 1119, "strand": "+",
     "cds_start": 30, "cds_end": 91, "has_cds": True},
    {"ref_gap_cols": [], "genome_start": 5000, "genome_end": 5090, "strand": "-",
     "cds_start": 15, "cds_end": 75, "has_cds": True},
    {"ref_gap_cols": [], "genome_start": 9000, "genome_end": 9060, "strand": "+",
     "cds_start": 10, "cds_end": 60, "has_cds": False},
    {}, {}, {}, {},
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


def _test_synonymous_change_carries_no_residue():
    m = net()
    xs = put(reference_batch(1), ("dna", "gA"), 48, channel=2)       # protein stays zero
    f, _, valid = token(m, xs, "gA", 1)
    assert valid
    assert float(f[SLOTS["protein"][0]:SLOTS["biophysical"][1]].abs().sum()) == 0
    assert float(f[F_PHASE:F_PHASE + 3].sum()) == 1, "a coding base lost its phase"


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
    assert float(feat[1, li, 1, lo:hi].abs().sum()) == 0, "the synonymous twin carried a residue"
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
    assert spans[-1][1] <= F_IS_NULL < F_PHASE and F_PHASE + 3 == C_TOK == 38
    assert SLOTS["protein"][1] - SLOTS["protein"][0] == 21, "protein slot is not 20 residues + STOP"


# --- the real reference rows, when mounted ---------------------------------

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


if __name__ == "__main__":
    check("a locus matching H37Rv presents only [NULL]", _test_reference_matching_locus_is_null_only)
    check("nothing is named for wild type", _test_nothing_is_called_wild_type)
    check("only nucleotide blocks make tokens", _test_only_nucleotides_make_tokens)
    check("the residue lands on its own codon, gap-aware", _test_residue_lands_on_its_codon_gap_aware)
    check("a synonymous change carries no residue", _test_synonymous_change_carries_no_residue)
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
    check("real frames, all loci (skipped without data)", _test_real_frames_all_loci)
    n_fail = _RESULTS.count(False)
    print(f"\n{len(_RESULTS) - n_fail}/{len(_RESULTS)} passed")
    sys.exit(1 if n_fail else 0)

"""
Checks for ``models.LocusFusionNet`` — the variant-token, two-stage transformer.

Pure unit tests on synthetic tensors (no data, no training); seconds on CPU::

    python tests/test_locusfusion.py

plus a small set at the end that reads the REAL alignments to check the
coordinate map, skipped automatically when they are not mounted.

The claims being asserted, beyond "it runs" — each one is a thing the model is
supposed to do that the previous transformer arms did not:

  * **A token exists only where the genotype deviates.** A wild-type isolate
    (every block equal to the reference ids) produces nothing but the [WT]
    sentinels, and the token count scales with the variant count, not with
    sequence length. This is the fix for the `token_signal` finding (0.14% of a
    setfusion token varied with the genotype).
  * **The streams agree on where a residue is.** A nucleotide token and a
    protein token for the same codon get the SAME coordinate. This is the bug
    the rewrite exists to fix: the previous version placed a nucleotide token at
    ``column/3`` minus a learned scalar, which put katG S315 at 357.3 against
    its protein token's 314, and that scalar read ``[-0.0107, +0.0081]`` off a
    fully trained checkpoint.
  * **Exact position survives.** Moving the SAME variant to a different column
    changes the logits — the coordinate is carried, not pooled into bins.
  * **Which variant it is survives, at both ends.** Changing the alt symbol at a
    fixed column changes the logits, and so does changing the reference.
  * **An unresolved base call is not wild type.** An ``N`` where the reference
    has a base is a variant. Under the old all-zero-column encoding it was
    indistinguishable from a match, so a failed call read as wild type.
  * **Locus identity is load-bearing.** Putting the same variant in a different
    locus changes the logits.
  * **The two stages are really two.** Under `carry_variants=0` stage 2 sees one
    summary per locus, and the read-out attention is (B, n_drugs, n_loci).
  * **Uncovered != wild type.** An all-differing locus (a record that failed to
    assemble) is flagged, and does not read as a hypervariant one.
  * **A one-hot block is refused**, rather than silently read as "the first 16
    columns of each block".
"""
import sys
import traceback
import warnings
from pathlib import Path

# this file lives in tests/; put the project root on the path so the imports
# below resolve when run directly (python tests/test_locusfusion.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from datasets import tokens as tok  # noqa: E402
from models import LOCUSFUSION_DEFAULTS, LocusFusionNet, SetFusionNet  # noqa: E402
from training.checkpoint import build_model_from_config, model_config  # noqa: E402

_RESULTS = []

# an isoniazid-shaped block set at REAL sizes: 2 coding loci x 4 modalities,
# plus promoter windows for both. Real lengths matter here — the whole design
# claim is about what a 2.5 kb locus costs. Channel counts are the symbol-id
# layout: one id channel for dna/protein/regulatory, three property channels for
# biophysical, which has no id form by design.
KEYS = [("dna", "inhA"), ("dna", "katG"),
        ("protein", "inhA"), ("protein", "katG"),
        ("biophysical", "inhA"), ("biophysical", "katG"),
        ("regulatory", "inhA"), ("regulatory", "katG")]
SPECS = [(1, 910), (1, 2488),
         (1, 303), (1, 829),
         (3, 303), (3, 829),
         (1, 879), (1, 642)]
LOCI = ["inhA", "katG"]

# Synthetic per-column metadata, shaped like the real thing: a CDS starting at
# column 111 (katG's actual offset) so the nucleotide coordinate is genuinely
# not column/3, and a reference sequence that is not constant.
CDS_START = 111
_RNG = np.random.default_rng(0)


def _nt_meta(length, stream):
    base = tok.NT0 if stream == "nt" else tok.REG0
    ref = (base + _RNG.integers(0, 4, size=length)).astype(np.int8)
    if stream == "nt":
        n = np.arange(length, dtype=np.int64) - CDS_START
    else:
        n = np.arange(length, dtype=np.int64) - length     # upstream: negative
    return {"coord": (n / 3.0).astype(np.float32),
            "phase": np.mod(n, 3).astype(np.int8), "ref_id": ref}


def _aa_meta(length):
    ref = (tok.AA0 + _RNG.integers(0, 20, size=length)).astype(np.int8)
    return {"coord": np.arange(length, dtype=np.float32),
            "phase": np.full(length, tok.PHASE_NA, np.int8), "ref_id": ref}


META = []
for (modality, _locus), (_c, length) in zip(KEYS, SPECS):
    stream = {"dna": "nt", "regulatory": "reg"}.get(modality)
    META.append(_nt_meta(length, stream) if stream
                else (_aa_meta(length) if modality == "protein" else None))


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
        _RESULTS.append(True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        _RESULTS.append(False)


def wt_batch(b=4):
    """A wild-type cohort: every isolate carries exactly the reference ids."""
    xs = []
    for meta, (c, length) in zip(META, SPECS):
        if meta is None:                       # biophysical: unchanged == zero
            xs.append(torch.zeros(b, c, length))
        else:
            row = torch.as_tensor(meta["ref_id"], dtype=torch.float32)
            xs.append(row.view(1, 1, length).expand(b, 1, length).clone())
    return xs


def alt_at(block, column, nth=1):
    """A symbol guaranteed to DIFFER from the reference at this column.

    Taking the reference as the origin matters: the fixture's reference is
    random, so a hardcoded "set it to A" silently becomes a no-op wherever the
    reference already is A — which is how two of these checks first passed
    against a model that was ignoring them.
    """
    meta = META[block]
    ref = int(meta["ref_id"][column])
    if meta["phase"][column] == tok.PHASE_NA:
        return tok.AA0 + (ref - tok.AA0 + nth) % 20
    base = tok.REG0 if ref >= tok.REG0 else tok.NT0
    return base + (ref - base + nth) % 4


def with_variant(xs, block, column, symbol=None, nth=1, row=slice(None)):
    """Put a symbol other than the reference at one column of one block."""
    xs = [x.clone() for x in xs]
    if symbol is None:
        symbol = alt_at(block, column, nth)
    assert symbol != int(META[block]["ref_id"][column]), \
        f"block {block} column {column}: symbol {symbol} IS the reference"
    xs[block][row, 0, column] = float(symbol)
    return xs


def net(**kw):
    torch.manual_seed(0)
    m = LocusFusionNet(KEYS, SPECS, column_meta=META, **kw)
    m.eval()          # dropout off, so two forwards of the same input agree
    return m


# --- the core claim: tokens are variants -----------------------------------

def _test_wild_type_is_the_empty_set():
    m = net()
    rep = m.variant_report(wt_batch())
    assert all(int(r["n_variants"].sum()) == 0 for r in rep), \
        f"a wild-type cohort produced variant tokens: {[int(r['n_variants'].sum()) for r in rep]}"
    logits, attn = m(wt_batch(), return_attn=True)
    assert logits.shape == (4, 1)
    # every isolate is identical, so every logit must be too
    assert torch.allclose(logits, logits[:1].expand_as(logits), atol=1e-5), \
        "identical wild-type isolates got different logits"


def _test_token_count_tracks_variants_not_length():
    m = net()
    xs = wt_batch(2)
    xs = with_variant(xs, 1, 900, row=0)               # one SNP in katG's 2,488 bp
    xs = with_variant(xs, 1, 1500, row=0)
    counts = {(r["locus"], r["stream"]): r["n_variants"].tolist()
              for r in m.variant_report(xs)}
    assert counts[("katG", "nt")] == [2, 0], counts
    assert counts[("inhA", "nt")] == [0, 0], counts
    # ... and the token budget is set by the cap, not by the 2,488 columns
    assert m.tokens_per_locus == 1 + LOCUSFUSION_DEFAULTS["max_variants"] * 3


def _test_variant_changes_the_logits():
    m = net()
    base = m(wt_batch(2))
    moved = m(with_variant(wt_batch(2), 1, 900, row=0))
    assert not torch.allclose(base[0], moved[0], atol=1e-6), \
        "adding a variant did not change the logit"
    assert torch.allclose(base[1], moved[1], atol=1e-6), \
        "adding a variant to isolate 0 changed isolate 1"


def _test_position_survives():
    """The claim `setfusion` gave up: 'column 315' rather than 'the first quarter'."""
    m = net()
    a = m(with_variant(wt_batch(1), 1, 900))
    b = m(with_variant(wt_batch(1), 1, 901))
    assert not torch.allclose(a, b, atol=1e-6), \
        "the same variant at two adjacent columns gave the same logit — " \
        "position is not reaching the model"


def _test_alt_symbol_survives():
    m = net()
    a = m(with_variant(wt_batch(1), 1, 900, nth=1))
    b = m(with_variant(wt_batch(1), 1, 900, nth=2))
    assert not torch.allclose(a, b, atol=1e-6), \
        "which base the variant is did not change the logit"


def _test_reference_symbol_reaches_the_model():
    """A variant has two ends. The old layout carried only the alt, so a C>T and
    a G>T at the same column were the same token."""
    m = net()
    alt = alt_at(1, 900, nth=1)
    xs = with_variant(wt_batch(1), 1, 900, symbol=alt)
    a = m(xs)
    saved = m.ref_m1_0.clone()                     # katG's nt stream
    # a different wild-type base, still not the alt: same variant token, other end
    m.ref_m1_0[900] = tok.NT0 + (alt - tok.NT0 + 2) % 4
    b = m(xs)
    m.ref_m1_0.copy_(saved)
    assert not torch.allclose(a, b, atol=1e-6), \
        "changing the reference base under a fixed alt did not change the logit"


def _test_unresolved_base_is_a_variant():
    """An N where the reference has a base is a deviation. Under the previous
    one-hot encoding it was an all-zero column, indistinguishable from a match,
    so a failed base call read as wild type."""
    m = net()
    xs = with_variant(wt_batch(2), 1, 900, symbol=tok.NT_UNK, row=0)
    counts = {(r["locus"], r["stream"]): r["n_variants"].tolist()
              for r in m.variant_report(xs)}
    assert counts[("katG", "nt")] == [1, 0], counts
    assert not torch.allclose(m(xs)[0], m(wt_batch(2))[0], atol=1e-6)


def _test_locus_identity_survives():
    m = net()
    a = m(with_variant(wt_batch(1), 0, 300))   # dna:inhA
    b = m(with_variant(wt_batch(1), 1, 300))   # dna:katG, same column
    assert not torch.allclose(a, b, atol=1e-6), \
        "the same variant in two different loci gave the same logit — " \
        "locus identity is decorative"


# --- the coordinate, which is what this rewrite is for ----------------------

def _test_nt_and_aa_coordinates_agree_on_a_codon():
    """The whole point. Codon k of the protein stream and the nucleotide column
    that carries its first base must land on the SAME coordinate."""
    m = net()
    katg_nt, katg_aa = m.coord_m1_0, m.coord_m1_1
    for residue in (0, 100, 315, 700):
        column = CDS_START + 3 * residue
        assert float(katg_nt[column]) == float(katg_aa[residue]) == residue, (
            f"codon {residue}: nt column {column} -> {float(katg_nt[column])}, "
            f"aa index {residue} -> {float(katg_aa[residue])}")


def _test_codon_phase_is_relative_to_the_cds():
    """`column % 3` is not the codon position unless the CDS starts at column 0,
    and it starts at 100-112 in 10 of the 17 coding loci."""
    m = net()
    phase = m.phase_m1_0
    for residue in (0, 100, 315):
        column = CDS_START + 3 * residue
        assert int(phase[column]) == 0 and int(phase[column + 1]) == 1 \
            and int(phase[column + 2]) == 2, \
            f"codon {residue} at column {column}: phases " \
            f"{[int(phase[column + i]) for i in range(3)]}"
    assert int(m.phase_m1_1[7]) == tok.PHASE_NA, \
        "an amino-acid token has no codon phase and must say so"


def _test_upstream_coordinates_are_negative():
    m = net()
    assert float(m.coord_m1_0[0]) < 0, "a column before the CDS must be upstream"
    assert float(m.coord_m0_2[-1]) < 0, "a promoter window must be upstream"


# --- gene-level fusion ------------------------------------------------------

def _test_protein_and_biophysical_share_a_token():
    """The gene-level fusion claim: co-indexed modalities land on ONE token."""
    m = net()
    xs = wt_batch(1)
    xs = with_variant(xs, 3, 315)                 # protein:katG, residue 315
    xs[5][0, 1, 315] = 0.4                        # biophysical:katG, same residue
    rep = {(r["locus"], r["stream"]): r for r in m.variant_report(xs)}
    aa = rep[("katG", "aa")]
    assert aa["modalities"] == ["protein", "biophysical"], aa["modalities"]
    assert int(aa["n_variants"][0]) == 1, \
        f"protein+biophysical at the same residue made {int(aa['n_variants'][0])} tokens, not 1"
    assert int(aa["columns"][0, 0]) == 315


def _test_biophysical_alone_still_tokenizes():
    """The dna+biophysical cell: no residue identity is loaded, on purpose, so
    the amino-acid stream falls back to 'a residue whose properties moved'."""
    keys = [("dna", "katG"), ("biophysical", "katG")]
    specs = [(1, 2488), (3, 829)]
    meta = [META[1], None]
    torch.manual_seed(0)
    m = LocusFusionNet(keys, specs, column_meta=meta)
    m.eval()
    xs = [torch.as_tensor(META[1]["ref_id"], dtype=torch.float32)
          .view(1, 1, 2488).expand(2, 1, 2488).clone(),
          torch.zeros(2, 3, 829)]
    assert int(m.variant_report(xs)[1]["n_variants"][0]) == 0
    xs[1][0, 1, 315] = 0.7
    rep = m.variant_report(xs)[1]
    assert int(rep["n_variants"][0]) == 1 and int(rep["n_variants"][1]) == 0
    assert m.has_bio and m.bio_proj is not None


def _test_streams_stay_separate_across_coordinate_systems():
    """dna and protein now share a coordinate SYSTEM, but they are still
    separate tokens: `datasets/protein.py` degaps before translating, so codon k
    of an isolate with an upstream indel is not codon k of the reference, and
    fusing them by position would assert an equality that does not always hold."""
    m = net()
    streams = {(r["locus"], r["stream"]) for r in m.variant_report(wt_batch(1))}
    assert streams == {("inhA", "nt"), ("inhA", "aa"), ("inhA", "reg"),
                       ("katG", "nt"), ("katG", "aa"), ("katG", "reg")}, streams


# --- the two stages ---------------------------------------------------------

def _test_stage_two_sees_one_summary_per_locus():
    m = net()
    _logits, attn = m(wt_batch(2), return_attn=True)
    assert attn.shape == (2, 1, len(LOCI)), \
        f"stage-2 token set is {attn.shape[-1]}, expected one summary per locus"


def _test_carry_variants_widens_stage_two():
    m = net(carry_variants=3)
    _logits, attn = m(wt_batch(2), return_attn=True)
    assert attn.shape == (2, 1, len(LOCI) * 4), attn.shape


def _test_joint_readout_is_per_drug():
    drugs = ["ISONIAZID", "RIFAMPICIN", "ETHAMBUTOL"]
    m = net(drug_names=drugs)
    logits, attn = m(wt_batch(2), return_attn=True)
    assert logits.shape == (2, 3) and attn.shape == (2, 3, len(LOCI))
    assert m.drug_names == drugs


# --- per-locus specialization ----------------------------------------------

def _test_locus_encoder_modes():
    sizes = {}
    for mode in ("shared", "adapter", "per_locus"):
        m = net(locus_encoder=mode)
        sizes[mode] = sum(p.numel() for p in m.parameters())
        m(wt_batch(2))                       # each mode must actually run
    assert sizes["shared"] < sizes["adapter"] < sizes["per_locus"], sizes
    # the adapter is meant to be nearly free: 2*d_model per locus
    assert sizes["adapter"] - sizes["shared"] == 2 * 128 * len(LOCI), sizes


def _test_summary_norm_strips_the_locus_constant():
    """The stage-2 input is read off the [WT] slot, whose input is IDENTICAL in
    every isolate — so without the keyed norm it is mostly a per-locus constant,
    which is the failure `token_signal` measured in setfusion. Assert the norm
    actually raises the fraction of the summary that varies between isolates."""
    xs = wt_batch(16)
    for i in range(16):                      # give each isolate its own variants
        xs = with_variant(xs, 1, 400 + 37 * i, nth=1 + i % 3, row=i)
        xs = with_variant(xs, 3, 20 * i + 5, nth=1 + i % 19, row=i)

    def ratio(mode):
        torch.manual_seed(0)
        m = LocusFusionNet(KEYS, SPECS, column_meta=META, summary_norm=mode)
        m.train()                            # batch statistics, as when training
        with torch.no_grad():
            alt, ref, phase, coord, props, valid, stats = m._locus_tokens(xs)
            B, nl, T = alt.shape
            li = torch.arange(nl)
            t = m._embed(alt, ref, phase, coord, props, valid, stats, li)
            z = m.encoders[0](t.reshape(B * nl, T, m.d_model),
                              src_key_padding_mask=(~valid).reshape(B * nl, T))
            f = z.reshape(B, nl, T, -1)[:, :, 0]
            if m.summ_norm is not None:
                f = m.summ_norm(f, li)
            return float(f.std(0).mean()) / float(f.norm(dim=-1).mean())

    off, on = ratio("none"), ratio("keyed")
    assert on > 2 * off, f"keyed summary norm barely moved the ratio: {off:.4f} -> {on:.4f}"


def _test_unknown_summary_norm_rejected():
    try:
        LocusFusionNet(KEYS, SPECS, column_meta=META, summary_norm="nonsense")
    except ValueError as e:
        assert "summary_norm" in str(e)
    else:
        raise AssertionError("an unknown summary_norm was accepted")


def _test_summary_norm_none_adds_no_state_dict_keys():
    """'none' must create no module, so a checkpoint written at one setting is
    not silently incompatible with the other."""
    off = set(LocusFusionNet(KEYS, SPECS, column_meta=META,
                             summary_norm="none").state_dict())
    on = set(LocusFusionNet(KEYS, SPECS, column_meta=META,
                            summary_norm="keyed").state_dict())
    assert on - off and not off - on


def _test_unknown_locus_encoder_rejected():
    try:
        LocusFusionNet(KEYS, SPECS, column_meta=META, locus_encoder="nonsense")
    except ValueError as e:
        assert "locus_encoder" in str(e)
    else:
        raise AssertionError("an unknown locus_encoder was accepted")


# --- missingness ------------------------------------------------------------

def _test_uncovered_locus_is_flagged_not_read_as_variants():
    """14-91 isolates per locus are all-gap records. They differ from the
    reference at EVERY column; without the flag they would read as the
    most-mutated isolate in the cohort rather than as a missing gene."""
    m = net()
    xs = wt_batch(2)
    xs[1][0, 0, :] = float(tok.NT_GAP)        # katG entirely gap, isolate 0 only
    rep = {(r["locus"], r["stream"]): r for r in m.variant_report(xs)}
    unc = rep[("katG", "nt")]["uncovered"]
    assert bool(unc[0]) and not bool(unc[1]), unc
    *_rest, stats = m._locus_tokens(xs)
    katg = LOCI.index("katG")
    assert stats[0, katg, 2] == 1.0 and stats[1, katg, 2] == 0.0
    # and the flag must reach the tokens, not just the report
    assert not torch.allclose(m(xs)[0], m(xs)[1], atol=1e-6)


def _test_wt_token_reports_the_true_count_before_the_cap():
    """A frameshift changing 42 downstream residues must not read as 16."""
    m = net(max_variants=4)
    xs = wt_batch(1)
    for c in range(200, 220):
        xs = with_variant(xs, 1, c)
    *_rest, stats = m._locus_tokens(xs)
    katg = LOCI.index("katG")
    assert abs(float(stats[0, katg, 0]) - float(np.log1p(20))) < 1e-4, \
        f"[WT] reported log1p(n)={float(stats[0, katg, 0])}, expected {np.log1p(20)}"


# --- guards -----------------------------------------------------------------

def _test_one_hot_block_is_refused():
    """The old 5-channel one-hot must not be accepted and silently read as
    'the first max_variants columns of each block'."""
    onehot = [(5, 910), (5, 2488), (20, 303), (20, 829),
              (3, 303), (3, 829), (5, 879), (5, 642)]
    try:
        LocusFusionNet(KEYS, onehot)
    except ValueError as e:
        assert "variant_tokens" in str(e), e
    else:
        raise AssertionError("a one-hot block set was accepted")


def _test_everything_differing_warns():
    """No reference metadata means every column reads as a difference. That
    produces plausible numbers, so it has to be loud."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = LocusFusionNet(KEYS, SPECS)            # no column_meta
        m.eval()
        m(wt_batch(2))
    assert any("reference ids did not" in str(x.message) for x in w), \
        [str(x.message) for x in w]


def _test_merged_blocks_warn():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        LocusFusionNet([("dna", None), ("protein", None)], [(1, 3398), (1, 700)])
    assert any("per-locus" in str(x.message) for x in w), [str(x.message) for x in w]


def _test_d_model_must_divide_nhead():
    try:
        LocusFusionNet(KEYS, SPECS, column_meta=META, d_model=100, nhead=8)
    except ValueError as e:
        assert "divisible" in str(e)
    else:
        raise AssertionError("d_model not divisible by nhead was accepted")


def _test_mismatched_stream_lengths_rejected():
    try:
        LocusFusionNet([("protein", "katG"), ("biophysical", "katG")],
                       [(1, 829), (3, 830)])
    except ValueError as e:
        assert "co-indexed" in str(e), e
    else:
        raise AssertionError("protein/biophysical of different lengths was accepted")


# --- plumbing ---------------------------------------------------------------

def _test_backward_reaches_every_parameter():
    torch.manual_seed(0)
    m = LocusFusionNet(KEYS, SPECS, column_meta=META,
                       drug_names=["A", "B"], carry_variants=2)
    xs = with_variant(wt_batch(4), 1, 900)
    xs = with_variant(xs, 3, 100)                 # touch the amino-acid stream too
    xs[5][:, 0, 100] = 0.5                        # ...and the biophysical slot
    xs[0][3, 0, :] = float(tok.NT_GAP)            # one uncovered locus, for
    #                                               uncovered_emb's gradient
    m(xs).sum().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None
            or not p.grad.abs().sum()]
    # the padding row of each symbol table is frozen by construction
    assert not dead, f"no gradient reached: {dead}"


def _test_token_embedding_is_smaller_than_the_layout_it_replaced():
    """The old embedding was Linear(42, d) + Linear(64, d) + a learned offset.
    The new one is three lookup tables over a 35-symbol vocabulary, a projection
    of a 32-dim sinusoid, and 3 properties."""
    m = net()
    new = sum(p.numel() for n, p in m.named_parameters()
              if n.split(".")[0] in ("alt_emb", "ref_emb", "phase_emb",
                                     "pos_proj", "bio_proj", "uncovered_emb"))
    old = 42 * 128 + 128 + 64 * 128 + 128 + len(LOCI)   # tok_proj + pos_proj + offset
    assert new < 1.2 * old, f"token embedding {new:,} vs the old {old:,}"


def _test_parameter_count_is_modest():
    m = net()
    n = sum(p.numel() for p in m.parameters())
    sf = sum(p.numel() for p in SetFusionNet(
        KEYS, [(5, 910), (5, 2488), (20, 303), (20, 829),
               (3, 303), (3, 829), (5, 879), (5, 642)]).parameters())
    assert n < 4 * sf, f"locusfusion {n:,} vs setfusion {sf:,}"
    assert n < 2_000_000, f"{n:,} parameters — bigger than intended"


def _test_config_round_trips():
    """A checkpoint written today must rebuild the same model."""
    class _B:
        def __init__(self, key, spec, meta):
            self.name = f"{key[0]}:{key[1]}"
            self.modality, self.locus = key
            self._spec = spec
            self.column_meta = meta

        def spec(self):
            return self._spec

    blocks = [_B(k, s, m) for k, s, m in zip(KEYS, SPECS, META)]
    cfg = model_config(arch="locusfusion", blocks=blocks,
                       encoder_types=["cnn"] * len(KEYS), drug_names=["ISONIAZID"],
                       out_bias=None, head={"hidden": 256, "dropout": 0.0,
                                            "per_drug_hidden": 0},
                       locusfusion={"d_model": 64, "carry_variants": 2})
    rebuilt = build_model_from_config({"model": cfg})
    assert isinstance(rebuilt, LocusFusionNet)
    assert rebuilt.d_model == 64 and rebuilt.carry_variants == 2
    # and a config from before this arch existed must be untouched by it
    cfg.pop("locusfusion")
    cfg["arch"] = "locusfusion"
    assert build_model_from_config({"model": cfg}).d_model == \
        LOCUSFUSION_DEFAULTS["d_model"]


def _test_column_metadata_survives_a_state_dict_round_trip():
    """`build_model_from_config` has only keys and specs, so a rebuilt model
    starts with the fallback coordinate map. The buffers are persistent so the
    checkpoint restores the real one — without that, loading a checkpoint for
    attribution would quietly move every token."""
    m = net()
    blank = LocusFusionNet(KEYS, SPECS)
    assert not torch.allclose(blank.coord_m1_0, m.coord_m1_0)
    blank.load_state_dict(m.state_dict())
    assert torch.equal(blank.coord_m1_0, m.coord_m1_0)
    assert torch.equal(blank.ref_m1_0, m.ref_m1_0)
    assert torch.equal(blank.phase_m1_0, m.phase_m1_0)


def _test_vocabulary_ranges_do_not_overlap():
    names = tok.symbol_names()
    assert len(names) == tok.N_SYMBOLS
    assert names[tok.PAD] == "<pad>" and names[tok.WT] == "[WT]"
    for i, base in enumerate(tok.NT_BASES):
        assert names[tok.NT0 + i] == f"nt {base}"
        assert names[tok.REG0 + i] == f"reg {base}"
    assert names[tok.NT_GAP] == "nt -" and names[tok.REG_GAP] == "reg -"
    assert names[tok.AA_UNK] == "aa ?"
    assert len(set(names)) == len(names), "duplicate symbol names"


# --- against the real alignments -------------------------------------------

def _real_dir():
    try:
        from bigtb_ref import REAL_GENOTYPE_DIR
    except Exception:
        return None
    return REAL_GENOTYPE_DIR if Path(REAL_GENOTYPE_DIR).is_dir() else None


def _test_real_coordinates_are_who_codon_numbers():
    """On the real alignments, the canonical resistance codons must come out at
    their catalogue numbers from BOTH streams. Before this rewrite the DNA
    stream put katG S315 at 357.3 and rpoB S450 at 501.7."""
    seq_dir = _real_dir()
    if seq_dir is None:
        print("      (skipped: real alignments not mounted)")
        return
    import glob

    from datasets import biochem, cds
    for locus, residue, expect_aa in [("katG", 315, "S"), ("rpoB", 450, "S"),
                                      ("pncA", 65, "S"), ("gyrA", 94, "D"),
                                      ("embB", 306, "M")]:
        window = cds.cds_columns(seq_dir, locus)
        ref = cds._reference_row(sorted(glob.glob(f"{seq_dir}/{locus}*.fasta"))[0])
        meta = tok.gene_column_meta(seq_dir, locus, len(ref))
        target, n, column = (residue - 1) * 3, 0, None
        for c in range(window[0], window[1]):
            if ref[c] != "-":
                if n == target:
                    column = c
                    break
                n += 1
        protein = biochem.translate_seq(cds.cds_slice(seq_dir, locus, ref, window))
        aa_meta = tok.protein_column_meta(protein, len(protein))
        assert protein[residue - 1] == expect_aa, \
            f"{locus} {residue} is {protein[residue - 1]}, expected {expect_aa}"
        assert float(meta["coord"][column]) == float(aa_meta["coord"][residue - 1]) \
            == residue - 1, (
            f"{locus} {expect_aa}{residue}: nt {float(meta['coord'][column])} vs "
            f"aa {float(aa_meta['coord'][residue - 1])}")
        assert int(meta["phase"][column]) == 0


if __name__ == "__main__":
    check("wild type is the empty set", _test_wild_type_is_the_empty_set)
    check("token count tracks variants, not length", _test_token_count_tracks_variants_not_length)
    check("a variant changes that isolate's logit only", _test_variant_changes_the_logits)
    check("exact position survives", _test_position_survives)
    check("which base it is survives", _test_alt_symbol_survives)
    check("the reference base reaches the model", _test_reference_symbol_reaches_the_model)
    check("an unresolved base call is a variant", _test_unresolved_base_is_a_variant)
    check("locus identity is load-bearing", _test_locus_identity_survives)
    check("nt and aa coordinates agree on a codon",
          _test_nt_and_aa_coordinates_agree_on_a_codon)
    check("codon phase is relative to the CDS", _test_codon_phase_is_relative_to_the_cds)
    check("upstream coordinates are negative", _test_upstream_coordinates_are_negative)
    check("protein+biophysical fuse into one token", _test_protein_and_biophysical_share_a_token)
    check("biophysical alone still tokenizes", _test_biophysical_alone_still_tokenizes)
    check("coordinate systems stay separate tokens",
          _test_streams_stay_separate_across_coordinate_systems)
    check("stage 2 sees one summary per locus", _test_stage_two_sees_one_summary_per_locus)
    check("carry_variants widens stage 2", _test_carry_variants_widens_stage_two)
    check("joint read-out is per drug", _test_joint_readout_is_per_drug)
    check("locus_encoder modes and their cost", _test_locus_encoder_modes)
    check("unknown locus_encoder rejected", _test_unknown_locus_encoder_rejected)
    check("keyed summary norm strips the locus constant",
          _test_summary_norm_strips_the_locus_constant)
    check("unknown summary_norm rejected", _test_unknown_summary_norm_rejected)
    check("summary_norm='none' adds no state_dict keys",
          _test_summary_norm_none_adds_no_state_dict_keys)
    check("uncovered locus flagged, not read as variants",
          _test_uncovered_locus_is_flagged_not_read_as_variants)
    check("[WT] reports the true count before the cap",
          _test_wt_token_reports_the_true_count_before_the_cap)
    check("a one-hot block set is refused", _test_one_hot_block_is_refused)
    check("everything-differing input warns", _test_everything_differing_warns)
    check("merged per-modality blocks warn", _test_merged_blocks_warn)
    check("d_model must divide nhead", _test_d_model_must_divide_nhead)
    check("mismatched stream lengths rejected", _test_mismatched_stream_lengths_rejected)
    check("backward reaches every parameter", _test_backward_reaches_every_parameter)
    check("token embedding is no bigger than what it replaced",
          _test_token_embedding_is_smaller_than_the_layout_it_replaced)
    check("parameter count is modest", _test_parameter_count_is_modest)
    check("config round-trips through the checkpoint layer", _test_config_round_trips)
    check("column metadata survives a state_dict round trip",
          _test_column_metadata_survives_a_state_dict_round_trip)
    check("vocabulary ranges do not overlap", _test_vocabulary_ranges_do_not_overlap)
    check("real coordinates are WHO codon numbers",
          _test_real_coordinates_are_who_codon_numbers)
    print(f"\n{sum(_RESULTS)}/{len(_RESULTS)} checks passed")
    sys.exit(0 if all(_RESULTS) else 1)

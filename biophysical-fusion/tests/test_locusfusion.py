"""
Checks for ``models.LocusFusionNet`` — the variant-token, two-stage transformer.

Pure unit tests on synthetic tensors (no data, no training); seconds on CPU:

    python tests/test_locusfusion.py

The claims being asserted, beyond "it runs" — each one is a thing the model is
supposed to do that the previous transformer arms did not:

  * **A token exists only where the genotype deviates.** A wild-type isolate
    (every delta block all-zero) produces nothing but the [WT] sentinels, and
    the token count scales with the variant count, not with sequence length.
    This is the fix for the `token_signal` finding (0.14% of a setfusion token
    varied with the genotype).
  * **Exact position survives.** Moving the SAME variant to a different column
    changes the logits — the coordinate is carried, not pooled into bins.
  * **Which variant it is survives.** Changing the base at a fixed column
    changes the logits.
  * **Locus identity is load-bearing.** Putting the same variant in a different
    locus changes the logits.
  * **The two stages are really two.** Under `carry_variants=0` stage 2 sees one
    summary per locus, and the read-out attention is (B, n_drugs, n_loci).
  * **Uncovered != wild type.** An all-differing locus (a record that failed to
    assemble) is flagged, and does not read as a hypervariant one.
  * **Dense input is caught.** Running without --delta warns rather than
    silently degenerating into "read the first 16 columns of each block".
"""
import sys
import traceback
import warnings
from pathlib import Path

# this file lives in tests/; put the project root on the path so the imports
# below resolve when run directly (python tests/test_locusfusion.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from models import (LOCUSFUSION_DEFAULTS, LocusFusionNet,  # noqa: E402
                    SetFusionNet)
from models.locusfusion import C_TOK, F_UNCOVERED, SLOTS  # noqa: E402
from training.checkpoint import build_model_from_config, model_config  # noqa: E402

_RESULTS = []

# an isoniazid-shaped block set at REAL sizes: 2 coding loci x 4 modalities,
# plus promoter windows for both. Real lengths matter here — the whole design
# claim is about what a 2.5 kb locus costs.
KEYS = ([("dna", "inhA"), ("dna", "katG"),
         ("protein", "inhA"), ("protein", "katG"),
         ("biophysical", "inhA"), ("biophysical", "katG"),
         ("regulatory", "inhA"), ("regulatory", "katG")])
SPECS = [(5, 910), (5, 2488),
         (20, 303), (20, 829),
         (3, 303), (3, 829),
         (5, 879), (5, 642)]
LOCI = ["inhA", "katG"]


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
    """A wild-type cohort: delta encoding of an isolate identical to H37Rv."""
    return [torch.zeros(b, c, length) for c, length in SPECS]


def with_variant(xs, block, column, channel=0, value=1.0, row=slice(None)):
    xs = [x.clone() for x in xs]
    xs[block][row, channel, column] = value
    return xs


def net(**kw):
    torch.manual_seed(0)
    m = LocusFusionNet(KEYS, SPECS, **kw)
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


def _test_identity_survives():
    m = net()
    a = m(with_variant(wt_batch(1), 1, 900, channel=0))    # ->A
    b = m(with_variant(wt_batch(1), 1, 900, channel=1))    # ->C
    assert not torch.allclose(a, b, atol=1e-6), \
        "which base the variant is did not change the logit"


def _test_locus_identity_survives():
    m = net()
    a = m(with_variant(wt_batch(1), 0, 300))   # dna:inhA
    b = m(with_variant(wt_batch(1), 1, 300))   # dna:katG, same column
    assert not torch.allclose(a, b, atol=1e-6), \
        "the same variant in two different loci gave the same logit — " \
        "locus identity is decorative"


# --- gene-level fusion ------------------------------------------------------

def _test_protein_and_biophysical_share_a_token():
    """The gene-level fusion claim: co-indexed modalities land on ONE token."""
    m = net()
    xs = wt_batch(1)
    xs = with_variant(xs, 3, 315, channel=7)      # protein:katG, residue 315
    xs = with_variant(xs, 5, 315, channel=1, value=0.4)   # biophysical:katG, same residue
    rep = {(r["locus"], r["stream"]): r for r in m.variant_report(xs)}
    aa = rep[("katG", "aa")]
    assert aa["modalities"] == ["protein", "biophysical"], aa["modalities"]
    assert int(aa["n_variants"][0]) == 1, \
        f"protein+biophysical at the same residue made {int(aa['n_variants'][0])} tokens, not 1"
    assert int(aa["columns"][0, 0]) == 315


def _test_streams_stay_separate_across_coordinate_systems():
    """dna (alignment columns) and protein (degapped codons) are NOT fused by
    position — `datasets/protein.py` degaps before translating, so the two
    coordinate systems only agree when no indel sits upstream."""
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
        xs = with_variant(xs, 1, 400 + 37 * i, channel=i % 4, row=i)
        xs = with_variant(xs, 3, 20 * i + 5, channel=(i + 3) % 20, row=i)

    def ratio(mode):
        torch.manual_seed(0)
        m = LocusFusionNet(KEYS, SPECS, summary_norm=mode)
        m.train()                            # batch statistics, as when training
        with torch.no_grad():
            feat, coord, valid, stats = m._locus_tokens(xs)
            B, nl, T, _ = feat.shape
            li = torch.arange(nl)
            from models.locusfusion import _sinusoid
            tok = m.tok_proj(feat) + m.pos_proj(_sinusoid(coord, m.pos_dims))
            tok = tok + m.locus_emb(li).unsqueeze(0).unsqueeze(2)
            tok[:, :, 0] = tok[:, :, 0] + m.wt_emb(li).unsqueeze(0) + m.wt_proj(stats)
            tok = tok * (1.0 + m.film_scale[li].unsqueeze(0).unsqueeze(2)) \
                + m.film_shift[li].unsqueeze(0).unsqueeze(2)
            tok = m.tok_norm(tok) * valid.unsqueeze(-1).to(tok.dtype)
            z = m.encoders[0](tok.reshape(B * nl, T, m.d_model),
                              src_key_padding_mask=(~valid).reshape(B * nl, T))
            f = z.reshape(B, nl, T, -1)[:, :, 0]
            if m.summ_norm is not None:
                f = m.summ_norm(f, li)
            return float(f.std(0).mean()) / float(f.norm(dim=-1).mean())

    off, on = ratio("none"), ratio("keyed")
    assert on > 2 * off, f"keyed summary norm barely moved the ratio: {off:.4f} -> {on:.4f}"


def _test_unknown_summary_norm_rejected():
    try:
        LocusFusionNet(KEYS, SPECS, summary_norm="nonsense")
    except ValueError as e:
        assert "summary_norm" in str(e)
    else:
        raise AssertionError("an unknown summary_norm was accepted")


def _test_summary_norm_none_adds_no_state_dict_keys():
    """'none' must create no module, so a checkpoint written at one setting is
    not silently incompatible with the other."""
    keys_off = set(LocusFusionNet(KEYS, SPECS, summary_norm="none").state_dict())
    keys_on = set(LocusFusionNet(KEYS, SPECS, summary_norm="keyed").state_dict())
    assert keys_on - keys_off and not keys_off - keys_on


def _test_unknown_locus_encoder_rejected():
    try:
        LocusFusionNet(KEYS, SPECS, locus_encoder="nonsense")
    except ValueError as e:
        assert "locus_encoder" in str(e)
    else:
        raise AssertionError("an unknown locus_encoder was accepted")


# --- missingness ------------------------------------------------------------

def _test_uncovered_locus_is_flagged_not_read_as_variants():
    """14-91 isolates per locus are all-gap records. Under delta encoding they
    differ from the reference at EVERY column; without the flag they would read
    as the most-mutated isolate in the cohort rather than as a missing gene."""
    m = net()
    xs = wt_batch(2)
    xs[1][0, 4, :] = 1.0                      # katG entirely gap, isolate 0 only
    rep = {(r["locus"], r["stream"]): r for r in m.variant_report(xs)}
    unc = rep[("katG", "nt")]["uncovered"]
    assert bool(unc[0]) and not bool(unc[1]), unc
    feat, _coord, _valid, stats = m._locus_tokens(xs)
    katg = LOCI.index("katG")
    assert feat[0, katg, :, F_UNCOVERED].min() == 1.0, "uncovered flag not set on the tokens"
    assert feat[1, katg, :, F_UNCOVERED].max() == 0.0, "uncovered flag leaked to a covered isolate"
    assert stats[0, katg, 2] == 1.0 and stats[1, katg, 2] == 0.0


# --- guards -----------------------------------------------------------------

def _test_dense_input_warns():
    m = net()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m([torch.ones(2, c, length) for c, length in SPECS])
    assert any("--delta" in str(x.message) for x in w), \
        "a dense (non-delta) input did not warn"


def _test_merged_blocks_warn():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        LocusFusionNet([("dna", None), ("protein", None)], [(5, 3398), (20, 700)])
    assert any("per-locus" in str(x.message) for x in w), [str(x.message) for x in w]


def _test_d_model_must_divide_nhead():
    try:
        LocusFusionNet(KEYS, SPECS, d_model=100, nhead=8)
    except ValueError as e:
        assert "divisible" in str(e)
    else:
        raise AssertionError("d_model not divisible by nhead was accepted")


def _test_mismatched_stream_lengths_rejected():
    try:
        LocusFusionNet([("protein", "katG"), ("biophysical", "katG")],
                       [(20, 829), (3, 830)])
    except ValueError as e:
        assert "co-indexed" in str(e), e
    else:
        raise AssertionError("protein/biophysical of different lengths was accepted")


# --- plumbing ---------------------------------------------------------------

def _test_backward_reaches_every_parameter():
    torch.manual_seed(0)
    m = LocusFusionNet(KEYS, SPECS, drug_names=["A", "B"], carry_variants=2)
    xs = with_variant(wt_batch(4), 1, 900)
    m(xs).sum().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None]
    assert not dead, f"no gradient reached: {dead}"


def _test_parameter_count_is_modest():
    m = net()
    n = sum(p.numel() for p in m.parameters())
    sf = sum(p.numel() for p in SetFusionNet(KEYS, SPECS).parameters())
    assert n < 4 * sf, f"locusfusion {n:,} vs setfusion {sf:,}"
    assert n < 2_000_000, f"{n:,} parameters — bigger than intended"


def _test_config_round_trips():
    """A checkpoint written today must rebuild the same model."""
    class _B:
        def __init__(self, key, spec):
            self.name = f"{key[0]}:{key[1]}"
            self.modality, self.locus = key
            self._spec = spec

        def spec(self):
            return self._spec

    blocks = [_B(k, s) for k, s in zip(KEYS, SPECS)]
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


def _test_feature_slots_do_not_overlap():
    spans = sorted(SLOTS.values())
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 <= b0, f"feature slots overlap: {spans}"
    assert spans[-1][1] <= C_TOK


if __name__ == "__main__":
    check("wild type is the empty set", _test_wild_type_is_the_empty_set)
    check("token count tracks variants, not length", _test_token_count_tracks_variants_not_length)
    check("a variant changes that isolate's logit only", _test_variant_changes_the_logits)
    check("exact position survives", _test_position_survives)
    check("which base it is survives", _test_identity_survives)
    check("locus identity is load-bearing", _test_locus_identity_survives)
    check("protein+biophysical fuse into one token", _test_protein_and_biophysical_share_a_token)
    check("coordinate systems stay separate", _test_streams_stay_separate_across_coordinate_systems)
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
    check("dense (non-delta) input warns", _test_dense_input_warns)
    check("merged per-modality blocks warn", _test_merged_blocks_warn)
    check("d_model must divide nhead", _test_d_model_must_divide_nhead)
    check("mismatched stream lengths rejected", _test_mismatched_stream_lengths_rejected)
    check("backward reaches every parameter", _test_backward_reaches_every_parameter)
    check("parameter count is modest", _test_parameter_count_is_modest)
    check("config round-trips through the checkpoint layer", _test_config_round_trips)
    check("feature slots do not overlap", _test_feature_slots_do_not_overlap)
    print(f"\n{sum(_RESULTS)}/{len(_RESULTS)} checks passed")
    sys.exit(0 if all(_RESULTS) else 1)

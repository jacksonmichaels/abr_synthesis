"""
Checks for ``models.SetFusionNet`` — the locus-keyed set-fusion architecture.

These are pure unit tests on synthetic tensors (no data, no training); they
finish in seconds on CPU. Run:

    python tests/test_setfusion.py

What is actually being asserted, beyond "it runs":

  * the encoder is genuinely SHARED per modality (one parameter set for all 12
    promoter windows, not 12), which is the claim that makes a varying number
    of regulatory blocks cost nothing,
  * the encoder is LENGTH-AGNOSTIC (inhA 269 and katG 432 hit the same weights),
  * block COUNT and ORDER are free — permuting the block list permutes the
    attention map but leaves the logits unchanged, since fusion is over a set,
  * locus identity actually reaches the model: two blocks keyed to the same
    locus get the same locus embedding, and swapping which promoter is keyed to
    which gene changes the output (the "shuffle test" in miniature — if it did
    NOT change, the pairing would be decorative).
"""
import sys
import traceback
import warnings
from pathlib import Path

# this file lives in tests/; put the project root on the path so the imports
# below resolve when run directly (python tests/test_setfusion.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from models import SetFusionNet, parse_block_key  # noqa: E402

_RESULTS = []

# an isoniazid-shaped block set: 2 coding loci, 12 promoter windows. The count
# mismatch that positional layouts cannot express is the point of the fixture.
KEYS = ([("dna", g) for g in ("inhA", "katG")]
        + [("protein", g) for g in ("inhA", "katG")]
        + [("biophysical", g) for g in ("inhA", "katG")]
        + [("regulatory", r) for r in ("inhA", "katG", "ahpC", "dnaA", "mshA",
                                       "hadA", "ndh", "glpK")])
SPECS = ([(5, 1710), (5, 2223)]          # dna: nucleotides per locus
         + [(20, 269), (20, 432)]        # protein: differing lengths on purpose
         + [(3, 269), (3, 432)]          # biophysical
         + [(5, w) for w in (879, 545, 351, 245, 300, 275, 420, 300)])


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
        _RESULTS.append(True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        _RESULTS.append(False)


def _net(**kw):
    torch.manual_seed(0)
    return SetFusionNet(KEYS, SPECS, **kw)


def _blocks(batch=2, specs=SPECS, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(batch, c, l, generator=g) for c, l in specs]


# ---------------------------------------------------------------------------
# name parsing: the only coupling to the dataset layer, and it must handle the
# merged per-modality block ('dna') as well as the per-locus one ('dna:katG')
# ---------------------------------------------------------------------------
def _test_key_parsing():
    assert parse_block_key("protein:katG") == ("protein", "katG")
    assert parse_block_key("regulatory:Rv1258c") == ("regulatory", "Rv1258c")
    assert parse_block_key("dna") == ("dna", None), "merged block -> no locus"


# ---------------------------------------------------------------------------
# forward contract: same as MultiModalNet (list of blocks -> (B, n_drugs))
# ---------------------------------------------------------------------------
def _test_forward_shapes():
    for n_drugs in (1, 11):
        model = _net(n_drugs=n_drugs)
        out = model(_blocks(batch=3))
        assert out.shape == (3, n_drugs), f"{out.shape} != (3, {n_drugs})"
        assert torch.isfinite(out).all()


def _test_drug_names_set_width():
    drugs = ["ISONIAZID", "RIFAMPICIN", "ETHAMBUTOL"]
    model = _net(drug_names=drugs)
    assert model.drug_names == drugs
    assert model(_blocks()).shape[1] == len(drugs)


def _test_out_bias_init():
    model = _net(out_bias=1.855)
    assert torch.allclose(model.fc_out.bias, torch.tensor([1.855])), model.fc_out.bias


# ---------------------------------------------------------------------------
# the weight-sharing claim: ONE encoder per modality, not one per block
# ---------------------------------------------------------------------------
def _test_encoders_shared_per_modality():
    model = _net()
    assert set(model.encoders) == {"dna", "protein", "biophysical", "regulatory"}, \
        list(model.encoders)
    # 8 regulatory blocks, one regulatory encoder -> the 9th costs no weights
    grew = SetFusionNet(KEYS + [("regulatory", "eis")], SPECS + [(5, 400)])
    assert set(grew.encoders) == set(model.encoders)
    n_before = sum(p.numel() for p in model.encoders.parameters())
    n_after = sum(p.numel() for p in grew.encoders.parameters())
    assert n_before == n_after, f"encoder params grew with block count: {n_before}->{n_after}"


def _test_encoder_is_length_agnostic():
    """The same weights must accept any L — that is what lets inhA and katG,
    or a new promoter window of unseen width, share one encoder."""
    enc = _net().encoders["protein"]
    outs = [enc(torch.rand(2, 20, l)) for l in (269, 432, 17)]
    for o in outs:
        assert o.shape == outs[0].shape == (2, 128), o.shape


def _test_mixed_channels_rejected():
    try:
        SetFusionNet([("dna", "a"), ("dna", "b")], [(5, 100), (4, 100)])
    except ValueError as e:
        assert "channel" in str(e)
    else:
        raise AssertionError("one shared encoder cannot span 5ch and 4ch blocks")


# ---------------------------------------------------------------------------
# the set claim: count and order of blocks are free
# ---------------------------------------------------------------------------
def _test_permutation_invariance():
    """Fusion is over a SET, so reordering the blocks (with their keys) must not
    move the logits. Positional architectures fail this by construction."""
    model = _net().eval()
    xs = _blocks()
    perm = [7, 0, 3, 11, 2, 9, 1, 5, 13, 4, 8, 12, 6, 10]
    with torch.no_grad():
        a = model(xs)
        b = model([xs[i] for i in perm], keys=[KEYS[i] for i in perm])
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def _test_subset_of_blocks_runs():
    """A drug with fewer promoter windows is just a shorter token set — no
    rebuild, no zero-padding, no reindexing."""
    model = _net(n_drugs=2).eval()
    sub = [0, 2, 4, 6, 7]
    out = model([_blocks()[i] for i in sub], keys=[KEYS[i] for i in sub])
    assert out.shape == (2, 2) and torch.isfinite(out).all()


def _test_unknown_locus_is_rejected():
    model = _net()
    try:
        model(_blocks()[:1], keys=[("regulatory", "not_a_gene")])
    except ValueError as e:
        assert "unknown locus" in str(e)
    else:
        raise AssertionError("an unseen locus has no embedding; must not pass silently")


# ---------------------------------------------------------------------------
# the pairing claim: locus identity reaches the model and changes its output
# ---------------------------------------------------------------------------
def _test_same_locus_shares_embedding():
    """dna:katG and regulatory:katG must land on ONE locus vector — that shared
    key is what attention can match on."""
    model = _net()
    ids = model._default_ids
    dna_katg = KEYS.index(("dna", "katG"))
    reg_katg = KEYS.index(("regulatory", "katG"))
    assert int(ids[dna_katg, 1]) == int(ids[reg_katg, 1]), "same gene, same locus row"
    assert int(ids[dna_katg, 0]) != int(ids[reg_katg, 0]), "different modality row"
    # ...and the vocabulary holds each gene once, not once per block
    assert len(model.locus_vocab) == len({l for _, l in KEYS}) + 1, model.locus_vocab


def _test_shuffling_locus_keys_changes_output():
    """The shuffle test in miniature: hand the model the SAME tensors but swap
    which promoter is keyed to which gene. If the logits were unchanged, the
    locus keying would be decorative."""
    model = _net().eval()
    xs = _blocks()
    swapped = list(KEYS)
    i, j = KEYS.index(("regulatory", "inhA")), KEYS.index(("regulatory", "katG"))
    swapped[i], swapped[j] = ("regulatory", "katG"), ("regulatory", "inhA")
    with torch.no_grad():
        a, b = model(xs), model(xs, keys=swapped)
    assert not torch.allclose(a, b, atol=1e-6), "locus identity had no effect"


def _test_merged_blocks_warn():
    """Merged per-modality blocks ('dna', 'protein', ...) carry no locus, so the
    keying degenerates — the model must say so rather than pretend."""
    keys = [parse_block_key(n) for n in ("dna", "protein", "biophysical", "regulatory")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SetFusionNet(keys, [(5, 3398), (20, 701), (3, 701), (5, 3815)])
    assert any("no-op" in str(w.message) for w in caught), [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# attention read-out: which locus did each drug look at?
# ---------------------------------------------------------------------------
def _test_attention_readout():
    model = _net(drug_names=["ISONIAZID", "RIFAMPICIN"]).eval()
    with torch.no_grad():
        logits, attn = model(_blocks(batch=4), return_attn=True)
    assert logits.shape == (4, 2)
    assert attn.shape == (4, 2, len(KEYS)), attn.shape       # (B, n_drugs, n_blocks)
    assert torch.allclose(attn.sum(-1), torch.ones(4, 2), atol=1e-5), "not a distribution"


# ---------------------------------------------------------------------------
# it must be trainable, and cheaper than the flatten-concat head it replaces
# ---------------------------------------------------------------------------
def _test_backward_reaches_every_parameter():
    model = _net(n_drugs=3)
    model(_blocks(batch=2)).sum().backward()
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not dead, f"no/!finite gradient for: {dead[:5]}"


def _test_parameter_count_is_modest():
    """The flatten-concat head is the parameter sink it replaces: MultiModalNet's
    fc1 alone is sum(out_features) x 256, which on these specs runs to millions.
    Pooled tokens keep the whole net well under that."""
    from models import MultiModalNet
    n_set = sum(p.numel() for p in _net().parameters())
    n_late = sum(p.numel() for p in MultiModalNet(SPECS).parameters())
    assert n_set < n_late, f"set-fusion {n_set:,} !< late-fusion {n_late:,}"
    print(f"      set-fusion {n_set:,} params vs late-fusion {n_late:,}")


# ---------------------------------------------------------------------------
# capacity knobs (results/experiments/setfusion_scaling). The defaults must stay
# EXACTLY what produced full_run / full_run_v2, or the sweep has no control.
# ---------------------------------------------------------------------------
def _test_defaults_are_the_recorded_configuration():
    from models import SETFUSION_DEFAULTS
    assert SETFUSION_DEFAULTS == {
        "d_model": 128, "nhead": 4, "layers": 2, "dim_ff": 256, "dropout": 0.1,
        "enc_width": 64, "enc_out_channels": 32, "enc_depth": 1, "bins": 4}, \
        f"defaults drifted from the full_run configuration: {SETFUSION_DEFAULTS}"
    # constructing with them explicitly must equal constructing with none
    n_implicit = sum(p.numel() for p in _net().parameters())
    n_explicit = sum(p.numel() for p in _net(**SETFUSION_DEFAULTS).parameters())
    assert n_implicit == n_explicit, f"{n_implicit:,} vs {n_explicit:,}"


def _test_depth_one_encoder_is_unchanged():
    """depth=1 must add no state_dict keys, or every saved setfusion checkpoint
    stops loading."""
    from models import SharedBlockEncoder
    enc = SharedBlockEncoder(5)
    assert not any("extra_stages" in k for k in enc.state_dict()), \
        "depth=1 leaked parameters into the state_dict"
    assert len(SharedBlockEncoder(5, depth=3).state_dict()) > len(enc.state_dict())
    for depth in (1, 2, 3):
        out = SharedBlockEncoder(5, depth=depth)(torch.randn(2, 5, 300))
        assert out.shape == (2, 128), (depth, out.shape)
    try:
        SharedBlockEncoder(5, depth=0)
    except ValueError:
        pass
    else:
        raise AssertionError("depth=0 should be rejected")


def _test_each_capacity_knob_changes_the_model():
    base = sum(p.numel() for p in _net(n_drugs=4).parameters())
    for kw in ({"d_model": 256}, {"layers": 4}, {"dim_ff": 1024}, {"hidden": 512},
               {"enc_width": 128}, {"enc_out_channels": 64}, {"enc_depth": 2},
               {"bins": 16}, {"per_drug_hidden": 64}):
        model = _net(n_drugs=4, **kw)
        n = sum(p.numel() for p in model.parameters())
        assert n > base, f"{kw} did not grow the model ({n:,} vs {base:,})"
        assert model(_blocks(batch=2)).shape == (2, 4), kw


def _test_d_model_must_divide_nhead():
    try:
        _net(d_model=100, nhead=8)
    except ValueError as e:
        assert "divisible" in str(e), e
    else:
        raise AssertionError("d_model=100 with nhead=8 should be rejected")


def _test_per_drug_head_is_per_drug():
    """With per-drug branches on, drug j's logit must depend only on branch j —
    otherwise the axis-D arms are measuring nothing."""
    model = _net(n_drugs=3, per_drug_hidden=8).eval()
    xs = _blocks(batch=2)
    with torch.no_grad():
        before = model(xs)
        for p in model.drug_hidden[1].parameters():   # perturb ONLY drug 1
            p.add_(1.0)
        after = model(xs)
    assert torch.equal(before[:, 0], after[:, 0]), "drug 0 moved with drug 1's branch"
    assert torch.equal(before[:, 2], after[:, 2]), "drug 2 moved with drug 1's branch"
    assert not torch.equal(before[:, 1], after[:, 1]), "drug 1's own branch did nothing"
    # single-drug models have one output and nothing to separate: knob is a no-op
    assert _net(n_drugs=1, per_drug_hidden=64).fc_out is not None


def _test_out_bias_reaches_per_drug_heads():
    model = _net(n_drugs=3, per_drug_hidden=8, out_bias=-1.5)
    for branch in model.drug_out:
        assert torch.allclose(branch.bias, torch.tensor([-1.5])), branch.bias


if __name__ == "__main__":
    check("block name -> (modality, locus)", _test_key_parsing)
    check("forward returns (B, n_drugs)", _test_forward_shapes)
    check("drug_names sets the output width", _test_drug_names_set_width)
    check("out_bias initialises the output bias", _test_out_bias_init)
    check("one encoder per modality, shared across loci", _test_encoders_shared_per_modality)
    check("shared encoder is length-agnostic", _test_encoder_is_length_agnostic)
    check("mixed channel counts in a modality rejected", _test_mixed_channels_rejected)
    check("block order does not change the logits", _test_permutation_invariance)
    check("a subset of blocks still runs", _test_subset_of_blocks_runs)
    check("unknown locus rejected", _test_unknown_locus_is_rejected)
    check("same gene -> same locus embedding", _test_same_locus_shares_embedding)
    check("shuffling locus keys changes the output", _test_shuffling_locus_keys_changes_output)
    check("merged per-modality blocks warn", _test_merged_blocks_warn)
    check("attention read-out is (B, n_drugs, n_blocks)", _test_attention_readout)
    check("backward reaches every parameter", _test_backward_reaches_every_parameter)
    check("parameter count below late fusion", _test_parameter_count_is_modest)
    check("defaults are the full_run configuration", _test_defaults_are_the_recorded_configuration)
    check("depth=1 encoder is byte-identical", _test_depth_one_encoder_is_unchanged)
    check("every capacity knob grows the model", _test_each_capacity_knob_changes_the_model)
    check("d_model must divide nhead", _test_d_model_must_divide_nhead)
    check("per-drug head is genuinely per-drug", _test_per_drug_head_is_per_drug)
    check("out_bias reaches per-drug heads", _test_out_bias_reaches_per_drug_heads)
    print(f"\n{sum(_RESULTS)}/{len(_RESULTS)} checks passed")
    sys.exit(0 if all(_RESULTS) else 1)

"""
Checks for ``models.CisFusionNet`` — promoter concatenated onto its own CDS.

Pure unit tests on synthetic tensors (no data, no training), seconds on CPU:

    python tests/test_cisfusion.py

Where SetFusionNet makes the promoter/CDS pairing *learnable*, this model makes
it *structural*, so the assertions are about the layout it builds rather than
about what it learns:

  * a paired locus becomes ONE branch of width L_promoter (+ spacer) + L_cds,
    with the promoter first — transcription order, so a conv kernel at the
    junction straddles both,
  * the 6th channel marks which segment each column came from (0 promoter,
    1 CDS), and promoter / CDS / spacer columns stay mutually distinguishable,
  * the three unit types all survive: paired, promoter-only (a WHO window with
    no CDS loaded — the common case), CDS-only,
  * protein and biophysical blocks pass through untouched, since a 20ch or 3ch
    block cannot be spliced onto a nucleotide axis,
  * which promoter is glued to which CDS changes the output.
"""
import sys
import traceback
import warnings
from pathlib import Path

# this file lives in tests/; put the project root on the path so the imports
# below resolve when run directly (python tests/test_cisfusion.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from models import CisFusionNet, parse_block_key  # noqa: E402

_RESULTS = []

# isoniazid-shaped: 2 coding loci, 4 promoter windows of which only inhA/katG
# have a CDS. ahpC and mshA are orphans — WHO lists them, DRUG_TO_LOCI does not.
KEYS = [("dna", "inhA"), ("dna", "katG"),
        ("protein", "inhA"), ("protein", "katG"),
        ("biophysical", "inhA"), ("biophysical", "katG"),
        ("regulatory", "inhA"), ("regulatory", "katG"),
        ("regulatory", "ahpC"), ("regulatory", "mshA")]
SPECS = [(5, 1710), (5, 2223),
         (20, 269), (20, 432),
         (3, 269), (3, 432),
         (5, 879), (5, 844),
         (5, 351), (5, 300)]
LEN = dict(zip(KEYS, [l for _c, l in SPECS]))


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
    return CisFusionNet(KEYS, SPECS, **kw)


def _blocks(batch=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(batch, c, l, generator=g) for c, l in SPECS]


def _unit(model, name):
    return model.unit_names.index(name)


# ---------------------------------------------------------------------------
# the regrouping: blocks in, cis-units out
# ---------------------------------------------------------------------------
def _test_units_and_names():
    model = _net()
    assert model.unit_names == [
        "cis:inhA", "cis:katG",                       # paired, at the dna slots
        "protein:inhA", "protein:katG",
        "biophysical:inhA", "biophysical:katG",
        "regulatory:ahpC", "regulatory:mshA",         # orphan windows
    ], model.unit_names
    # 10 blocks -> 8 branches: the two regulatory blocks with a CDS were absorbed
    assert len(model.encoders) == 8
    assert model.unit_kinds.count("cis") == 2


def _test_paired_spec_is_promoter_plus_cds():
    model = _net()
    for gene in ("inhA", "katG"):
        c, l = model.unit_specs[_unit(model, f"cis:{gene}")]
        assert c == 6, f"5 nucleotide channels + 1 segment marker, got {c}"
        assert l == LEN[("regulatory", gene)] + LEN[("dna", gene)], l


def _test_spacer_widens_only_paired_units():
    plain, spaced = _net(), _net(spacer=16)
    i, j = _unit(plain, "cis:katG"), _unit(plain, "regulatory:ahpC")
    assert spaced.unit_specs[i][1] == plain.unit_specs[i][1] + 16
    assert spaced.unit_specs[j][1] == plain.unit_specs[j][1], "orphan needs no gap"


def _test_orphans_keep_their_own_branch():
    """10 of INH's 12 real windows have no CDS. They must not be dropped, and
    must not be silently glued to somebody else's gene."""
    model = _net()
    for gene in ("ahpC", "mshA"):
        c, l = model.unit_specs[_unit(model, f"regulatory:{gene}")]
        assert (c, l) == (6, LEN[("regulatory", gene)])


def _test_cds_only_locus():
    keys = [("dna", "pncA"), ("regulatory", "clpC1")]
    model = CisFusionNet(keys, [(5, 700), (5, 300)])
    assert model.unit_names == ["dna:pncA", "regulatory:clpC1"], model.unit_names
    xs = [torch.rand(2, 5, 700), torch.rand(2, 5, 300)]
    built = dict(model.cis_inputs(xs))
    assert torch.equal(built["dna:pncA"][:, 5], torch.ones(2, 700)), "CDS-only -> flag 1"
    assert torch.equal(built["regulatory:clpC1"][:, 5], torch.zeros(2, 300)), "promoter -> 0"


def _test_non_nucleotide_blocks_pass_through():
    model = _net()
    xs = _blocks()
    built = dict(model.cis_inputs(xs))
    assert torch.equal(built["protein:katG"], xs[KEYS.index(("protein", "katG"))])
    assert torch.equal(built["biophysical:inhA"], xs[KEYS.index(("biophysical", "inhA"))])


def _test_mixed_channels_rejected():
    try:
        CisFusionNet([("dna", "katG")], [(4, 100)])
    except ValueError as e:
        assert "nucleotide alphabet" in str(e)
    else:
        raise AssertionError("a 4-channel 'dna' block is not the ACTG- alphabet")


# ---------------------------------------------------------------------------
# the concatenation itself: order, contents, segment marker
# ---------------------------------------------------------------------------
def _test_promoter_comes_first_and_content_is_preserved():
    """Transcription order, and neither segment is altered on the way in."""
    model = _net()
    xs = _blocks()
    fused = dict(model.cis_inputs(xs))["cis:katG"]
    reg, dna = xs[KEYS.index(("regulatory", "katG"))], xs[KEYS.index(("dna", "katG"))]
    n = reg.shape[-1]
    assert torch.equal(fused[:, :5, :n], reg), "promoter must lead, verbatim"
    assert torch.equal(fused[:, :5, n:], dna), "CDS must follow, verbatim"


def _test_segment_channel_marks_the_junction():
    model = _net()
    fused = dict(model.cis_inputs(_blocks()))["cis:inhA"]
    n = LEN[("regulatory", "inhA")]
    seg = fused[:, 5]
    assert torch.equal(seg[:, :n], torch.zeros_like(seg[:, :n])), "promoter flag 0"
    assert torch.equal(seg[:, n:], torch.ones_like(seg[:, n:])), "CDS flag 1"


def _test_spacer_columns_are_distinguishable():
    """promoter (one-hot, flag 0) / spacer (all zero) / CDS (one-hot, flag 1)
    must be three different column patterns, or 'gap' reads as 'promoter'."""
    model = _net(spacer=8)
    fused = dict(model.cis_inputs(_blocks()))["cis:katG"]
    n = LEN[("regulatory", "katG")]
    gap = fused[:, :, n:n + 8]
    assert torch.equal(gap, torch.zeros_like(gap)), "spacer is all-zero incl. the flag"
    assert fused[:, :5, :n].abs().sum() > 0 and fused[:, 5, n + 8:].min() == 1


# ---------------------------------------------------------------------------
# model contract
# ---------------------------------------------------------------------------
def _test_forward_shapes():
    for n_drugs in (1, 11):
        out = _net(n_drugs=n_drugs)(_blocks(batch=3))
        assert out.shape == (3, n_drugs), out.shape
        assert torch.isfinite(out).all()


def _test_drug_names_and_out_bias():
    drugs = ["ISONIAZID", "RIFAMPICIN"]
    model = _net(drug_names=drugs, out_bias=1.855)
    assert model.drug_names == drugs
    assert model(_blocks()).shape[1] == 2
    assert torch.allclose(model.head.fc_out.bias, torch.full((2,), 1.855))


def _test_per_kind_encoder_choice():
    model = _net(branch_models={"protein": "transformer"})
    kinds = dict(zip(model.unit_names, model.encoder_types))
    assert kinds["protein:katG"] == "transformer"
    assert kinds["cis:katG"] == "cnn" and kinds["regulatory:ahpC"] == "cnn"


def _test_backward_reaches_every_parameter():
    model = _net(n_drugs=3)
    model(_blocks()).sum().backward()
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not dead, f"no/!finite gradient for: {dead[:5]}"


def _test_pairing_actually_matters():
    """Swap which promoter is glued to which gene; the logits must move. Here
    that is structural rather than learned, but it is the same shuffle test."""
    model = _net().eval()
    xs = _blocks()
    r_inh, r_kat = KEYS.index(("regulatory", "inhA")), KEYS.index(("regulatory", "katG"))
    swapped = list(xs)
    # same widths are not guaranteed, so swap the model's view via keys instead
    keys = list(KEYS)
    keys[r_inh], keys[r_kat] = ("regulatory", "katG"), ("regulatory", "inhA")
    other = CisFusionNet(keys, SPECS)
    other.load_state_dict(model.state_dict())
    with torch.no_grad():
        a, b = model(xs), other.eval()(swapped)
    assert not torch.allclose(a, b, atol=1e-6), "promoter/CDS assignment had no effect"


def _test_merged_blocks_warn():
    """Merged per-modality blocks have no gene to pair on — say so."""
    keys = [parse_block_key(n) for n in ("dna", "protein", "regulatory")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = CisFusionNet(keys, [(5, 3398), (20, 701), (5, 3815)])
    assert any("nothing was cis-paired" in str(w.message) for w in caught)
    assert model.unit_names == ["dna", "protein", "regulatory"], model.unit_names
    assert model(  # still a working model, just an unpaired one
        [torch.rand(2, 5, 3398), torch.rand(2, 20, 701), torch.rand(2, 5, 3815)]
    ).shape == (2, 1)


if __name__ == "__main__":
    check("blocks regroup into cis-units", _test_units_and_names)
    check("paired unit = promoter + CDS, 6 channels", _test_paired_spec_is_promoter_plus_cds)
    check("spacer widens only paired units", _test_spacer_widens_only_paired_units)
    check("orphan promoters keep a branch", _test_orphans_keep_their_own_branch)
    check("CDS-only locus flagged as CDS", _test_cds_only_locus)
    check("protein/biophysical pass through", _test_non_nucleotide_blocks_pass_through)
    check("non-nucleotide channel count rejected", _test_mixed_channels_rejected)
    check("promoter first, contents verbatim", _test_promoter_comes_first_and_content_is_preserved)
    check("segment channel marks the junction", _test_segment_channel_marks_the_junction)
    check("spacer distinguishable from both segments", _test_spacer_columns_are_distinguishable)
    check("forward returns (B, n_drugs)", _test_forward_shapes)
    check("drug_names + out_bias", _test_drug_names_and_out_bias)
    check("per-kind encoder choice", _test_per_kind_encoder_choice)
    check("backward reaches every parameter", _test_backward_reaches_every_parameter)
    check("promoter/CDS assignment changes output", _test_pairing_actually_matters)
    check("merged per-modality blocks warn", _test_merged_blocks_warn)
    print(f"\n{sum(_RESULTS)}/{len(_RESULTS)} checks passed")
    sys.exit(0 if all(_RESULTS) else 1)

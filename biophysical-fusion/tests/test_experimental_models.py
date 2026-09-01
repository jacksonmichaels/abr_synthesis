"""
Checks for ``models.experimental_models`` — the six variant-set aggregators.

Pure unit tests on synthetic tensors (no data, no training); seconds on CPU:

    python tests/test_experimental_models.py

These six share one tokenizer and differ ONLY in how the variant set is
aggregated, so the tests are organised the same way: a block of shared
substrate checks, then one block per model asserting the property that model
exists to have. Those properties are the design, so if one of these fails the
model is not the thing its docstring claims.

  * `catalogue`  CANNOT score a variant it never saw (the memorisation control)
  * `additive`   CAN, from features — and its contributions sum to the logit
  * `noisyor`    is monotone AND pointed at the right class: under this
                 project's R=0/S=1 encoding a variant can only LOWER the logit
  * `gatedpool`  gates do NOT sum to 1 (that is the whole point vs softmax)
  * `deepsets`   is permutation-invariant and uses no attention at all
  * `fm`         has a genuine pairwise term: two variants together != the sum
"""
import sys
import traceback
import warnings
from pathlib import Path

# this file lives in tests/; put the project root on the path so the imports
# below resolve when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from models import (EXPERIMENTAL_DEFAULTS, EXPERIMENTAL_MODELS,  # noqa: E402
                    AdditiveVariantNet, CatalogueNet, make_experimental)
from models.experimental_models import C_TOK, VariantSet  # noqa: E402
from training.checkpoint import (build_model_from_config,  # noqa: E402
                                 model_config)

_RESULTS = []

# an isoniazid-shaped block set at REAL sizes
KEYS = [("dna", "inhA"), ("dna", "katG"),
        ("protein", "inhA"), ("protein", "katG"),
        ("biophysical", "inhA"), ("biophysical", "katG"),
        ("regulatory", "inhA"), ("regulatory", "katG")]
SPECS = [(5, 910), (5, 2488), (20, 303), (20, 829),
         (3, 303), (3, 829), (5, 879), (5, 642)]
KATG_DNA, KATG_PROT = 1, 3


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
        _RESULTS.append(True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        _RESULTS.append(False)


def wt(b=4):
    """A wild-type cohort: delta encoding of an isolate identical to H37Rv."""
    return [torch.zeros(b, c, length) for c, length in SPECS]


def with_variant(xs, block, column, channel=0, value=1.0, row=slice(None)):
    xs = [x.clone() for x in xs]
    xs[block][row, channel, column] = value
    return xs


def net(name, seed=0, **kw):
    torch.manual_seed(seed)
    m = make_experimental(name, KEYS, SPECS, n_drugs=1, hidden=64,
                          **{**EXPERIMENTAL_DEFAULTS, **kw})
    m.eval()                 # dropout off, so two forwards of one input agree
    return m


# --- shared substrate -------------------------------------------------------

def _test_tokenizer_is_flat_and_wild_type_is_empty():
    tok = VariantSet(KEYS, SPECS, max_variants=16)
    v = tok(wt(3))
    assert v["feat"].shape == (3, 8 * 16, C_TOK), v["feat"].shape
    assert not v["valid"].any(), "a wild-type cohort produced variant tokens"
    assert v["n_occ"].sum() == 0


def _test_variant_ids_are_exact_and_collision_free():
    """`catalogue` is only a fair control if its ids do not collide."""
    tok = VariantSet(KEYS, SPECS, max_variants=16)
    assert tok.vocab_size == sum(c * length for c, length in SPECS), tok.vocab_size
    seen = set()
    for b, (c, length) in enumerate(SPECS):
        for col in (0, length // 2, length - 1):
            for ch in (0, c - 1):
                v = tok(with_variant(wt(1), b, col, channel=ch))
                ids = v["vid"][v["valid"]].tolist()
                assert len(ids) == 1, (b, col, ch, ids)
                assert ids[0] not in seen, f"vid collision at block {b} col {col} ch {ch}"
                seen.add(ids[0])


def _test_uncovered_block_is_flagged():
    tok = VariantSet(KEYS, SPECS, max_variants=16)
    xs = wt(2)
    xs[KATG_DNA][0, 4, :] = 1.0                 # katG entirely gap, isolate 0
    v = tok(xs)
    assert v["uncovered"][0, KATG_DNA] == 1.0 and v["uncovered"][1, KATG_DNA] == 0.0
    assert v["n_occ"][0, KATG_DNA] == SPECS[KATG_DNA][1]


def _test_dense_input_warns():
    tok = VariantSet(KEYS, SPECS, max_variants=16)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tok([torch.ones(2, c, length) for c, length in SPECS])
    assert any("--delta" in str(x.message) for x in w), [str(x.message) for x in w]


def _test_every_model_runs_single_and_joint():
    xs = with_variant(wt(4), KATG_DNA, 944, channel=2)
    for name in EXPERIMENTAL_MODELS:
        for n_drugs, names in ((1, None), (3, ["A", "B", "C"])):
            torch.manual_seed(0)
            m = make_experimental(name, KEYS, SPECS, n_drugs=n_drugs,
                                  drug_names=names, hidden=64,
                                  **EXPERIMENTAL_DEFAULTS)
            out = m(xs)
            assert out.shape == (4, n_drugs), (name, out.shape)
            out.sum().backward()
            dead = [p for p, t in m.named_parameters() if t.grad is None]
            assert not dead, (name, dead)


def _test_out_bias_is_the_wild_type_prior():
    """No deviations from H37Rv -> the model's learned base rate, exactly."""
    for name in ("catalogue", "additive", "fm"):
        m = net(name, out_bias=-1.25)
        assert torch.allclose(m(wt(2)), torch.full((2, 1), -1.25), atol=1e-5), name


# --- 1. catalogue: memorisation, and its limit ------------------------------

def _test_catalogue_cannot_score_an_unseen_variant():
    """The control the family is built around: a zero-initialised per-identity
    table contributes exactly nothing for a variant absent from training."""
    m = net("catalogue")
    base = m(wt(1))
    with torch.no_grad():                        # "train" on ONE variant
        seen = m.tok(with_variant(wt(1), KATG_DNA, 944, channel=2))
        m.w.weight[seen["vid"][seen["valid"]][0]] = 3.0
    hit = m(with_variant(wt(1), KATG_DNA, 944, channel=2))
    miss = m(with_variant(wt(1), KATG_DNA, 945, channel=2))
    assert torch.allclose(hit - base, torch.tensor(3.0), atol=1e-5), hit - base
    assert torch.allclose(miss, base, atol=1e-6), \
        "catalogue scored a variant it never saw — it is not a memorisation control"


def _test_catalogue_entries_readable():
    m = net("catalogue")
    with torch.no_grad():
        m.w.weight[123] = 2.0
    top = m.catalogue_entries(top=3)
    assert top[0][0] == 123 and abs(top[0][1] - 2.0) < 1e-6, top


# --- 2. additive: featurisation generalises, and is its own attribution -----

def _test_additive_generalises_to_an_unseen_variant():
    """The counterpart measurement: the weight comes from the variant's
    FEATURES, so a substitution absent from training still gets one."""
    m = net("additive")
    with torch.no_grad():                        # break the zero init
        m.mlp[-1].weight.normal_(0, 0.5)
    base = m(wt(1))
    a = m(with_variant(wt(1), KATG_PROT, 315, channel=7))
    b = m(with_variant(wt(1), KATG_PROT, 316, channel=7))
    assert not torch.allclose(a, base, atol=1e-6), "no effect from a variant"
    assert not torch.allclose(a, b, atol=1e-6), \
        "two different unseen residues scored identically — features are not reaching w"


def _test_additive_contributions_sum_to_the_logit():
    m = net("additive")
    with torch.no_grad():
        m.mlp[-1].weight.normal_(0, 0.5)
    xs = with_variant(with_variant(wt(3), KATG_DNA, 944, channel=2),
                      KATG_PROT, 315, channel=7)
    got = m.contributions(xs).sum(1).squeeze(-1)
    want = (m(xs) - m.bias).squeeze(-1)
    assert torch.allclose(got, want, atol=1e-5), (got, want)


def _test_residual_catalogue_adds_the_identity_table():
    plain = net("additive")
    hybrid = net("additive", residual_catalogue=True)
    assert hybrid.w is not None and plain.w is None
    assert sum(p.numel() for p in hybrid.parameters()) > \
        sum(p.numel() for p in plain.parameters())


# --- 3. noisyor: absolute and monotone --------------------------------------

def _test_noisyor_is_monotone_in_evidence():
    """Every factor is in (0,1), so the product — P(SUSCEPTIBLE) — can only fall
    as variants are added. That is the 'susceptible unless something confers
    resistance' prior, and it is also the model's main restriction: it cannot
    learn a protective variant.

    The DIRECTION is the point of this test, not just the monotonicity. This
    project encodes R=0 / S=1, so the logit is for the susceptible class and a
    variant must push it DOWN. The first version of the model had the sign the
    other way and scored macro CV 0.4956 — below chance — because a monotone
    aggregator pointed at the wrong class is structurally anti-predictive."""
    m = net("noisyor")
    with torch.no_grad():
        m.mlp[-1].weight.normal_(0, 1.0)
    prev = m(wt(1))
    cols = [400, 900, 1400, 1900]
    xs = wt(1)
    for i, col in enumerate(cols):
        xs = with_variant(xs, KATG_DNA, col, channel=i % 4)
        now = m(xs)
        assert (now <= prev + 1e-6).all(), \
            f"adding a variant raised the susceptible logit: {float(prev)} -> {float(now)}"
        prev = now
    assert (prev < m(wt(1)) - 1e-3).all(), "four variants moved the logit by nothing"


def _test_noisyor_saturates_rather_than_accumulates():
    """Two independent resistance mutations give 'resistant', not twice the
    logit — the thing a plain sum gets wrong when an isolate carries several."""
    m = net("noisyor")
    with torch.no_grad():
        m.mlp[-1].bias.fill_(2.0)                # every variant strongly resistant
    base = float(m(wt(1)))
    one = float(m(with_variant(wt(1), KATG_DNA, 900)))
    two = float(m(with_variant(with_variant(wt(1), KATG_DNA, 900),
                               KATG_DNA, 1400, channel=1)))
    assert two < one < base                      # falling: R=0/S=1
    assert (one - two) < (base - one), \
        f"noisy-OR did not saturate: {base:.3f} -> {one:.3f} -> {two:.3f}"


def _test_noisyor_is_finite_at_both_extremes():
    m = net("noisyor")
    with torch.no_grad():
        m.mlp[-1].bias.fill_(20.0)               # p_v -> 1, log P(S) -> -inf
    xs = wt(1)
    for col in range(0, 1600, 100):
        xs = with_variant(xs, KATG_DNA, col)
    out = torch.cat([m(wt(1)), m(xs)])
    assert torch.isfinite(out).all(), out


# --- 4. gatedpool: absolute relevance, not a share of a budget --------------

def _test_gates_do_not_sum_to_one():
    """The whole reason this model exists. Softmax attention must spend weight on
    the neutral tokens; a sigmoid gate can open for one and stay shut for the
    rest, which is what a needle detector needs."""
    m = net("gatedpool")
    with torch.no_grad():
        m.gate.weight.normal_(0, 2.0)
    xs = wt(1)
    for i, col in enumerate((300, 700, 1100, 1500, 1900)):
        xs = with_variant(xs, KATG_DNA, col, channel=i % 4)
    g = m.gates(xs)[0, :, 0]
    total = float(g.sum())
    assert abs(total - 1.0) > 1e-3, f"gates summed to {total:.4f} — that is a softmax"
    assert (g >= 0).all() and (g <= 1).all()


def _test_gatedpool_handles_an_all_wild_type_isolate():
    """The max branch over an empty set must not leak -inf into the head."""
    m = net("gatedpool")
    assert torch.isfinite(m(wt(3))).all()


# --- 5. deepsets: no attention at all ---------------------------------------

def _test_deepsets_is_permutation_invariant():
    """Shuffling the TOKEN axis must not move the logit — sum, max and count are
    all symmetric functions.

    Note what is *not* being claimed. The block LIST order is part of a model's
    identity here (it fixes the locus vocabulary, the per-block uncovered
    weights, and the variant-id offsets), unlike `SetFusionNet`, which is keyed
    and can be handed a reordering at call time. What these aggregators are
    invariant to is the order tokens arrive in, which is what makes them a
    genuine set function over the variants."""
    m = net("deepsets")
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.2)
    xs = with_variant(with_variant(wt(2), KATG_DNA, 944, channel=2),
                      KATG_PROT, 315, channel=7)
    v = m.tok(xs)
    h = m.emb(v)
    tokens = h.shape[1]
    torch.manual_seed(1)
    perm = torch.randperm(tokens)
    v_perm = {k: (t[:, perm] if t.dim() >= 2 and t.shape[1] == tokens else t)
              for k, t in v.items()}
    a = m.aggregate(h, v)
    b = m.aggregate(h[:, perm], v_perm)
    assert torch.allclose(a, b, atol=1e-5), (a, b)


def _test_deepsets_has_no_attention_module():
    m = net("deepsets")
    names = " ".join(n for n, _ in m.named_modules())
    assert "attn" not in names and "gate" not in names, names


# --- 6. fm: a genuine pairwise term -----------------------------------------

def _test_fm_interaction_is_not_additive():
    """logit(a,b) - logit(0) must differ from [logit(a)-logit(0)] +
    [logit(b)-logit(0)]; the gap IS the interaction term."""
    m = net("fm")
    with torch.no_grad():
        m.second.weight.normal_(0, 1.0)
        m.factor.weight.normal_(0, 0.5)
    base = float(m(wt(1)))
    a = float(m(with_variant(wt(1), KATG_DNA, 900))) - base
    b = float(m(with_variant(wt(1), 0, 300))) - base       # a different locus
    ab = float(m(with_variant(with_variant(wt(1), KATG_DNA, 900), 0, 300))) - base
    assert abs(ab - (a + b)) > 1e-4, \
        f"no pairwise term: a={a:.4f} b={b:.4f} ab={ab:.4f}"


def _test_fm_rank_changes_the_model():
    small = sum(p.numel() for p in net("fm", fm_rank=4).parameters())
    big = sum(p.numel() for p in net("fm", fm_rank=32).parameters())
    assert big > small, (small, big)


# --- factory + plumbing -----------------------------------------------------

def _test_factory_rejects_a_changed_foreign_knob():
    """Passing the whole defaults dict is fine; moving a knob that belongs to a
    different member off its default is an error, so a sweep arm cannot quietly
    run as its own control."""
    make_experimental("deepsets", KEYS, SPECS, **EXPERIMENTAL_DEFAULTS)   # fine
    try:
        make_experimental("deepsets", KEYS, SPECS,
                          **{**EXPERIMENTAL_DEFAULTS, "fm_rank": 16})
    except ValueError as e:
        assert "fm_rank" in str(e) and "fm" in str(e), e
    else:
        raise AssertionError("a foreign knob was accepted")


def _test_unknown_model_rejected():
    try:
        make_experimental("nonsense", KEYS, SPECS)
    except ValueError as e:
        assert "unknown experimental model" in str(e)
    else:
        raise AssertionError("an unknown model name was accepted")


def _test_configs_round_trip():
    class _B:
        def __init__(self, key, spec):
            self.name = f"{key[0]}:{key[1]}"
            self.modality, self.locus = key
            self._spec = spec

        def spec(self):
            return self._spec

    blocks = [_B(k, s) for k, s in zip(KEYS, SPECS)]
    for name, knobs in (("fm", {"fm_rank": 16}),
                        ("additive", {"residual_catalogue": True}),
                        ("catalogue", {}), ("noisyor", {"d_model": 64})):
        cfg = model_config(arch=name, blocks=blocks,
                           encoder_types=["cnn"] * len(KEYS),
                           drug_names=["ISONIAZID"], out_bias=None,
                           head={"hidden": 256, "dropout": 0.0,
                                 "per_drug_hidden": 0},
                           experimental=knobs)
        rebuilt = build_model_from_config({"model": cfg})
        assert type(rebuilt) is EXPERIMENTAL_MODELS[name], (name, type(rebuilt))
        if name == "fm":
            assert rebuilt.fm_rank == 16
        if name == "additive":
            assert rebuilt.w is not None
        if name == "noisyor":
            assert rebuilt.d_model == 64


def _test_parameter_counts_are_modest():
    for name in EXPERIMENTAL_MODELS:
        n = sum(p.numel() for p in net(name).parameters())
        assert n < 500_000, f"{name}: {n:,} parameters — bigger than intended"


def _test_merged_blocks_warn():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        make_experimental("deepsets", [("dna", None)], [(5, 3398)])
    assert any("per_modality_branch" in str(x.message) for x in w), \
        [str(x.message) for x in w]


if __name__ == "__main__":
    check("tokenizer is flat; wild type is the empty set",
          _test_tokenizer_is_flat_and_wild_type_is_empty)
    check("variant ids are exact and collision-free",
          _test_variant_ids_are_exact_and_collision_free)
    check("an uncovered block is flagged", _test_uncovered_block_is_flagged)
    check("dense (non-delta) input warns", _test_dense_input_warns)
    check("every model runs single-drug and joint",
          _test_every_model_runs_single_and_joint)
    check("out_bias is the wild-type prior", _test_out_bias_is_the_wild_type_prior)
    check("catalogue cannot score an unseen variant",
          _test_catalogue_cannot_score_an_unseen_variant)
    check("catalogue entries are readable", _test_catalogue_entries_readable)
    check("additive generalises to an unseen variant",
          _test_additive_generalises_to_an_unseen_variant)
    check("additive contributions sum to the logit",
          _test_additive_contributions_sum_to_the_logit)
    check("residual_catalogue adds the identity table",
          _test_residual_catalogue_adds_the_identity_table)
    check("noisyor is monotone in evidence", _test_noisyor_is_monotone_in_evidence)
    check("noisyor saturates rather than accumulates",
          _test_noisyor_saturates_rather_than_accumulates)
    check("noisyor is finite at both extremes",
          _test_noisyor_is_finite_at_both_extremes)
    check("gates do not sum to 1", _test_gates_do_not_sum_to_one)
    check("gatedpool handles an all-wild-type isolate",
          _test_gatedpool_handles_an_all_wild_type_isolate)
    check("deepsets is permutation-invariant", _test_deepsets_is_permutation_invariant)
    check("deepsets has no attention module", _test_deepsets_has_no_attention_module)
    check("fm has a genuine pairwise term", _test_fm_interaction_is_not_additive)
    check("fm_rank changes the model", _test_fm_rank_changes_the_model)
    check("factory rejects a changed foreign knob",
          _test_factory_rejects_a_changed_foreign_knob)
    check("unknown model name rejected", _test_unknown_model_rejected)
    check("configs round-trip through the checkpoint layer", _test_configs_round_trip)
    check("parameter counts are modest", _test_parameter_counts_are_modest)
    check("merged per-modality blocks warn", _test_merged_blocks_warn)
    print(f"\n{sum(_RESULTS)}/{len(_RESULTS)} checks passed")
    sys.exit(0 if all(_RESULTS) else 1)

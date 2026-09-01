"""models.BranchedHead — the Learn-to-Branch head (Luo et al. 2024, subject->drug).

The tests that matter here are not "does it produce a tensor of the right
shape". They are the ones that would have caught the failure the head was first
built with: a routing mechanism that is present, differentiable, and completely
inert, which is exactly what `token_signal` found in setfusion. So the file
asserts on gradient flow into theta, on the hard/soft forward actually
differing, and on the anneal reaching its endpoint.
"""
import sys
import traceback
from pathlib import Path

# this file lives in tests/; put the project root on the path so the imports
# below resolve when run directly (python tests/test_branched.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from models import (BRANCHED_DEFAULTS, BranchedHead, DenseHead,  # noqa: E402
                    MDCNNNet, MultiDrugNet, make_head)
from training.core import (anneal_branch_temperature,  # noqa: E402
                           branch_assignments, branch_aux_loss,
                           masked_weighted_bce)

_RESULTS = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
        _RESULTS.append(True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        _RESULTS.append(False)


def approx(a, b, tol=1e-5):
    assert abs(float(a) - float(b)) <= tol, f"{a} != {b}"

DRUGS = ["A", "B", "C", "D"]


def head(**kw):
    cfg = dict(hidden=32, n_groups=3, per_drug_hidden=8)
    cfg.update(kw)
    return BranchedHead(64, DRUGS, **cfg)


# --------------------------------------------------------------- shape/contract
def test_forward_shape_matches_densehead_contract():
    h = head()
    assert h(torch.randn(7, 64)).shape == (7, len(DRUGS))


def test_make_head_returns_dense_for_single_output():
    """A single-drug run has no tasks to group, so --head branched must be a
    no-op rather than an error — one runner drives both scopes."""
    assert isinstance(make_head(64, out_dim=1, branched={"n_groups": 4}), DenseHead)
    assert isinstance(make_head(64, out_dim=4, drug_names=DRUGS,
                                branched={"n_groups": 4}), BranchedHead)


def test_requires_at_least_two_drugs():
    try:
        BranchedHead(64, ["only"], hidden=32)
    except ValueError as e:
        assert "multi-task" in str(e), e
    else:
        raise AssertionError("a 1-drug BranchedHead should raise")


def test_defaults_are_the_documented_ones():
    """BRANCHED_DEFAULTS is what the runner advertises; drift here silently
    changes what every result JSON claims it ran."""
    h = head(**{k: v for k, v in BRANCHED_DEFAULTS.items()
                if k in ("n_groups", "per_drug_hidden", "tau_start", "tau_end")})
    assert h.n_groups == BRANCHED_DEFAULTS["n_groups"] == 4
    assert h.hard is BRANCHED_DEFAULTS["hard"] is True
    assert h.tau_start == 5.0 and h.tau_end == 0.5


# ------------------------------------------------------------------- annealing
def test_anneal_spans_tau_start_to_tau_end():
    h = head(tau_start=5.0, tau_end=0.5)
    h.anneal(0, 10)
    approx(h.tau, 5.0)
    h.anneal(9, 10)
    approx(h.tau, 0.5)
    h.anneal(5, 10)          # monotone in between
    assert 0.5 < float(h.tau) < 5.0


def test_anneal_is_clamped_and_survives_a_single_epoch_run():
    h = head(tau_start=5.0, tau_end=0.5)
    h.anneal(99, 10)                      # past the end
    approx(h.tau, 0.5)
    h.anneal(0, 1)                        # epochs=1 -> no range to interpolate
    approx(h.tau, 0.5)


def test_trainer_hook_anneals_through_a_whole_net():
    net = MDCNNNet([(5, 200)] * 3, drug_names=DRUGS, hidden=32,
                   branched={"n_groups": 2, "per_drug_hidden": 8})
    anneal_branch_temperature(net, 0, 10)
    approx(net.head.tau, net.head.tau_start)
    anneal_branch_temperature(net, 9, 10)
    approx(net.head.tau, net.head.tau_end)


def test_anneal_hook_is_a_noop_without_a_branched_head():
    net = MultiDrugNet([(5, 200)], DRUGS)          # plain DenseHead
    anneal_branch_temperature(net, 3, 10)          # must not raise
    assert branch_assignments(net) is None
    assert branch_aux_loss(net, torch.full((4, len(DRUGS)), 0.5)) == 0.0


# --------------------------------------------------------------------- routing
def test_hard_routing_gives_one_hot_weights():
    h = head(hard=True)
    h.train()
    w = h.branch_weights()
    assert w.shape == (len(DRUGS), 3)
    assert torch.allclose(w.sum(-1), torch.ones(len(DRUGS)))
    assert torch.allclose(w.max(-1).values, torch.ones(len(DRUGS)))   # exactly one


def test_soft_routing_is_not_one_hot():
    h = head(hard=False, tau_start=5.0)
    h.train()
    w = h.branch_weights()
    assert torch.allclose(w.sum(-1), torch.ones(len(DRUGS)))
    assert w.max().item() < 0.99


def test_eval_routing_is_deterministic():
    """Gumbel noise at eval would make the val metric a coin flip between
    epochs and early stopping would chase it."""
    h = head()
    h.eval()
    a, b = h.branch_weights(), h.branch_weights()
    assert torch.equal(a, b)


def test_eval_matches_the_hard_training_forward():
    h = head(hard=True)
    h.eval()
    assert torch.equal(h.branch_weights().argmax(-1), h.theta.argmax(-1))


# ----------------------------------------------------- the inertness regression
def test_theta_receives_gradient():
    """The bug this file exists for: theta differentiable but never updated."""
    h = head(hard=True)
    h.train()
    h(torch.randn(16, 64)).sum().backward()
    assert h.theta.grad is not None
    assert float(h.theta.grad.abs().sum()) > 0.0


def test_hard_routing_makes_group_gradients_disjoint():
    """Under hard routing a group node must be updated ONLY by the drugs routed
    to it. That disjointness is what breaks the symmetry between group nodes;
    with soft mixing every group sees every drug and they never specialize."""
    torch.manual_seed(0)
    h = head(hard=True, n_groups=2, theta_init_std=0.0)
    with torch.no_grad():                       # force a known split: A,B -> 0
        h.theta.copy_(torch.tensor([[9.0, -9.0], [9.0, -9.0],
                                    [-9.0, 9.0], [-9.0, 9.0]]))
    h.train()
    h.set_tau(0.1)                              # near-deterministic routing
    out = h(torch.randn(32, 64))
    out[:, 0].sum().backward()                  # drug A only
    g0 = sum(float(p.grad.abs().sum()) for p in h.groups[0].parameters()
             if p.grad is not None)
    g1 = sum(float(p.grad.abs().sum()) for p in h.groups[1].parameters()
             if p.grad is not None)
    assert g0 > 0.0, "group A's own node got no gradient from drug A"
    assert g1 == 0.0, "group B was updated by a drug that does not route to it"


def test_theta_init_std_breaks_the_tie():
    torch.manual_seed(0)
    assert float(head(theta_init_std=0.0).theta.abs().max()) == 0.0
    assert float(head(theta_init_std=0.1).theta.abs().max()) > 0.0


# ------------------------------------------------------------ the generic head
def test_aux_logits_cached_on_train_and_cleared_on_eval():
    h = head()
    h.train(); h(torch.randn(5, 64))
    assert h.aux_logits is not None and h.aux_logits.shape == (5, len(DRUGS))
    h.eval(); h(torch.randn(5, 64))
    assert h.aux_logits is None, "a stale aux tensor could be picked up off-step"


def test_branch_aux_loss_scales_with_generic_weight():
    torch.manual_seed(0)
    alpha = torch.full((8, len(DRUGS)), 0.5)
    x = torch.randn(8, 64)
    losses = []
    for gw in (0.0, 0.5):
        torch.manual_seed(0)
        h = head(generic_weight=gw)
        h.train(); h(x)
        losses.append(float(branch_aux_loss(h, alpha)))
    assert losses[0] == 0.0
    assert losses[1] > 0.0


def test_aux_loss_is_masked_like_the_main_loss():
    """alpha=0 means 'this drug has no label for this isolate'. The generic head
    must honour the same mask or it trains on phantom labels."""
    torch.manual_seed(0)
    h = head(generic_weight=1.0)
    h.train(); h(torch.randn(6, 64))
    full = torch.full((6, len(DRUGS)), 0.5)
    masked = full.clone(); masked[:, 2:] = 0.0
    a = float(masked_weighted_bce(h.aux_logits, full))
    b = float(masked_weighted_bce(h.aux_logits, masked))
    assert abs(a - b) > 1e-9, "masking changed nothing — the mask is ignored"


# ------------------------------------------------------------------ reporting
def test_assignments_partition_every_drug_exactly_once():
    h = head(n_groups=3)
    got = branch_assignments(h)
    flat = [d for v in got.values() for d in v]
    assert sorted(flat) == sorted(DRUGS)
    assert len(flat) == len(set(flat))


def test_single_group_puts_everything_together():
    h = head(n_groups=1)
    assert branch_assignments(h) == {0: DRUGS}


# ---------------------------------------------------------------- integration
def test_end_to_end_step_through_mdcnn(n_groups=2):
    torch.manual_seed(0)
    net = MDCNNNet([(5, 300)] * 4, drug_names=DRUGS, hidden=32,
                   branched={"n_groups": n_groups, "per_drug_hidden": 8})
    xs = [torch.randn(6, 5, 300) for _ in range(4)]
    alpha = torch.full((6, len(DRUGS)), 0.5)
    net.train()
    anneal_branch_temperature(net, 0, 5)
    loss = masked_weighted_bce(net(xs), alpha) + branch_aux_loss(net, alpha)
    loss.backward()
    assert net.head.theta.grad is not None
    assert torch.isfinite(loss)


def test_branched_cost_matches_the_analytic_formula():
    """The capacity control has to stay credible, so the extra parameters must
    be a known quantity rather than whatever the code happens to build.

    Against DenseHead, BranchedHead drops fc2 (hidden^2+hidden) and fc_out
    (hidden*n_drugs+n_drugs), and adds G group nodes (G*(hidden^2+hidden)), a
    per-drug MLP each (n_drugs*(hidden*k+k + k+1)), theta itself
    (n_drugs*G) and the generic read-out (hidden*n_drugs+n_drugs). Note the RATIO depends entirely on how big the
    trunk is -- ~1.7x on the toy net below, ~1.10x on the real 19-locus joint
    DNA model -- so the ratio is not the thing to assert on."""
    G, K, H, ND = 4, 64, 256, len(DRUGS)
    specs, kw = [(5, 300)] * 4, dict(drug_names=DRUGS, hidden=H)
    n0 = sum(p.numel() for p in MDCNNNet(specs, **kw).parameters())
    n1 = sum(p.numel() for p in MDCNNNet(
        specs, branched={"n_groups": G, "per_drug_hidden": K}, **kw).parameters())
    expected = (G * (H * H + H) + ND * (H * K + K + K + 1) + ND * G
                + (H * ND + ND) - (H * H + H) - (H * ND + ND))
    assert n1 - n0 == expected, f"added {n1 - n0:,}, formula says {expected:,}"


def test_branched_is_a_small_fraction_of_a_realistic_trunk():
    """On a trunk the size the sweep actually runs, the head is a rounding
    error -- which is what lets a branched-vs-perdrug64 comparison isolate
    ROUTING rather than capacity."""
    specs = [(5, 4000)] * 19          # the joint 19-locus DNA input
    n0 = sum(p.numel() for p in MDCNNNet(specs, drug_names=DRUGS).parameters())
    n1 = sum(p.numel() for p in MDCNNNet(
        specs, drug_names=DRUGS,
        branched={"n_groups": 4, "per_drug_hidden": 64}).parameters())
    assert n1 / n0 < 1.2, f"branched head is {n1 / n0:.2f}x a realistic model"


if __name__ == "__main__":
    check("forward shape matches the DenseHead contract",
          test_forward_shape_matches_densehead_contract)
    check("make_head returns DenseHead for a single output",
          test_make_head_returns_dense_for_single_output)
    check("a 1-drug branched head is rejected", test_requires_at_least_two_drugs)
    check("defaults are the documented ones", test_defaults_are_the_documented_ones)
    check("anneal spans tau_start -> tau_end", test_anneal_spans_tau_start_to_tau_end)
    check("anneal is clamped / survives epochs=1",
          test_anneal_is_clamped_and_survives_a_single_epoch_run)
    check("trainer hook anneals through a whole net",
          test_trainer_hook_anneals_through_a_whole_net)
    check("hooks are a no-op without a branched head",
          test_anneal_hook_is_a_noop_without_a_branched_head)
    check("hard routing gives one-hot weights", test_hard_routing_gives_one_hot_weights)
    check("soft routing is not one-hot", test_soft_routing_is_not_one_hot)
    check("eval routing is deterministic", test_eval_routing_is_deterministic)
    check("eval matches the hard training forward",
          test_eval_matches_the_hard_training_forward)
    check("theta receives gradient (the inertness regression)",
          test_theta_receives_gradient)
    check("hard routing makes group gradients disjoint",
          test_hard_routing_makes_group_gradients_disjoint)
    check("theta_init_std breaks the tie", test_theta_init_std_breaks_the_tie)
    check("aux logits cached on train, cleared on eval",
          test_aux_logits_cached_on_train_and_cleared_on_eval)
    check("aux loss scales with generic_weight",
          test_branch_aux_loss_scales_with_generic_weight)
    check("aux loss honours the label mask", test_aux_loss_is_masked_like_the_main_loss)
    check("assignments partition every drug once",
          test_assignments_partition_every_drug_exactly_once)
    check("n_groups=1 puts everything together", test_single_group_puts_everything_together)
    for g in (1, 2, 4):
        check(f"end-to-end step through mdcnn (G={g})",
              lambda g=g: test_end_to_end_step_through_mdcnn(g))
    check("branched cost matches the analytic formula",
          test_branched_cost_matches_the_analytic_formula)
    check("branched is a small fraction of a realistic trunk",
          test_branched_is_a_small_fraction_of_a_realistic_trunk)
    print(f"\n{sum(_RESULTS)}/{len(_RESULTS)} checks passed")
    sys.exit(0 if all(_RESULTS) else 1)

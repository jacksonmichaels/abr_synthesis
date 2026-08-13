"""
Static / assertion checks for the BIG-TB SD-CNN protocol alignment (the TODO
items #1-#9). These do NOT launch a real training run — every check is either a
pure unit test or a tiny synthetic-fixture integration test that finishes in
seconds on CPU. Run:

    python tests/test_baseline_alignment.py

Each check prints PASS/FAIL and the script exits non-zero if any fails.

Where the pasted TODO's expected numbers disagreed with the *actual* BIG-TB
reference code (Big-TB-benchmark/dna-tasks/SD-CNN), the reference wins and the
discrepancy is called out in the check's comment:
  * alpha values are {-a, 0, +a} with a = R/(R+S) ≈ 0.135 for MOXI, NOT
    {0, 1, 6.39} (the TODO assumed inverse-frequency weighting; tb.alpha_mat
    does not do that).
  * output bias is +log(n_S/n_R) ≈ +1.855, NOT -1.855 (the sigmoid target is
    y==1 = susceptible, the majority class).
  * DNA channel order is (A, C, T, G, gap), NOT (A, C, G, T, gap).
"""
import sys
import tempfile
import traceback
from pathlib import Path

# this file lives in tests/; put the project root on the path so the imports
# below resolve when run directly (python tests/test_baseline_alignment.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402  (import after the sys.path bootstrap, by design)
import torch  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from bigtb_ref import tb  # noqa: E402
from datasets import load_dataset  # noqa: E402
from datasets.fixtures import build_fixture_dataset  # noqa: E402
from datasets.sequences import NT_CHANNELS, one_hot_nt  # noqa: E402
from models import DenseHead  # noqa: E402
from training.core import EarlyStopper, masked_weighted_bce  # noqa: E402
from training.multimodal import run_modal_cv  # noqa: E402

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


# ---------------------------------------------------------------------------
# #3  masked_weighted_bce reduction: invariant to masked padding rows
# ---------------------------------------------------------------------------
def _test_bce_masking_invariance():
    torch.manual_seed(0)
    logits = torch.randn(8, 1)
    # 8 valid rows: a mix of resistant (-a) and susceptible (+a)
    a = 0.135
    alpha = torch.tensor([[+a], [-a], [+a], [+a], [-a], [+a], [-a], [+a]])
    loss_valid = masked_weighted_bce(logits, alpha)

    # same 8 rows padded with 24 all-missing (alpha==0) rows
    pad_logits = torch.cat([logits, torch.randn(24, 1)], dim=0)
    pad_alpha = torch.cat([alpha, torch.zeros(24, 1)], dim=0)
    loss_padded = masked_weighted_bce(pad_logits, pad_alpha)

    assert torch.allclose(loss_valid, loss_padded, atol=1e-6), \
        f"{loss_valid.item()} != {loss_padded.item()} (padding changed the loss)"


# ---------------------------------------------------------------------------
# #4  EarlyStopper: stop at epoch 8, restore epoch-3 weights
# ---------------------------------------------------------------------------
def _test_early_stopper():
    losses = [1.0, 0.8, 0.7, 0.75, 0.76, 0.77, 0.78, 0.79]

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

    model = Dummy()
    stopper = EarlyStopper(patience=5, min_delta=1e-4)
    stop_epoch = None
    for ep, loss in enumerate(losses, start=1):
        model.p.data.fill_(float(ep))          # weights == the epoch number
        if stopper.step(ep, loss, model):
            stop_epoch = ep
            break
    assert stop_epoch == 8, f"stopped at {stop_epoch}, expected 8"
    assert stopper.best_epoch == 3, f"best_epoch={stopper.best_epoch}, expected 3"
    stopper.restore(model)
    assert model.p.item() == 3.0, f"restored weights = epoch {model.p.item()}, expected 3"


# ---------------------------------------------------------------------------
# #2  alpha_mat on a MOXI-shaped label vector -> {-0.135, 0, +0.135}
# ---------------------------------------------------------------------------
def _test_alpha_values():
    y = np.array([0] * 388 + [1] * 2480 + [-1] * 100, dtype=np.float32).reshape(-1, 1)
    vals, counts = np.unique(tb.alpha_mat(y, None, weight=1.0), return_counts=True)
    a = 388 / (388 + 2480)                      # ≈ 0.1353
    assert np.allclose(sorted(vals), [-a, 0.0, a], atol=1e-4), \
        f"alpha values {vals} != approx [-{a:.4f}, 0, +{a:.4f}]"
    # resistant(0) -> -a (388), missing(-1) -> 0 (100), susceptible(1) -> +a (2480)
    count_of = {round(float(v), 4): int(c) for v, c in zip(vals, counts)}
    assert count_of[round(-a, 4)] == 388, count_of
    assert count_of[0.0] == 100, count_of
    assert count_of[round(a, 4)] == 2480, count_of
    # NOTE: the TODO expected {0, 1, 6.39}; the real tb.alpha_mat does not do
    # inverse-frequency weighting, so that expectation is wrong.


# ---------------------------------------------------------------------------
# #5  StratifiedKFold(5, seed 42): every fold's positive rate ~ global
# ---------------------------------------------------------------------------
def _test_stratified_fold_balance():
    # MOXI-scale training split: 2294 rows at the 0.135 resistant base rate.
    rng = np.random.default_rng(0)
    n, pos_rate = 2294, 0.135
    y = (rng.random(n) < pos_rate).astype(int)      # 1 = resistant here, rate ~0.135
    global_rate = y.mean()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, va) in enumerate(skf.split(np.arange(n), y)):
        rate = y[va].mean()
        assert abs(rate - global_rate) < 0.02, \
            f"fold {fold} rate {rate:.4f} vs global {global_rate:.4f} (>0.02)"


# ---------------------------------------------------------------------------
# #6  DenseHead output bias initialised to log(n_pos/n_neg)
# ---------------------------------------------------------------------------
def _test_output_bias_init():
    n_S, n_R = 2480, 388                       # positive class = susceptible (y==1)
    expected = float(np.log(n_S / n_R))        # +1.855, NOT the TODO's -1.855
    head = DenseHead(in_features=32, out_dim=1, out_bias=expected)
    assert abs(head.fc_out.bias.item() - expected) < 1e-5, \
        f"bias {head.fc_out.bias.item():.4f} != {expected:.4f}"
    assert abs(expected - 1.855) < 1e-3, f"log-odds {expected:.4f} != +1.855"
    # out_bias=None leaves PyTorch's default Linear bias untouched (small
    # uniform, NOT zero — that is a Keras-vs-Torch init difference that cannot
    # affect AUC ranking); just confirm the log-odds path took effect above.


# ---------------------------------------------------------------------------
# #7  DNA one-hot encoder: (A,C,T,G,gap); N -> all-zero, distinct from gap
# ---------------------------------------------------------------------------
def _test_dna_one_hot():
    assert NT_CHANNELS == ['A', 'C', 'T', 'G', '-'], f"channel order {NT_CHANNELS}"
    oh = one_hot_nt("ACGT-N")                  # (L=6, 5)
    expected = np.array([
        [1, 0, 0, 0, 0],  # A -> col 0
        [0, 1, 0, 0, 0],  # C -> col 1
        [0, 0, 0, 1, 0],  # G -> col 3
        [0, 0, 1, 0, 0],  # T -> col 2
        [0, 0, 0, 0, 1],  # - (gap) -> col 4
        [0, 0, 0, 0, 0],  # N -> all zero (does NOT collide with gap)
    ], dtype=np.float32)
    assert np.array_equal(oh, expected), f"one-hot mismatch:\n{oh}"
    # every position sums to exactly 1 (a defined base or gap) or 0 (unknown)
    assert set(np.unique(oh.sum(axis=1))) <= {0.0, 1.0}, "position sums not in {0,1}"


# ---------------------------------------------------------------------------
# #8  MOXI locus span: gene_order == [gyrB, gyrA], DNA len == sum of gene lens
# ---------------------------------------------------------------------------
def _test_moxi_locus_span():
    assert tb.DRUG_TO_LOCI["MOXIFLOXACIN"] == ["gyrB", "gyrA"], \
        f"DRUG_TO_LOCI[MOXI] = {tb.DRUG_TO_LOCI['MOXIFLOXACIN']}"
    with tempfile.TemporaryDirectory() as tmp:
        n_codons = 30
        geno_dir, pheno = build_fixture_dataset(
            tmp, genes=["gyrB", "gyrA"], drugs=["MOXIFLOXACIN"],
            n_isolates=40, n_codons=n_codons, seed=0)
        data = load_dataset("MOXIFLOXACIN", ["dna"], geno_dir, pheno, verbose=False)
        assert data.gene_order == ["gyrB", "gyrA"], f"gene_order {data.gene_order}"
        dna = next(b for b in data.blocks if b.modality == "dna")
        # aligned fixtures: each gene is n_codons*3 columns; the block is the two
        # concatenated along the sequence axis, in [gyrB, gyrA] order.
        assert dna.length == 2 * n_codons * 3, \
            f"DNA block length {dna.length} != gyrB+gyrA ({2 * n_codons * 3})"
        assert dna.channels == 5, f"DNA channels {dna.channels} != 5"


# ---------------------------------------------------------------------------
# #1/#9 integration: run_modal_cv drops missing, reports CV mean+std and TEST
# ---------------------------------------------------------------------------
def _test_run_modal_cv_contract():
    with tempfile.TemporaryDirectory() as tmp:
        geno_dir, pheno = build_fixture_dataset(
            tmp, genes=["gyrB", "gyrA"], drugs=["MOXIFLOXACIN"],
            n_isolates=300, n_codons=20, seed=1)
        data = load_dataset("MOXIFLOXACIN", ["dna"], geno_dir, pheno, verbose=False)
        n_missing = int((data.y == -1).sum())
        assert n_missing > 0, "fixture should contain some missing phenotypes to drop"
        # out_bias='auto' exercises the log-odds init path (#6); the shipped
        # default is now out_bias=None (see the MOXI collapse fix in
        # training.multimodal's docstring), which would make out_bias None here.
        res = run_modal_cv(data, epochs=3, n_splits=3, batch_size=64,
                           device="cpu", seed=0, patience=2, min_delta=1e-4,
                           out_bias="auto")
        # #1: only valid isolates survive the filter
        assert res["n_valid"] == data.n - n_missing, "missing rows not dropped"
        assert res["n_resistant"] + res["n_susceptible"] == res["n_valid"]
        # #9: CV mean AND std present and finite, TEST reported separately
        for k in ("cv_auc_mean", "cv_auc_std"):
            assert k in res and np.isfinite(res[k]), f"missing/NaN {k}"
        assert "test" in res and "auc" in res["test"], "no TEST metrics"
        assert "test_model_fold" in res, "TEST model provenance not recorded"
        # #6: bias recorded and equals train-split log-odds sign (positive here?
        # depends on the fixture's class balance — just assert it's finite)
        assert np.isfinite(res["out_bias"]), "out_bias not finite"


def main():
    check("#3  masked_weighted_bce invariant to masked padding", _test_bce_masking_invariance)
    check("#4  EarlyStopper stops @8, restores @3", _test_early_stopper)
    check("#2  alpha_mat values {-0.135, 0, +0.135}", _test_alpha_values)
    check("#5  StratifiedKFold fold balance within +/-0.02", _test_stratified_fold_balance)
    check("#6  DenseHead output-bias = +log(n_pos/n_neg)", _test_output_bias_init)
    check("#7  DNA one-hot (A,C,T,G,gap); N->zero", _test_dna_one_hot)
    check("#8  MOXI gene_order [gyrB,gyrA] + DNA length", _test_moxi_locus_span)
    check("#1/#9 run_modal_cv filters missing + reports CV/TEST", _test_run_modal_cv_contract)

    n_pass = sum(_RESULTS)
    print(f"\n{n_pass}/{len(_RESULTS)} checks passed")
    raise SystemExit(0 if n_pass == len(_RESULTS) else 1)


if __name__ == "__main__":
    main()

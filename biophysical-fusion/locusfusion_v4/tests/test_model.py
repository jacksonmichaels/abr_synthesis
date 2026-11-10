"""
Augmenter and model, on a small synthetic batch (CPU, seconds, no data).

    cd biophysical-fusion
    python -m locusfusion_v4.tests.test_model
"""
import functools

import numpy as np
import torch

from locusfusion_v4.augment import BATCH_KEYS, SynonymousAugmenter
from locusfusion_v4.layout import C_BASE, DIST_MISSING, SLOTS, SYN
from locusfusion_v4.model import LocusFusionV4Net, proximity_table

B, L, V = 64, 3, 6


@functools.cache
def setup():
    rng = np.random.default_rng(0)
    # candidates: loci 0 and 1 are coding (codons 1..19), locus 2 has none
    cand = {"locus": [], "codon": [], "coord": [], "site_pos": []}
    for l in (0, 1):
        for k in range(1, 20):
            cand["locus"].append(l)
            cand["codon"].append(k)
            cand["coord"].append(k + 0.33)
            cand["site_pos"].append(k)
    M = len(cand["locus"])
    feat = np.zeros((M, C_BASE), np.float32)
    feat[:, SLOTS["dna"][0]] = 1
    feat[:, SLOTS["consequence"][0] + SYN] = 1
    cand = {k: np.asarray(v) for k, v in cand.items()} | {"feat": feat}
    n_tok = rng.integers(0, V + 1, (B, L))
    n_tok[0] = V                                   # isolate 0: every locus full
    valid = np.arange(V)[None, None, :] < n_tok[..., None]
    codon = np.where(valid, rng.integers(0, 20, (B, L, V)), -1).astype(np.int16)
    batch = {"feat": torch.as_tensor(rng.normal(size=(B, L, V, C_BASE)).astype(np.float32)
                                     * valid[..., None]),
             "coord": torch.as_tensor(rng.uniform(0, 20, (B, L, V)).astype(np.float32)),
             "valid": torch.as_tensor(valid),
             "codon": torch.as_tensor(codon),
             "site_pos": torch.as_tensor(np.where(valid, codon, -1).astype(np.int16)),
             "n_occ": torch.as_tensor(n_tok.astype(np.int32))}
    return cand, batch


def test_augment_respects_cap_codons_and_counts():
    cand, batch = setup()
    aug = SynonymousAugmenter(cand, "cpu")
    gen = torch.Generator().manual_seed(0)
    before = {k: v.clone() for k, v in batch.items()}
    out, ins = aug(batch, k_max=3, gen=gen)
    for k in BATCH_KEYS:                                       # input untouched
        assert torch.equal(batch[k], before[k]), k
    assert int(ins.sum()) > 0
    assert ins[0] == 0                                         # full isolate gets nothing
    new = out["valid"] & ~batch["valid"]
    assert torch.equal(new.sum((1, 2)), ins)
    assert torch.equal(out["n_occ"] - batch["n_occ"], new.sum(2).to(torch.int32))
    assert (out["valid"].sum(2) <= V).all()
    assert not new[:, 2].any()                                 # no candidates on locus 2
    # a new token's codon was not already varied in that isolate's locus
    for b, l, s in zip(*torch.nonzero(new, as_tuple=True)):
        others = out["codon"][b, l][batch["valid"][b, l]]
        assert out["codon"][b, l, s] not in others.tolist()
        assert out["feat"][b, l, s, SLOTS["consequence"][0] + SYN] == 1
    # within one copy, two new tokens never share a codon
    for b in range(B):
        for l in range(L):
            c = out["codon"][b, l][new[b, l]].tolist()
            assert len(c) == len(set(c))


def test_exact_k():
    cand, batch = setup()
    aug = SynonymousAugmenter(cand, "cpu")
    _, ins = aug(batch, gen=torch.Generator().manual_seed(1), exact_k=2)
    assert ins.max() <= 2 and (ins == 2).float().mean() > 0.5


def _model(n_dist, drugs, seed=0):
    torch.manual_seed(seed)
    dist = {("D0", "a"): np.linspace(0, 40, 25).astype(np.float32),
            ("D1", "b"): np.full(25, np.nan, np.float32)}
    prox = proximity_table(["D0", "D1"][:n_dist], ["a", "b", "c"], dist)
    return LocusFusionV4Net(["a", "b", "c"], kept=[100, 100, 100], drugs=drugs, prox=prox,
                            dist_drugs=["D0", "D1"][:n_dist], max_variants=V).eval(), prox


def test_proximity_table():
    _, prox = _model(2, ["D0", "D1"])
    assert prox.shape == (2, 3, 26)
    assert np.isclose(prox[0, 0, 0], 1.0) and np.isclose(prox[0, 0, 24], np.exp(-4.0))
    assert (prox[0, 1:] == DIST_MISSING).all() and (prox[1] == DIST_MISSING).all()
    assert (prox[:, :, -1] == DIST_MISSING).all()              # the 'no position' slot


def test_tokens_null_and_padding():
    _, batch = setup()
    m, _ = _model(2, ["D0", "D1"])
    x, coord, valid = m.tokens(batch["feat"], batch["coord"], batch["valid"], batch["site_pos"])
    assert x.shape == (B, L, V + 1, C_BASE + 2)
    assert (x[:, :, 0, SLOTS["is_null"][0]] == 1).all()
    assert (x[:, :, 0, C_BASE:] == DIST_MISSING).all()
    assert (x[:, :, 1:][~batch["valid"]] == 0).all()
    assert valid[:, :, 0].all()
    sp = batch["site_pos"].long()
    on_a = batch["valid"][:, 0] & (sp[:, 0] >= 0)
    want = torch.exp(-torch.linspace(0, 40, 25)[sp[:, 0][on_a]] / 10)
    assert torch.allclose(x[:, 0, 1:, C_BASE][on_a], want, atol=1e-5)


def test_forward_shapes_and_slot_permutation_invariance():
    _, batch = setup()
    for n_dist, drugs in ((1, ["D0"]), (2, ["D0", "D1"]), (0, ["D0", "D1"])):
        m, _ = _model(n_dist, drugs)
        args = {k: batch[k] for k in ("feat", "coord", "valid", "site_pos", "n_occ")}
        with torch.no_grad():
            y = m(**args)
            perm = torch.randperm(V)
            y2 = m(**{k: (v if k == "n_occ" else v[:, :, perm]) for k, v in args.items()})
        assert y.shape == (B, len(drugs))
        assert torch.allclose(y, y2, atol=1e-5), "slot order must not matter"


def test_backward():
    _, batch = setup()
    m, _ = _model(2, ["D0", "D1"])
    m.train()
    y = m(**{k: batch[k] for k in ("feat", "coord", "valid", "site_pos", "n_occ")})
    y.sum().backward()
    assert m.tok_proj.weight.grad is not None and torch.isfinite(m.tok_proj.weight.grad).all()


if __name__ == "__main__":
    from locusfusion_v4.tests._run import run
    run(globals())

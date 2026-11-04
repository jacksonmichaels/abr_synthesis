"""
GenomeFusion model checks, on synthetic batches (seconds, CPU).

    cd biophysical-fusion && python -m genomefusion.tests.test_model

The one that matters is `test_bucketing`: the within-locus encoder pads each
bucket of pairs only to its own token count, so a locus with one token never
sits in a tensor sized for the busiest locus in the batch. That has to give the
same answer as running every pair at one width.
"""
import numpy as np
import torch

from genomefusion.model import BUCKETS, GenomeFusionNet


def fake_batch(B=5, n_loci=40, c_tok=57, seed=0, counts=None):
    """A packed batch with a deliberate spread of tokens per locus."""
    rng = np.random.default_rng(seed)
    pair_bi, pair_slot, pair_lid, pair_n = [], [], [], []
    for b in range(B):
        n_act = rng.integers(2, 8)
        for s in range(n_act):
            pair_bi.append(b)
            pair_slot.append(s + 1)
            pair_lid.append(int(rng.integers(0, n_loci)))
    P = len(pair_bi)
    if counts is None:
        # 1 token for most pairs, and a few far above the first bucket
        counts = rng.choice([1, 1, 1, 2, 3, 5, 9, 17, 40], size=P)
    counts = np.asarray(counts)[:P]
    tok_pair = np.repeat(np.arange(P), counts)
    N = len(tok_pair)
    t = lambda a, d=torch.long: torch.as_tensor(np.asarray(a), dtype=d)  # noqa: E731
    return {
        "B": B, "Lmax": max(pair_slot) + 1, "P": P,
        "tok_feat": torch.as_tensor(rng.normal(size=(N, c_tok)), dtype=torch.float16),
        "tok_coord": torch.as_tensor(rng.integers(0, 3000, N), dtype=torch.float32),
        "tok_site": t(np.full(N, -1)),
        "tok_pair": t(tok_pair),
        "pair_bi": t(pair_bi), "pair_slot": t(pair_slot), "pair_lid": t(pair_lid),
        "pair_stats": torch.as_tensor(rng.normal(size=(P, 4)), dtype=torch.float32),
        "pair_n": t(counts),
    }


def net(n_loci=40, n_feat=26, drugs=("ISONIAZID",), c_tok=57, seed=0, **kw):
    torch.manual_seed(seed)
    feats = np.random.default_rng(seed).normal(size=(n_loci, n_feat)).astype(np.float32)
    m = GenomeFusionNet(feats, list(drugs), c_tok=c_tok, dropout=0.0, **kw)
    return m.eval()


def test_forward():
    m, b = net(), fake_batch()
    out = m(b)
    assert out.shape == (b["B"], 1), out.shape
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert m.tok_proj.weight.grad.abs().sum() > 0
    print("test_forward: ok")


def test_bucketing():
    """Bucketed pairs must equal one pass at a single width."""
    m, b = net(), fake_batch(seed=3)
    with torch.no_grad():
        got = m.encode_pairs(b)
        import genomefusion.model as M
        saved = M.BUCKETS
        try:                                   # one bucket wide enough for every pair
            M.BUCKETS = (int(b["pair_n"].max()),)
            want = m.encode_pairs(b)
        finally:
            M.BUCKETS = saved
    d = (got - want).abs().max().item()
    assert d < 2e-5, f"bucketed encoder differs by {d}"
    print(f"test_bucketing: ok (max |diff| {d:.2e}, buckets {BUCKETS[:4]}...)")


def test_pair_isolation():
    """A pair's locus vector must not depend on the other pairs in the batch."""
    m = net(seed=1)
    b = fake_batch(B=4, seed=5)
    with torch.no_grad():
        full = m.encode_pairs(b)
        # keep only the pairs of batch row 0, renumbered
        keep = (b["pair_bi"] == 0)
        idx = torch.nonzero(keep, as_tuple=True)[0]
        remap = torch.full((b["P"],), -1, dtype=torch.long)
        remap[idx] = torch.arange(len(idx))
        tk = remap[b["tok_pair"]] >= 0
        sub = dict(b)
        sub.update(B=1, P=len(idx),
                   pair_bi=b["pair_bi"][idx], pair_slot=b["pair_slot"][idx],
                   pair_lid=b["pair_lid"][idx], pair_stats=b["pair_stats"][idx],
                   pair_n=b["pair_n"][idx],
                   tok_feat=b["tok_feat"][tk], tok_coord=b["tok_coord"][tk],
                   tok_site=b["tok_site"][tk], tok_pair=remap[b["tok_pair"][tk]])
        part = m.encode_pairs(sub)
    d = (full[idx] - part).abs().max().item()
    assert d < 2e-5, f"a pair's vector moved by {d} when other pairs were present"
    print(f"test_pair_isolation: ok (max |diff| {d:.2e})")


def test_reference_token_only():
    """An isolate with no active locus still gets a logit, from the reference token."""
    m = net()
    b = fake_batch()
    b.update(P=0, tok_pair=torch.zeros(0, dtype=torch.long),
             tok_feat=torch.zeros(0, 57, dtype=torch.float16),
             tok_coord=torch.zeros(0), tok_site=torch.zeros(0, dtype=torch.long),
             pair_bi=torch.zeros(0, dtype=torch.long),
             pair_slot=torch.zeros(0, dtype=torch.long),
             pair_lid=torch.zeros(0, dtype=torch.long),
             pair_stats=torch.zeros(0, 4), pair_n=torch.zeros(0, dtype=torch.long), Lmax=1)
    out = m(b)
    assert torch.isfinite(out).all() and out.shape == (b["B"], 1)
    assert (out - out[0]).abs().max() < 1e-6, "empty isolates must all score the same"
    print("test_reference_token_only: ok")


def test_padding_invariance():
    """Extra padded locus slots must not change any logit."""
    m, b = net(), fake_batch(seed=7)
    with torch.no_grad():
        a = m(b)
        b2 = dict(b, Lmax=b["Lmax"] + 25)
        c = m(b2)
    d = (a - c).abs().max().item()
    assert d < 2e-5, f"padding changed the logits by {d}"
    print(f"test_padding_invariance: ok (max |diff| {d:.2e})")


def test_distance_channels():
    m = net(n_dist=1, site_table=np.array([[1.0, 0.5, 0.0]], np.float32))
    b = fake_batch()
    b["tok_site"] = torch.as_tensor(
        np.random.default_rng(0).integers(-1, 3, len(b["tok_pair"])), dtype=torch.long)
    out = m(b)
    assert torch.isfinite(out).all()
    assert m.c_in == 58, m.c_in
    print("test_distance_channels: ok")


def test_samplers():
    """Both samplers must cover every row exactly once and respect the budget."""
    from genomefusion.data import budget_batches, length_grouped_batches
    rng = np.random.default_rng(0)
    n_active = rng.integers(50, 2500, 5000)
    rows = rng.permutation(5000)[:3000]
    for name, chunks in (("train", length_grouped_batches(n_active, rows, 32, rng,
                                                          locus_budget=32768)),
                         ("eval", budget_batches(n_active, rows, 64, 32768))):
        flat = np.concatenate(chunks)
        assert len(flat) == len(rows), f"{name}: {len(flat)} rows for {len(rows)}"
        assert set(flat.tolist()) == set(rows.tolist()), f"{name}: rows changed"
        worst = max(len(c) * (int(n_active[c].max()) + 1) for c in chunks)
        assert worst <= 32768 * 2, f"{name}: budget overshot at {worst}"
        assert all(len(c) >= 1 for c in chunks)
    print("test_samplers: ok (both cover every row once, batch x width bounded)")


def test_predict_order():
    """predict() must return rows in the order asked, not the order batched."""
    from genomefusion.train import predict

    class FakeBB:
        n_active = np.arange(100) * 20 + 10

        def build(self, rows):
            return {"rows": np.asarray(rows)}

    class FakeModel:
        def eval(self):
            pass

        def __call__(self, b):
            return torch.as_tensor(b["rows"], dtype=torch.float32).unsqueeze(1)

    rows = np.random.default_rng(1).permutation(100)[:60]
    got = predict(FakeModel(), FakeBB(), rows, 8, 1, locus_budget=512)
    assert np.array_equal(got[:, 0], rows.astype(np.float32)), "predict reordered its output"
    print("test_predict_order: ok")


if __name__ == "__main__":
    test_forward()
    test_bucketing()
    test_pair_isolation()
    test_reference_token_only()
    test_padding_invariance()
    test_distance_channels()
    test_samplers()
    test_predict_order()
    print("all model tests passed")

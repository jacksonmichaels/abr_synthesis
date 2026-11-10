"""One real forward pass of the GenomeFusion whole-genome model.

    sbatch scripts/sbatch/genomefusion_forward_trace.sh       # CPU inference; not for a login node

Writes results/figures/genomefusion/forward_trace.json, which
scripts/genomefusion_method_figures.py draws. The isolate is the one in method_trace.json.

Runs the frozen code the model was trained with, on the token store it read. Only the
isolates scored here are expanded, so it needs a few GB rather than the trainer's many.

Three checks fail the script rather than letting a wrong figure out:
  * the checkpoint's locus features must equal this partition's;
  * re-scoring the example and SAMPLE TEST isolates must match the predictions saved by
    the training run;
  * the step-by-step replay must reproduce model(batch), for the logit and for each
    traced locus's vector.

Labels: GenomeFusion encodes y = 1 resistant, so its sigmoid is P(resistant), although
the prediction files name that column P_S_<drug>.
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WS = PROJECT.parent
ap = argparse.ArgumentParser()
ap.add_argument("--code", default=str(WS / "gf_snapshot" / "20261105_024429" / "biophysical-fusion"))
ap.add_argument("--partition-cache", default=str(WS / "genomefusion_cache_v3"))
ap.add_argument("--weights", default=str(WS / "model_weights_gf" / "size6776" / "ISONIAZID" / "best.pt"))
ap.add_argument("--results", default=str(PROJECT / "results" / "experiments" / "genomefusion" / "size6776"))
ap.add_argument("--out", default=str(PROJECT / "results" / "figures" / "genomefusion" / "forward_trace.json"))
args = ap.parse_args()
sys.dont_write_bytecode = True                   # the snapshot is read-only
sys.path.insert(0, args.code)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from canonical import genomefusion as gf_model  # noqa: E402
from genomefusion.data import TOKEN_FIELDS, BatchBuilder  # noqa: E402
from genomefusion.layout import C_BASE  # noqa: E402
from genomefusion.model import GenomeFusionNet, _sinusoid  # noqa: E402
from genomefusion.partition import load  # noqa: E402

assert Path(gf_model.__file__).resolve().is_relative_to(Path(args.code).resolve()), gf_model.__file__
DRUG = "ISONIAZID"
TOKENS = WS / "genomefusion_tokens" / "tokens.npz"
SAMPLE, SEED, TOL = 200, 0, 1e-3
TRACE_LOCI = ("katG", "Rv1482c-Rv1483", "Rv0387c", "ctpI")
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))

M = json.loads((PROJECT / "results" / "figures" / "genomefusion" / "method_trace.json").read_text())
EX = M["example"]["isolate"]
res_dir = Path(args.results)
record = json.loads((res_dir / f"{DRUG}.json").read_text())
assert record["complete"] and len(record["cv_folds"]) == 5, "the run did not finish all folds"
saved = pd.read_csv(res_dir / f"{DRUG}_test_predictions.csv", dtype={"New_ID": str}).set_index("New_ID")
PCOL, YCOL = f"P_S_{DRUG}", f"y_{DRUG}"
assert roc_auc_score(saved[YCOL], saved[PCOL]) - record["test"]["auc"] < 1e-6
rng = np.random.default_rng(SEED)
chosen = [EX] + [i for i in rng.choice(saved.index.values, SAMPLE, replace=False) if i != EX]


# ---- a small store holding only these isolates --------------------------------------
class SubStore:
    def __init__(self, path, wanted):
        z = np.load(path, allow_pickle=False)
        all_ids = [str(x) for x in z["ids"]]
        where = {s: i for i, s in enumerate(all_ids)}
        rows = [where[w] for w in wanted]
        to, so = z["tok_off"], z["sum_off"]
        trow = np.concatenate([np.arange(to[r], to[r + 1]) for r in rows])
        srow = np.concatenate([np.arange(so[r], so[r + 1]) for r in rows])
        self.ids = list(wanted)
        self.tok = {k: z[k][trow] for k in TOKEN_FIELDS}
        for k in ("sum_locus", "sum_n", "sum_span", "sum_frame", "sum_oof"):
            setattr(self, k, z[k][srow])
        self.tok_off = np.concatenate([[0], np.cumsum([to[r + 1] - to[r] for r in rows])])
        self.sum_off = np.concatenate([[0], np.cumsum([so[r + 1] - so[r] for r in rows])])

    def __len__(self):
        return len(self.ids)


P = load(Path(args.partition_cache) / "partition.npz")
store = SubStore(TOKENS, chosen)
bb = BatchBuilder(store, P, device="cpu")
ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
cfg = ckpt["config"]
feats = P.feature_matrix(with_regulated=True)
net = GenomeFusionNet(feats, cfg["drugs"], c_tok=bb.c_tok, regulated_lid=P.regulated_index(),
                      **{k: cfg[k] for k in ("d_model", "nhead", "dropout", "enc_layers", "enc_dim_ff", "fusion_layers",
                                             "fusion_dim_ff", "pos_dims", "hidden", "residual_scale", "residual_dropout")})
assert np.array_equal(ckpt["state_dict"]["locus_feat"].numpy(), feats.astype(np.float32)), \
    "this checkpoint was trained on a different partition"
assert np.array_equal(ckpt["state_dict"]["regulated_lid"].numpy(), P.regulated_index()), "edges differ"
net.load_state_dict(ckpt["state_dict"], strict=True)
net.eval()
assert sum(p.numel() for p in net.parameters()) == record["n_params"]

# ---- check: re-score against the training run's saved predictions ---------------------
p_all, att_ref, att_max, n_slots = [], [], [], []
with torch.no_grad():
    for i in range(len(chosen)):
        logit, att = net(bb.build([i]), return_attn=True)
        p_all.append(float(torch.sigmoid(logit)))
        a = att[0, 0]
        att_ref.append(float(a[0]))
        att_max.append(float(a[1:].max()) if a.numel() > 1 else float("nan"))
        n_slots.append(int(a.numel()))
diff = np.abs(np.asarray(p_all) - saved.loc[chosen, PCOL].values)
print(f"rescored {len(chosen)} TEST isolates: max |difference| from the saved predictions {diff.max():.2e}", flush=True)
assert diff.max() < TOL, f"rebuilt model disagrees with the saved predictions by {diff.max()}"


# ---- the replay -----------------------------------------------------------------------
def encoder_with_attention(enc, x, pad):
    maps = []
    for layer in enc.layers:
        assert layer.norm_first
        q = layer.norm1(x)
        _, w = layer.self_attn(q, q, q, key_padding_mask=pad, need_weights=True, average_attn_weights=True)
        maps.append(w)
        x = layer(x, src_key_padding_mask=pad)
    assert enc.norm is None
    return x, maps


def vec(t, d=4):
    return np.round(t.detach().numpy().astype(float), d).tolist()


def nrm(t):
    return round(float(t.norm()), 3)


name_of = lambda L: L.gene or L.name        # noqa: E731
b = bb.build([0])
with torch.no_grad():
    lid = b["pair_lid"]
    lvec = net.locus_vectors(lid)
    scale, shift = net.film(lid)
    null = net.null_emb.unsqueeze(0) + net.null_proj(b["pair_stats"]) + lvec
    x = b["tok_feat"].to(torch.float32)
    t_feat = net.tok_proj(x)
    sin = _sinusoid(b["tok_coord"], net.pos_dims)
    t_pos = net.pos_proj(sin)
    t_sum = t_feat + t_pos + lvec[b["tok_pair"]]
    pair_vec = net.encode_pairs(b)
    names = [name_of(P.loci[int(i)]) for i in lid]

    traced = {}
    for loc in TRACE_LOCI:
        pi = names.index(loc)
        idx = torch.nonzero(b["tok_pair"] == pi, as_tuple=True)[0]
        seq = torch.cat([null[pi:pi + 1], t_sum[idx]], 0).unsqueeze(0)
        s, sh = scale[pi].view(1, 1, -1), shift[pi].view(1, 1, -1)
        film = seq * s + sh
        normed = net.tok_norm(film)
        out, maps = encoder_with_attention(net.encoder, normed, torch.zeros(1, seq.shape[1], dtype=torch.bool))
        assert torch.allclose(out[0, 0], pair_vec[pi], atol=1e-4), f"{loc}: replay != encode_pairs"
        pos = store.tok["pos"][bb.tok_rows[:b["tok_pair"].numel()]][idx.numpy()]
        traced[loc] = dict(pair=pi, positions=[int(v) for v in pos], attention=[vec(w[0]) for w in maps],
                           output_norm=nrm(pair_vec[pi]))

    # every term of one variant token and of its [NULL], at full width
    ki = names.index("katG")
    ti = int(torch.nonzero(b["tok_pair"] == ki, as_tuple=True)[0][0])
    L = int(lid[ki])
    feat_part = net.feat_proj(net.locus_feat[L:L + 1])[0]
    res_part = net.residual_scale * net.locus_residual.weight[L]
    tgt = int(net.regulated_lid[L])
    edge_part = (net.reg_gate * net.locus_residual.weight[tgt]) if tgt >= 0 else torch.zeros_like(res_part)
    assert torch.allclose(feat_part + res_part + edge_part, lvec[ki], atol=1e-4), "locus identity replay failed"
    film_tok = t_sum[ti] * scale[ki] + shift[ki]
    film_null = null[ki] * scale[ki] + shift[ki]
    showcase = dict(
        coord=float(b["tok_coord"][ti]), feat=vec(x[ti]), sinusoid=vec(sin[ti]), tok_proj=vec(t_feat[ti]),
        pos_proj=vec(t_pos[ti]), feat_proj=vec(feat_part), residual=vec(res_part), edge=vec(edge_part),
        locus=vec(lvec[ki]), sum=vec(t_sum[ti]), film=vec(film_tok), layernorm=vec(net.tok_norm(film_tok)),
        null_emb=vec(net.null_emb), stats=vec(b["pair_stats"][ki]), null_proj=vec(net.null_proj(b["pair_stats"][ki:ki + 1])[0]),
        null_sum=vec(null[ki]), null_film=vec(film_null), null_layernorm=vec(net.tok_norm(film_null)),
        film_scale=vec(scale[ki] - 1.0), film_shift=vec(shift[ki]), reg_gate=float(net.reg_gate))

    # stage 2 and the readout
    Lmax = b["Lmax"]
    fused = torch.zeros(1, Lmax, net.d_model)
    fused[:, 0] = net.reference
    fused[b["pair_bi"], b["pair_slot"]] = pair_vec
    normed = net.summ_norm(fused)
    pad = torch.zeros(1, Lmax, dtype=torch.bool)
    out, att2 = encoder_with_attention(net.fusion, normed, pad)
    q = net.drug_queries.unsqueeze(0)
    pooled, att3 = net.pool_attn(q, out, out, key_padding_mask=pad, need_weights=True, average_attn_weights=True)
    pn = net.norm(pooled)
    h = torch.relu(net.fc1(pn))
    logit = net.fc_out(h).squeeze(-1)
    direct = net(b)
assert torch.allclose(logit, direct, atol=1e-4), f"replay {logit.item()} != model {direct.item()}"
print(f"replay reproduces model(batch): logit {logit.item():.5f}", flush=True)

slots = ["reference"] + names
slot_of = {n: i for i, n in enumerate(slots)}
Q = att3[0, 0].detach().numpy()
A = [w[0].detach().numpy() for w in att2]


def top(v, k=15):
    return [dict(slot=slots[int(i)], weight=round(float(v[i]), 5)) for i in np.argsort(v)[::-1][:k]]


TRACK = ["reference"] + list(TRACE_LOCI)
stage2 = []
for a in A:
    stage2.append(dict(
        incoming_mean=top(a.mean(0), 15),
        diagonal_mean=round(float(np.diag(a).mean()), 5),
        max=round(float(a.max()), 4),
        rows={n: top(a[slot_of[n]], 8) for n in TRACK},
        # a small real block of the full matrix: the tracked slots and the most attended ones
        block_slots=None))
order = list(dict.fromkeys(TRACK + [t["slot"] for t in top(A[-1].mean(0), 12)]))[:16]
idx = [slot_of[n] for n in order]
for s_, a in zip(stage2, A):
    s_["block_slots"] = order
    s_["block"] = np.round(a[np.ix_(idx, idx)].astype(float), 4).tolist()

ids_y = saved.loc[chosen, YCOL].values
OUT = Path(args.out)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(dict(
    isolate=EX, drug=DRUG, weights=args.weights, partition_cache=args.partition_cache, code=args.code,
    fold=record["test_model_fold"], best_epoch=record["cv_folds"][record["test_model_fold"]]["best_epoch"],
    test_auc=record["test"]["auc"], cv_auc=record["cv_auc_mean"], test_n=int(len(saved)), test_R=int((saved[YCOL] == 1).sum()),
    checks=dict(rescored=len(chosen), max_diff=float(diff.max()), replay_logit=float(logit), direct_logit=float(direct)),
    n_tokens=int(b["tok_pair"].numel()), n_pairs=int(b["P"]), lmax=int(Lmax), traced=traced, showcase=showcase,
    pair_vector_norm=dict(median=float(pair_vec.norm(dim=1).median()), max=float(pair_vec.norm(dim=1).max())),
    stage2=stage2, query=dict(reference=round(float(Q[0]), 5), uniform=round(1 / Lmax, 5), max=round(float(Q.max()), 5),
                              min=round(float(Q.min()), 5), top=top(Q, 20)),
    query_vector_norm=nrm(net.drug_queries[0]), pooled_norm=nrm(pooled[0, 0]), layernorm_norm=nrm(pn[0, 0]),
    h_active=int((h[0, 0] > 0).sum()), logit=float(logit), P_R=float(torch.sigmoid(logit)), y=int(saved.loc[EX, YCOL]),
    sample=dict(n=len(chosen), P_R=[round(v, 4) for v in p_all], y=[int(v) for v in ids_y],
                query_reference=[round(v, 5) for v in att_ref], query_max_locus=[round(v, 5) for v in att_max], slots=n_slots)),
    indent=None))
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
print(f"example P(R) {float(torch.sigmoid(logit)):.4f}, query on reference {Q[0]:.4f}, max slot {Q.max():.4f}, uniform {1 / Lmax:.4f}")

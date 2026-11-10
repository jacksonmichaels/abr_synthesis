"""Trace one real forward pass of the LocusFusion v3 headline model.

    python scripts/locusfusion_forward_trace.py     # CPU, ~ a few GB, a few minutes

Writes results/figures/manuscript/forward_pass_trace.json, which
scripts/locusfusion_method_figures.py draws, so the figures can be restyled
without reloading isolates.

Model: the lineage-split, 19-locus, all-modalities ISONIAZID cell, fold 1 (the
checkpoint that cell's TEST metrics came from), rebuilt with the frozen code it was
trained with (lineage_snapshot). Only the 3,959 TEST isolates are read, through
filtered temporary copies of the phenotype table and FASTAs and CHUNK isolates at a
time, because blocks are dense float32 (~3.4 MB per isolate) and only held-out
isolates belong in this figure.

Two checks fail the script rather than letting a wrong figure out:
  * the rescored TEST AUC must match the training record;
  * the step-by-step replay must reproduce net(xs) exactly.

Labels: y = 0 resistant, y = 1 susceptible. The network's sigmoid is P(susceptible);
P(resistant) = 1 - sigmoid(logit). The isolate is chosen by what it carries (katG
S315T, a promoter base, a synonymous change and an uncalled base), never by whether
the model called it correctly.
"""
import gc
import itertools
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WS = PROJECT.parent
SNAP = WS / "lineage_snapshot" / "biophysical-fusion"
sys.dont_write_bytecode = True                  # the snapshot is read-only
sys.path.insert(0, str(SNAP))
DATA = WS / "data_snapshot"
os.environ.setdefault("ABR_GENOTYPE_DIR", str(DATA / "genomic_data" / "aligned"))
os.environ.setdefault("ABR_PHENOTYPE_CSV", str(DATA / "phenotype_data" / "master_resistance_table.csv"))
os.environ.setdefault("ABR_GENBANK", str(DATA / "genbank" / "genome.gbff"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from datasets import biochem, cds, load_dataset  # noqa: E402
from datasets.sequences import NT_CHANNELS_WITH_UNKNOWN, REFERENCE_IDS, reference_row  # noqa: E402
from models.locusfusion_v2 import _sinusoid  # noqa: E402
from training.checkpoint import load_model  # noqa: E402

DRUG = "ISONIAZID"
CELL = "lineage_split/sd19_all_modalities_unk__locusfusion_v2"
STEM = f"{DRUG}__dna+protein+biophysical+regulatory"
RESULT = PROJECT / "results" / "experiments" / CELL / f"{STEM}.json"
SPLIT = PROJECT / "results" / "experiments" / "lineage_split" / "splits" / f"{DRUG}.csv"
OUT = PROJECT / "results" / "figures" / "manuscript" / "forward_pass_trace.json"
AUC_TOL = 1e-3
BATCH = 128
CHUNK = 400
MAX_TOKENS = 30

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
net, cfg = load_model(CELL, STEM, weights_dir=WS / "model_weights_lineage")
assert net.unknown_symbols and cfg["best_fold"] == 1
D = cfg["data"]
record = json.loads(RESULT.read_text())
split = pd.read_csv(SPLIT, dtype={"New_ID": str})
test_ids = set(split.loc[split.role == "test", "New_ID"])

# --- reference sequences and token decoding ---------------------------------------
REF_NT = {g: reference_row(D["genotype_dir"], g) for g in net.loci}
REF_REG = {g: reference_row(D["regulatory_dir"], g) for g in net.loci}
REF_AA = {g: biochem.translate_reference_frame(REF_NT[g], REF_NT[g], cds.cds_columns(D["genotype_dir"], g))
          for g in net.loci if net.frames[g]["has_cds"]}
L = net.layout
NT = NT_CHANNELS_WITH_UNKNOWN
RES = [str(r) for r in list(biochem.RESIDUES_WITH_STOP) + ["-", "X"]]


def letter(v, alphabet):
    j = int(np.argmax(v))
    return alphabet[j] if v[j] > 0 else None


def decode(feat, rep, b, li, locus, keep_feat=True):
    s = L["slots"]
    out = []
    for t in range(1, feat.shape[2]):
        if not bool(rep["valid"][b, t - 1]):
            continue
        f = feat[b, li, t].numpy()
        region = "cds" if int(rep["region"][b, t - 1]) else "promoter"
        col = int(rep["column_in_block"][b, t - 1])
        codon = int(rep["codon"][b, t - 1])
        base = letter(f[slice(*s["dna"])], NT) or letter(f[slice(*s["regulatory"])], NT)
        ref = (REF_NT if region == "cds" else REF_REG)[locus][col]
        aa = letter(f[slice(*s["protein"])], RES)
        ref_aa = REF_AA[locus][codon] if (aa and locus in REF_AA and codon >= 0) else None
        tok = {"slot": t, "region": region, "column": col, "codon": codon,
               "coord_codons": float(rep["coord"][b, t - 1]), "ref": ref, "alt": base,
               "ref_aa": ref_aa, "alt_aa": aa,
               "phase": letter(f[L["f_phase"]:L["f_phase"] + 3], [0, 1, 2])}
        if keep_feat:
            tok["feat"] = [round(float(v), 4) for v in f]
        out.append(tok)
    return out


# --- load in chunks: a block is dense float32, ~3.4 MB per isolate -------------------
pheno = pd.read_csv(D["phenotype_csv"], dtype=str, keep_default_na=False, low_memory=False)
assert "New_ID" in pheno.columns


def filter_fastas(src, dst, wanted):
    """Copy src's FASTAs keeping the wanted isolates and every reference row. The
    alignments are untouched, so every column keeps its index and every block its width."""
    dst.mkdir()
    for f in sorted(Path(src).glob("*.fasta")):
        with f.open() as fin, (dst / f.name).open("w") as fout:
            on = False
            for line in fin:
                if line.startswith(">"):
                    rid = line[1:].split()[0]
                    on = rid in wanted or rid in REFERENCE_IDS or "H37Rv" in rid
                if on:
                    fout.write(line)


def load_ids(wanted, base_geno, base_reg, tmp):
    """load_dataset for these isolates, from filtered temporary copies."""
    sub = Path(tmp) / f"sub_{next(COUNTER)}"
    sub.mkdir()
    csv = sub / "phenotypes.csv"
    pheno[pheno.New_ID.isin(wanted)].to_csv(csv, index=False)
    filter_fastas(base_geno, sub / "geno", wanted)
    filter_fastas(base_reg, sub / "reg", wanted)
    d = load_dataset(DRUG, ["dna", "protein", "biophysical", "regulatory"], str(sub / "geno"), str(csv),
                     regulatory_dir=str(sub / "reg"), loci=D["loci_override"],
                     regulatory_loci=D["regulatory_loci_override"],
                     per_modality_branch=D["per_modality_branch"], extra_loci=D["extra_loci"],
                     all_regulatory=D["all_regulatory"], delta=True, protein_frame="reference",
                     locus_set=D.get("locus_set", "bigtb"), unknown_symbols=bool(D["unknown_symbols"]),
                     verbose=False)
    shutil.rmtree(sub)
    assert (d.y != -1).all(), "a TEST isolate has no label"
    assert [list(b.spec()) for b in d.blocks] == [list(s) for s in cfg["model"]["branch_specs"]], \
        "block shapes differ from the checkpoint's"
    return d


COUNTER = itertools.count()
n_loci = len(net.loci)
ids, y, P_S, ATT, N_TOK = [], [], [], [], []
best = None
TMP = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
base_geno, base_reg = Path(TMP.name) / "geno", Path(TMP.name) / "reg"
filter_fastas(D["genotype_dir"], base_geno, test_ids)
filter_fastas(D["regulatory_dir"], base_reg, test_ids)
ordered = sorted(test_ids)
for c in range(0, len(ordered), CHUNK):
    data = load_ids(set(ordered[c:c + CHUNK]), base_geno, base_reg, TMP.name)
    arrays = [b.array for b in data.blocks]
    chunk_ids = list(data.isolate_ids)
    for s in range(0, len(chunk_ids), BATCH):
        rows = np.arange(s, min(s + BATCH, len(chunk_ids)))
        xs = [torch.from_numpy(np.ascontiguousarray(a[rows])).float() for a in arrays]
        with torch.no_grad():
            logit, att = net(xs, return_attn=True)
            feat, _coord, _valid, stats = net._locus_tokens(xs)
            rep = net.variant_report(xs)
        P_S.append(torch.sigmoid(logit).numpy().ravel())
        ATT.append(att[:, 0].numpy())
        N_TOK.append(torch.expm1(stats[:, :, 0]).round().numpy())
        # choose the isolate by what it carries, never by the prediction
        for b, i in enumerate(rows):
            if data.y[i] != 0:                       # resistant isolates only
                continue
            n_tok = sum(int(r["valid"][b].sum()) for r in rep)
            if n_tok > MAX_TOKENS:                   # few enough rows to draw
                continue
            toks = [t for li, g in enumerate(net.loci) for t in
                    ({**x, "locus": g} for x in decode(feat, rep[li], b, li, g, keep_feat=False))]
            has = {"s315": any(t["locus"] == "katG" and t["codon"] == 314 and t["alt_aa"] not in (None, "S")
                               for t in toks),
                   "promoter": any(t["region"] == "promoter" and t["alt"] != "N" for t in toks),
                   "uncalled": any(t["alt"] == "N" for t in toks),
                   "synonymous": any(t["alt_aa"] and t["alt_aa"] == t["ref_aa"] and t["alt"] != "N"
                                     for t in toks)}
            score = 2 * has["s315"] + sum(has.values()) - 0.02 * n_tok
            if best is None or score > best[0]:
                best = (score, chunk_ids[i], has, n_tok)
    ids += chunk_ids
    y.append(data.y.copy())
    del data, arrays, xs
    gc.collect()
    print(f"{len(ids)}/{len(ordered)} scored; best so far {best[1] if best else None}", flush=True)

ids, y = np.asarray(ids, dtype=object), np.concatenate(y)
P_S, ATT, N_TOK = np.concatenate(P_S), np.concatenate(ATT), np.concatenate(N_TOK)
assert set(ids) == test_ids, f"{len(set(ids) ^ test_ids)} isolates differ from the split's TEST set"
auc = roc_auc_score(y, P_S)
print(f"rescored TEST AUC {auc:.5f}; recorded {record['test']['auc']:.5f}", flush=True)
assert abs(auc - record["test"]["auc"]) < AUC_TOL, "rebuilt model does not reproduce the recorded TEST AUC"

score, chosen, has, n_tok = best
assert has["s315"], "no held-out resistant isolate carries katG S315T under the token limit"
I = int(np.flatnonzero(ids == chosen)[0])
print(f"chose {chosen}: {has}, {n_tok} tokens; P(S)={P_S[I]:.4f}", flush=True)

# reload the chosen isolate alone: the replay below reads it as row 0
data = load_ids({chosen}, base_geno, base_reg, TMP.name)
arrays = [b.array for b in data.blocks]
assert list(data.isolate_ids) == [chosen]
TMP.cleanup()


def batch_of(rows):
    return [torch.from_numpy(np.ascontiguousarray(a[rows])).float() for a in arrays]


# --- the replay: every tensor of one forward pass --------------------------------
xs = batch_of(np.array([0]))


def encoder_with_attention(enc, x, pad):
    """nn.TransformerEncoder, one layer at a time, keeping each layer's attention."""
    maps = []
    for layer in enc.layers:
        assert layer.norm_first
        q = layer.norm1(x)
        _, w = layer.self_attn(q, q, q, key_padding_mask=pad, need_weights=True, average_attn_weights=True)
        maps.append(w)
        x = layer(x, src_key_padding_mask=pad)
    assert enc.norm is None
    return x, maps


with torch.no_grad():
    feat, coord, valid, stats = net._locus_tokens(xs)
    rep = net.variant_report(xs)
    B, n, T, _ = feat.shape
    li = torch.arange(n)
    sin = _sinusoid(coord, net.pos_dims)
    t_feat = net.tok_proj(feat)
    t_pos = net.pos_proj(sin)
    t_loc = net.locus_emb(li).unsqueeze(0).unsqueeze(2).expand_as(t_feat)
    t_sum = t_feat + t_pos + t_loc
    t_null_emb = net.null_emb(li).unsqueeze(0)
    t_null_stat = net.null_proj(stats)
    t_sum0 = t_sum.clone()
    t_sum0[:, :, 0] = t_sum0[:, :, 0] + t_null_emb + t_null_stat
    scale, shift = net.film_scale[li].unsqueeze(0).unsqueeze(2), net.film_shift[li].unsqueeze(0).unsqueeze(2)
    t_film = t_sum0 * (1.0 + scale) + shift
    tok = net.tok_norm(t_film) * valid.unsqueeze(-1).float()

    pad = ~valid
    z_all, att1 = encoder_with_attention(net.encoders[0], tok.reshape(B * n, T, net.d_model),
                                         pad.reshape(B * n, T))
    z_all = z_all.reshape(B, n, T, net.d_model)
    z = z_all[:, :, 0]
    z_norm = net.summ_norm(z, li)
    fused, att2 = encoder_with_attention(net.fusion, z_norm, torch.zeros(B, n, dtype=torch.bool))
    q = net.drug_queries.unsqueeze(0)
    pooled, att3 = net.pool_attn(q, fused, fused, key_padding_mask=torch.zeros(B, n, dtype=torch.bool),
                                 need_weights=True, average_attn_weights=True)
    pooled_norm = net.norm(pooled)
    h = torch.relu(net.fc1(pooled_norm))
    logit = net.fc_out(h).squeeze(-1)
    direct = net(xs)
assert torch.allclose(logit, direct, atol=1e-5), f"replay {logit.item()} != net(xs) {direct.item()}"
print(f"replay reproduces net(xs): logit {logit.item():.5f}", flush=True)


def vec(t, d=4):
    return np.round(t.detach().numpy().astype(float), d).tolist()


def nrm(t):
    return round(float(t.norm()), 3)


# --- the locus table: real widths for every block of every locus ------------------
frames_by_name = {b.name: (b.meta or {}) for b in data.blocks}
loci = []
blocks_by_locus = {g: {} for g in net.loci}
for k, (b, a) in enumerate(zip(data.blocks, arrays)):
    m, g = b.name.split(":", 1)
    occupied = int((np.abs(a[0]).sum(0) > 0).sum())
    blocks_by_locus[g][m] = {"name": b.name, "shape": [1, int(b.spec()[0]), int(b.spec()[1])], "occupied": occupied}
for lid, p in enumerate(net._plan):
    g = p["locus"]
    note = net.frames[g]
    ls = {"locus": g, "blocks": blocks_by_locus[g], "merged_columns": p["total"], "promoter_columns": p["reg_len"],
          "kept_columns": p["kept"], "dropped_duplicate_columns": note["duplicate_columns"],
          "n_codons": note["n_codons"], "has_cds": bool(note["has_cds"]), "cds_start": note["cds_start"],
          "promoter_offset_codons": note["promoter_offset"],
          "coord_min": float(getattr(net, f"coord_map_{lid}").min()),
          "coord_max": float(getattr(net, f"coord_map_{lid}").max()),
          "n_tokens_before_cap": int(round(float(torch.expm1(stats[0, lid, 0])))),
          "n_tokens": int(valid[0, lid, 1:].sum()),
          "stats": vec(stats[0, lid]),
          "tokens": decode(feat, rep[lid], 0, lid, g),
          "null_feat": [round(float(v), 4) for v in feat[0, lid, 0]],
          "attention_stage1": [vec(w[lid][: 1 + int(valid[0, lid, 1:].sum()), : 1 + int(valid[0, lid, 1:].sum())])
                               for w in att1],
          "null_output_norm": nrm(z[0, lid]), "summary_norm_output_norm": nrm(z_norm[0, lid])}
    loci.append(ls)

# the showcase token: katG S315T, each term of its embedding at full width
kid = net.loci.index("katG")
s315 = next(t for t in loci[kid]["tokens"] if t["codon"] == 314)
sl = s315["slot"]
showcase = {"slot": sl, "locus": "katG",
            "coord_codons": s315["coord_codons"], "sinusoid": vec(sin[0, kid, sl]),
            "tok_proj": vec(t_feat[0, kid, sl]), "pos_proj": vec(t_pos[0, kid, sl]),
            "locus_emb": vec(t_loc[0, kid, sl]), "sum": vec(t_sum0[0, kid, sl]),
            "film_scale": vec(scale[0, kid, 0]), "film_shift": vec(shift[0, kid, 0]),
            "film": vec(t_film[0, kid, sl]), "layernorm": vec(tok[0, kid, sl]),
            "stage1_output": vec(z_all[0, kid, sl])}
null_show = {"tok_proj": vec(t_feat[0, kid, 0]), "pos_proj": vec(t_pos[0, kid, 0]),
             "locus_emb": vec(t_loc[0, kid, 0]), "null_emb": vec(t_null_emb[0, kid]),
             "null_proj": vec(t_null_stat[0, kid]), "sum": vec(t_sum0[0, kid, 0]),
             "film": vec(t_film[0, kid, 0]), "layernorm": vec(tok[0, kid, 0]),
             "stage1_output": vec(z[0, kid])}

# --- parameters, by module ---------------------------------------------------------
groups = [("tok_proj", "tok_proj", "Linear 45→128"), ("pos_proj", "pos_proj", "Linear 32→128"),
          ("null_proj", "null_proj", "Linear 2→128"), ("locus_emb", "locus_emb", "19×128"),
          ("null_emb", "null_emb", "19×128"), ("tok_norm", "tok_norm", "LayerNorm 128"),
          ("film_scale", "film_scale", "19×128"), ("film_shift", "film_shift", "19×128"),
          ("encoders", "encoders", "2 layers, 4 heads, FF 256"), ("summ_norm", "summ_norm", "19×128 affine"),
          ("fusion", "fusion", "2 layers, 4 heads, FF 256"), ("drug_queries", "drug_queries", "1×128"),
          ("pool_attn", "pool_attn", "4 heads"), ("norm", "norm", "LayerNorm 128"),
          ("fc1", "fc1", "Linear 128→256"), ("fc_out", "fc_out", "Linear 256→1")]
params = [{"module": n, "shape": sh, "params": sum(p.numel() for k, p in net.named_parameters()
                                                   if k == pre or k.startswith(pre + "."))}
          for n, pre, sh in groups]
total = sum(p.numel() for p in net.parameters())
assert sum(p["params"] for p in params) == total == record["n_params"], (sum(p["params"] for p in params), total)

# --- the TEST cohort ---------------------------------------------------------------
res = y == 0
cohort = {"n": int(len(ids)), "n_resistant": int(res.sum()), "n_susceptible": int((~res).sum()),
          "auc_rescored": float(auc), "auc_recorded": float(record["test"]["auc"]),
          "threshold_P_S": float(record["test"]["threshold"]), "sens": float(record["test"]["sens"]),
          "spec": float(record["test"]["spec"]),
          "tokens_per_isolate_median": float(np.median(np.minimum(N_TOK, net.max_variants).sum(1))),
          "tokens_per_isolate_p5": float(np.percentile(np.minimum(N_TOK, net.max_variants).sum(1), 5)),
          "tokens_per_isolate_p95": float(np.percentile(np.minimum(N_TOK, net.max_variants).sum(1), 95)),
          "loci_over_cap": {g: int((N_TOK[:, j] > net.max_variants).sum()) for j, g in enumerate(net.loci)},
          "isolates_any_locus_over_cap": int((N_TOK > net.max_variants).any(1).sum()),
          "tokens_median_by_locus": {g: float(np.median(N_TOK[:, j])) for j, g in enumerate(net.loci)},
          "query_attention_mean_resistant": dict(zip(net.loci, np.round(ATT[res].mean(0).astype(float), 4).tolist())),
          "query_attention_mean_susceptible": dict(zip(net.loci, np.round(ATT[~res].mean(0).astype(float), 4).tolist()))}

OUT.write_text(json.dumps({
    "cell": CELL, "stem": STEM, "fold": cfg["best_fold"], "best_epoch": cfg["folds"][cfg["best_fold"]]["best_epoch"],
    "drug": DRUG, "isolate": str(ids[I]), "isolate_role": "test", "isolate_group": str(split.set_index("New_ID").group[ids[I]]),
    "y": int(y[I]), "has": has, "n_params": total, "params": params, "cohort": cohort,
    "layout": {"slots": {k: list(v) for k, v in L["slots"].items()}, "d_bio": list(L["d_bio"]),
               "f_is_null": L["f_is_null"], "f_phase": L["f_phase"], "c_tok": L["c_tok"]},
    "alphabets": {"nt": NT, "protein": RES}, "loci_order": net.loci, "n_blocks": len(arrays),
    "shapes": {"feat": list(feat.shape), "coord": list(coord.shape), "valid": list(valid.shape), "stats": list(stats.shape),
               "sinusoid": list(sin.shape), "tok": list(tok.shape), "stage1_in": [B * n, T, net.d_model],
               "z": list(z.shape), "fused": list(fused.shape), "query": list(q.shape), "pooled": list(pooled.shape),
               "h": list(h.shape), "logit": list(logit.shape)},
    "valid": valid[0].numpy().astype(int).tolist(), "loci": loci, "showcase": showcase, "null_show": null_show,
    "z": vec(z[0]), "z_norm": vec(z_norm[0]), "fused": vec(fused[0]),
    "attention_stage2": [vec(w[0]) for w in att2], "attention_query": vec(att3[0, 0]),
    "query": vec(q[0, 0]), "pooled": vec(pooled[0, 0]), "pooled_norm": vec(pooled_norm[0, 0]), "h": vec(h[0, 0]),
    "h_active": int((h[0, 0] > 0).sum()), "logit": float(logit), "P_S": float(torch.sigmoid(logit)),
    "P_R": float(1 - torch.sigmoid(logit)), "direct_logit": float(direct)}, indent=None))
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

"""Real numbers for the GenomeFusion method figures: whole genome -> tokens, and the limits.

    python scripts/genomefusion_method_extract.py      # no model inference; ~2 min, ~1 GB

Writes results/figures/genomefusion/method_trace.json, which
scripts/genomefusion_method_figures.py draws. Reads the corrected partition
(genomefusion_cache_v3), the token store the trainer reads, and 100 seeded VCFs plus
the example isolate's, run through the project's own `tokenize_isolate`.

Fails rather than writes if a stored token does not map to its locus under this
partition, or if re-tokenizing the example's VCF does not reproduce its stored tokens.

The example isolate is a held-out ISONIAZID TEST isolate chosen for what it carries
(katG S315T, fabG1 c.-15C>T, a frameshift, a deletion run, an uncalled base), and is the
plainest one with all of them. It is never chosen on a prediction.
"""
import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from genomefusion.build_tokens import VCF_DIR, VCF_SUFFIX  # noqa: E402
from genomefusion.data import TOKEN_FIELDS, BatchBuilder  # noqa: E402
from genomefusion.layout import (C_BASE, CONSEQUENCES, FRAMESHIFT, NT_SYMBOLS, REGIONS,  # noqa: E402
                                 RESIDUE_SYMBOLS, SLOTS, SYN, channel_names)
from genomefusion.model import BUCKETS, GF_DEFAULTS, GenomeFusionNet  # noqa: E402
from genomefusion.partition import CACHE, WORKSPACE, load  # noqa: E402
from genomefusion.reference import load_reference  # noqa: E402
from genomefusion.tokenizer import allele_category, parse_info, tokenize_isolate  # noqa: E402

OUT = PROJECT / "results" / "figures" / "genomefusion" / "method_trace.json"
SPLIT = PROJECT / "results" / "experiments" / "lineage_split" / "splits" / "ISONIAZID.csv"
TOKENS = WORKSPACE / "genomefusion_tokens" / "tokens.npz"
CACHE_V3 = WORKSPACE / "genomefusion_cache_v3"
KATG_S315, FABG1_C15 = 2155167, 1673424           # 0-based genome positions
N_SAMPLE, SEED = 100, 0
BIN = 5000
LOCUS_BUDGET, BATCH = 32768, 32
WINDOWS = {"fabG1-inhA": (1672900, 1675300), "katG": (2153300, 2157000)}


def name(L):
    return L.gene or L.name


def vcf_funnel(path, partition, ref, ref_arr):
    """One isolate through every filter, with the project's own functions."""
    raw = symbolic = cat_ref = cat_missing = cat_alt = long_missing = imprecise = hets = 0
    op = gzip.open if str(path).endswith((".gz", ".bgz")) else open
    with op(path, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            raw += 1
            r, a, qual, filt, info_s = f[3].upper(), f[4].upper(), f[5], f[6], f[7]
            if a.startswith("<"):
                symbolic += 1
                continue
            a = a.split(",")[0] if "," in a else a
            info = parse_info(info_s)
            q = None if qual in (".", "") else float(qual)
            c = allele_category(r, a, q, filt, info)
            imprecise += "IMPRECISE" in info
            if c == "ref":
                cat_ref += 1
            elif c == "missing":
                cat_missing += 1
                if abs(len(a) - len(r)) > 100:
                    long_missing += 1
                af = info.get("AF")
                if af and "," not in af and 0.25 < float(af) <= 0.75:
                    hets += 1
            else:
                cat_alt += 1
    keep = tokenize_isolate(path, partition, ref, drop_synonymous=False, ref_arr=ref_arr)
    drop = tokenize_isolate(path, partition, ref, drop_synonymous=True, ref_arr=ref_arr)
    return dict(records=raw, symbolic_alt=symbolic, qc_left_as_reference=cat_ref,
                called_alt=cat_alt, called_uncalled=cat_missing, long_uncalled_indel=long_missing,
                intermediate_af=hets, imprecise=imprecise,
                events=keep.stats["n_events"], masked=keep.stats["n_masked"],
                placed=len(keep.locus), synonymous=int((keep.consequence == SYN).sum()),
                tokens=len(drop.locus), active_loci=len(drop.sum_locus)), drop


def main():
    P = load(CACHE_V3 / "partition.npz")
    ref = load_reference(CACHE)
    ref_arr = np.frombuffer(ref, np.uint8)
    L = P.loci
    assert sum(x.is_19 for x in L) == 19 and any(x.is_19 and x.gene == "tlyA" for x in L)

    z = np.load(TOKENS, allow_pickle=False)
    ids = [str(x) for x in z["ids"]]
    tok_off, sum_off = z["tok_off"], z["sum_off"]
    T = {k: z[k] for k in ("locus", "pos", "base", "consequence", "region", "span", "flags")}
    sum_n, sum_locus = z["sum_n"], z["sum_locus"]
    n_iso, n_tok_all = len(ids), len(T["pos"])
    print(f"{n_iso:,} isolates, {n_tok_all:,} tokens", flush=True)

    # ---- the load-bearing check: the store and this partition agree on every token ----
    bad = int((P.locus_of[T["pos"]] != T["locus"]).sum())
    assert bad == 0, f"{bad:,} tokens map to a different locus under cache_v3"
    print("every stored token maps to its locus under cache_v3", flush=True)

    # ---- the example isolate -------------------------------------------------------
    split = pd.read_csv(SPLIT, dtype={"New_ID": str})
    test = set(split.loc[split.role == "test", "New_ID"])
    tok_iso = np.repeat(np.arange(n_iso), np.diff(tok_off))
    NB = NT_SYMBOLS.index
    has = lambda mask: np.bincount(tok_iso[mask], minlength=n_iso) > 0   # noqa: E731
    want = (has((T["pos"] == KATG_S315) & (T["span"] == 1))
            & has((T["pos"] == FABG1_C15) & (T["span"] == 1))
            & has(T["consequence"] == FRAMESHIFT)
            & has((T["span"] >= 10) & (T["base"] == NB("-")))
            & has(T["base"] == NB("N")))
    cand = [i for i in np.flatnonzero(want) if ids[i] in test]
    assert cand, "no TEST isolate carries all five token kinds"
    counts = np.diff(tok_off)
    I = min(cand, key=lambda i: counts[i])
    sid = ids[I]
    print(f"example {sid}: {counts[I]} tokens, {len(cand)} candidates", flush=True)
    a, b = tok_off[I], tok_off[I + 1]
    rows = np.arange(a, b)

    # ---- one isolate, every filter; then the sample ------------------------------------
    funnel, tk = vcf_funnel(VCF_DIR / f"{sid}{VCF_SUFFIX}", P, ref, ref_arr)
    assert funnel["tokens"] == counts[I], (funnel["tokens"], int(counts[I]))
    assert np.array_equal(tk.pos, T["pos"][rows]), "re-tokenizing does not reproduce the store"
    rng = np.random.default_rng(SEED)
    sample, failed = [], 0
    for j in rng.choice(n_iso, N_SAMPLE, replace=False):
        try:
            sample.append(vcf_funnel(VCF_DIR / f"{ids[j]}{VCF_SUFFIX}", P, ref, ref_arr)[0])
        except OSError:
            failed += 1
    assert not failed, f"{failed} VCFs unreadable"
    S = pd.DataFrame(sample)
    print(f"funnel sample n={len(S)}", flush=True)

    # ---- the example's tokens, channels from the trainer's own expansion -----------------
    Tfull = {k: z[k][rows] for k in TOKEN_FIELDS}
    start = np.asarray([x.start for x in L], np.int64)
    exp = SimpleNamespace(store=SimpleNamespace(tok=Tfull), c_tok=C_BASE, locus_start=start)
    feat, coord, _ = BatchBuilder._expand(exp, np.arange(len(rows)), None)
    loc = Tfull["locus"]
    n_in = np.bincount(loc, minlength=len(L))

    def describe(i):
        x = L[int(loc[i])]
        c = int(Tfull["consequence"][i])
        r, rr = int(Tfull["residue"][i]), int(Tfull["ref_residue"][i])
        return dict(i=int(i), locus=name(x), locus_tag=x.name, locus_kind=x.kind, strand=x.strand,
                    pos=int(Tfull["pos"][i]), coord=float(coord[i]), locus_tokens=int(n_in[loc[i]]),
                    locus_start=int(x.start), locus_end=int(x.end),
                    ref=chr(ref_arr[int(Tfull["pos"][i])]), base=NT_SYMBOLS[int(Tfull["base"][i])],
                    span=int(Tfull["span"][i]), codon=int(Tfull["codon"][i]),
                    residue=RESIDUE_SYMBOLS[r] if r >= 0 else None,
                    ref_residue=RESIDUE_SYMBOLS[rr] if rr >= 0 else None,
                    consequence=CONSEQUENCES[c] if c >= 0 else None,
                    truncation=round(float(Tfull["truncation"][i]), 3),
                    up_plus=int(Tfull["up_plus"][i]), up_minus=int(Tfull["up_minus"][i]),
                    region=REGIONS[int(Tfull["region"][i])], flags=int(Tfull["flags"][i]),
                    feat=[round(float(v), 3) for v in feat[i]])

    picks = {}

    def pick(label, mask):
        idx = np.flatnonzero(mask)
        if len(idx):
            picks[label] = int(idx[0])

    B, C_, S_, P_ = Tfull["base"], Tfull["consequence"], Tfull["span"], Tfull["pos"]
    pick("missense", P_ == KATG_S315)
    pick("intergenic", P_ == FABG1_C15)
    pick("frameshift", C_ == FRAMESHIFT)
    pick("deletion run", (S_ >= 10) & (B == NB("-")))
    pick("uncalled", (B == NB("N")) & (S_ == 1))
    pick("nonsense", C_ == CONSEQUENCES.index("nonsense"))
    pick("inframe indel", C_ == CONSEQUENCES.index("inframe_indel"))
    pick("RNA gene", Tfull["region"] == REGIONS.index("rna"))
    ref_nt_ex = np.array([NT_SYMBOLS.index(chr(c)) if chr(c) in NT_SYMBOLS else -1 for c in ref_arr[P_]])
    pick("insertion", B == ref_nt_ex)
    pick("overlap", (Tfull["flags"] & 1) > 0)
    chosen = [dict(kind=k, **describe(i)) for k, i in picks.items()]

    pair_n = z["sum_n"][sum_off[I]:sum_off[I + 1]]
    pair_l = z["sum_locus"][sum_off[I]:sum_off[I + 1]]
    hist = {}
    lo = 0
    for hi in BUCKETS:
        hist[str(hi)] = int(((pair_n > lo) & (pair_n <= hi)).sum())
        lo = hi
    busiest = np.argsort(pair_n)[::-1][:5]
    # the [NULL] token's four inputs, exactly as BatchBuilder computes them, for the loci drawn
    span_l = np.maximum(np.asarray([x.end - x.start for x in L], np.float32), 1.0)
    sf, so = z["sum_frame"][sum_off[I]:sum_off[I + 1]], z["sum_oof"][sum_off[I]:sum_off[I + 1]]
    show = list(busiest[:2]) + [int(np.flatnonzero(pair_l == loc[picks[k]])[0]) for k in ("missense", "intergenic", "frameshift")]
    null_examples = [dict(locus=name(L[int(pair_l[i])]), n=int(pair_n[i]), log1p=round(float(np.log1p(pair_n[i])), 4),
                          density=round(float(pair_n[i] / span_l[pair_l[i]]), 5),
                          frame=round(float(sf[i] / 3.0), 4), oof=round(float(so[i]), 4)) for i in show]
    example = dict(
        null_examples=null_examples, isolate=sid, role="ISONIAZID test", group=str(split.set_index("New_ID").group[sid]),
        phenotype_R=int(split.set_index("New_ID").R[sid]),
        n_tokens=int(counts[I]), n_active=int(len(pair_n)), lmax=int(len(pair_n)) + 1,
        funnel=funnel, tokens=chosen, buckets=hist,
        busiest=[dict(locus=name(L[int(pair_l[i])]), n=int(pair_n[i])) for i in busiest],
        tok_pos=[int(x) for x in P_], tok_locus=[int(x) for x in loc],
        tok_cons=[int(x) for x in C_], tok_base=[int(x) for x in B], tok_span=[int(x) for x in S_],
        region_counts={r: int((Tfull["region"] == i).sum()) for i, r in enumerate(REGIONS)})

    # ---- the genome: partition, by base -----------------------------------------------
    glen = P.genome_len
    masked = P.locus_of < 0
    kinds = np.array([{"cds": 0, "rna": 1, "intergenic": 2}[x.kind] for x in L])
    kind_of = np.where(masked, 3, kinds[np.clip(P.locus_of, 0, None)])
    edges = np.arange(0, glen + BIN, BIN)
    bins = (edges[:-1] + edges[1:]) // 2
    idx = np.minimum(np.arange(glen) // BIN, len(bins) - 1)
    masked_bin = np.bincount(idx, weights=masked, minlength=len(bins)) / np.bincount(idx, minlength=len(bins))
    all_bin = np.bincount(np.minimum(T["pos"] // BIN, len(bins) - 1), minlength=len(bins)) / n_iso
    ex_bin = np.bincount(np.minimum(P_ // BIN, len(bins) - 1), minlength=len(bins))
    genome = dict(length=int(glen), n_loci=len(L),
                  bases={k: int((kind_of == i).sum()) for i, k in enumerate(("cds", "rna", "intergenic", "masked"))},
                  loci={k: int((kinds == i).sum()) for i, k in enumerate(("cds", "rna", "intergenic"))},
                  meta={k: P.meta[k] for k in ("masked_bases", "masked_frac", "rlc_regions", "masked_category_genes")},
                  overlap_bases=int(P.overlap.sum()),
                  locus_bases_median=float(np.median([x.n_bases for x in L])),
                  locus_bases_max=int(max(x.n_bases for x in L)),
                  bin=BIN, masked_bin=np.round(masked_bin, 3).tolist(),
                  tokens_mean_bin=np.round(all_bin, 3).tolist(), tokens_example_bin=ex_bin.astype(int).tolist(),
                  n_regulatory=int(sum(x.is_regulatory for x in L)),
                  n_edges=int((P.regulated_index() >= 0).sum()),
                  sets={k: int(v) for k, v in [("19", 19)]})

    def window(lo, hi):
        out = []
        for x in L:
            if x.end > lo and x.start < hi:
                out.append(dict(name=name(x), tag=x.name, kind=x.kind, start=int(x.start), end=int(x.end),
                                strand=x.strand, is_19=bool(x.is_19), is_regulatory=bool(x.is_regulatory),
                                regulates=list(x.regulates), n_bases=int(x.n_bases), index=int(x.index)))
        ri = P.regulated_index()
        for o in out:
            o["edge_to"] = name(L[int(ri[o["index"]])]) if ri[o["index"]] >= 0 else None
        sel = (P_ >= lo) & (P_ < hi)
        return dict(lo=lo, hi=hi, loci=out,
                    masked=[[int(s), int(e)] for s, e in _runs(masked[lo:hi], lo)],
                    tokens=[dict(pos=int(p), kind=int(c), base=NT_SYMBOLS[int(bb)], span=int(sp))
                            for p, c, bb, sp in zip(P_[sel], C_[sel], B[sel], S_[sel])])

    windows = {k: window(*v) for k, v in WINDOWS.items()}

    # ---- the cohort -------------------------------------------------------------------
    n_active = np.diff(sum_off)
    n_tokens = np.diff(tok_off)
    ref_nt = np.array([NT_SYMBOLS.index(chr(c)) if chr(c) in NT_SYMBOLS else -1 for c in range(256)], np.int8)
    is_ins = T["base"] == ref_nt[ref_arr[T["pos"]]]
    cons = T["consequence"]
    per_locus = np.bincount(sum_locus, minlength=len(L))
    tok_locus = np.bincount(T["locus"], minlength=len(L))
    top_pairs = np.argsort(sum_n)[::-1][:1]
    budget_batch = np.clip(LOCUS_BUDGET // (n_active + 1), 1, BATCH)
    pct = lambda a, q: float(np.percentile(a, q))    # noqa: E731
    h_tok, e_tok = np.histogram(n_tokens, bins=60)
    h_act, e_act = np.histogram(n_active, bins=60)
    bucket_all = {}
    lo = 0
    for hi in BUCKETS:
        bucket_all[str(hi)] = int(((sum_n > lo) & (sum_n <= hi)).sum())
        lo = hi
    bucket_all[">512"] = int((sum_n > BUCKETS[-1]).sum())
    busy = np.argsort(tok_locus)[::-1][:12]
    stats = json.loads(str(z["stats"]))
    masked_mean = float(np.average([s["masked_events_mean"] for s in stats["shards"]],
                                   weights=[s["n_isolates"] for s in stats["shards"]]))
    cohort = dict(
        n=n_iso, tokens_total=n_tok_all, pairs_total=int(len(sum_n)),
        zero_token_isolates=int((n_tokens == 0).sum()),
        pairs_over_16=int((sum_n > 16).sum()),
        tokens=dict(min=int(n_tokens.min()), p5=pct(n_tokens, 5), median=pct(n_tokens, 50),
                    mean=float(n_tokens.mean()), p95=pct(n_tokens, 95), max=int(n_tokens.max())),
        active=dict(min=int(n_active.min()), p5=pct(n_active, 5), median=pct(n_active, 50),
                    mean=float(n_active.mean()), p95=pct(n_active, 95), max=int(n_active.max())),
        hist_tokens=[h_tok.tolist(), e_tok.tolist()], hist_active=[h_act.tolist(), e_act.tolist()],
        pair_buckets=bucket_all, pair_tokens_max=int(sum_n.max()),
        pair_tokens_max_locus=name(L[int(sum_locus[top_pairs[0]])]),
        pair_single_frac=float((sum_n == 1).mean()),
        consequence={("none (off a CDS)" if i < 0 else CONSEQUENCES[i]): int((cons == i).sum()) for i in range(-1, 8)},
        region={r: int((T["region"] == i).sum()) for i, r in enumerate(REGIONS)},
        uncalled_tokens=int((T["base"] == NB("N")).sum()), deletion_tokens=int((T["base"] == NB("-")).sum()),
        insertion_tokens=int(is_ins.sum()), overlap_tokens=int(((T["flags"] & 1) > 0).sum()),
        homopolymer_tokens=int(((T["flags"] & 2) > 0).sum()), imprecise_tokens=int(((T["flags"] & 4) > 0).sum()),
        span_max=int(T["span"].max()), span_ge10=int((T["span"] >= 10).sum()),
        masked_events_mean=masked_mean,
        loci_never_active=int((per_locus == 0).sum()), loci_active_lt10=int((per_locus < 10).sum()),
        loci_active_ge_half=int((per_locus >= n_iso / 2).sum()),
        busiest_loci=[dict(locus=name(L[int(i)]), tokens=int(tok_locus[i]), isolates=int(per_locus[i])) for i in busy],
        batch=dict(budget=LOCUS_BUDGET, requested=BATCH, isolates_below_requested=int((budget_batch < BATCH).sum()),
                   min_batch=int(budget_batch.min())))
    # how much of the active-locus count is lineage: one-way eta squared, top-level lineage
    lin = pd.read_csv(WORKSPACE / "data_snapshot" / "lineage" / "lineage_calls.csv", dtype={"New_ID": str})
    D = pd.DataFrame({"New_ID": ids, "active": n_active}).merge(lin[["New_ID", "lineage"]], on="New_ID", how="left")
    D["top"] = D.lineage.astype("string").str.extract(r"(lineage\d+|La\d)", expand=False)
    D = D[D.top.notna()]
    fitted = D.groupby("top").active.transform("mean")
    cohort["lineage_eta2"] = float(((fitted - D.active.mean()) ** 2).sum() / ((D.active - D.active.mean()) ** 2).sum())
    cohort["lineage_n"] = int(len(D))
    cohort["active_median_by_lineage"] = {k: float(v) for k, v in D.groupby("top").active.median().items()}
    # are the channels ever used?
    used = {}
    for nm, (lo_, hi_) in SLOTS.items():
        used[nm] = [lo_, hi_]
    sample_rows = rng.choice(n_tok_all, min(2_000_000, n_tok_all), replace=False)
    Ts = {k: z[k][sample_rows] for k in TOKEN_FIELDS}
    exp2 = SimpleNamespace(store=SimpleNamespace(tok=Ts), c_tok=C_BASE, locus_start=start)
    f2 = BatchBuilder._expand(exp2, np.arange(len(sample_rows)), None)[0]
    nm = channel_names()
    never = [nm[j] for j in range(C_BASE) if not np.any(f2[:, j] != 0)]
    print("channels never nonzero in 2M sampled tokens:", never, flush=True)

    # ---- parameters, from the real class -------------------------------------------------
    feats = P.feature_matrix(with_regulated=True)
    net = GenomeFusionNet(feats, ["ISONIAZID"], C_BASE, regulated_lid=P.regulated_index(), **GF_DEFAULTS)
    from collections import OrderedDict
    groups = OrderedDict()
    for k, p_ in net.named_parameters():
        groups[k.split(".")[0]] = groups.get(k.split(".")[0], 0) + p_.numel()
    n_params = sum(groups.values())
    res = json.loads((PROJECT / "results/experiments/genomefusion/superseded_ahpC_20261106/size6776/ISONIAZID.json").read_text())
    assert n_params == res["n_params"], (n_params, res["n_params"])

    # ---- known resistance genes the partition cannot see --------------------------------
    from datasets.loader import drug_loci
    names = {x.gene for x in L} | {x.name for x in L}
    missing = {}
    for d in ("AMIKACIN", "CAPREOMYCIN", "ETHAMBUTOL", "ETHIONAMIDE", "ISONIAZID", "KANAMYCIN",
              "LEVOFLOXACIN", "MOXIFLOXACIN", "PYRAZINAMIDE", "RIFAMPICIN", "STREPTOMYCIN"):
        gone = [g for g in drug_loci(d, locus_set="who_t12") if g not in names]
        if gone:
            missing[d] = gone

    OUT.write_text(json.dumps(dict(
        genome=genome, example=example, windows=windows, cohort=cohort,
        funnel_sample={c: dict(p5=pct(S[c], 5), median=pct(S[c], 50), p95=pct(S[c], 95)) for c in S.columns},
        funnel_sample_n=len(S), channels=nm, slots=used, never_nonzero=never,
        params=dict(total=n_params, groups=groups, feat_width=int(feats.shape[1]),
                    locus_residual_params=int(groups["locus_residual"])),
        who_t12_not_in_partition=missing, buckets=list(BUCKETS),
        protocol=res["protocol"]), indent=None))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


def _runs(mask, offset):
    out, on, s = [], False, 0
    for i, m in enumerate(mask):
        if m and not on:
            on, s = True, i
        elif not m and on:
            on = False
            out.append((s + offset, i + offset))
    if on:
        out.append((s + offset, len(mask) + offset))
    return out


if __name__ == "__main__":
    main()
